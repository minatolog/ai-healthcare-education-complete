"""Explicit full-volume manifests; original selectively retained PNGs are not inputs."""
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import PreprocessConfig, preprocess_mask, preprocess_slice, validate_volume

SPLIT_COUNTS = {"train": 20, "validation": 10, "test": 10}
AUDIT_FLAGS = ("image_mask_pairing_complete", "all_axial_slices_available",
               "hu_and_geometry_verified", "label_semantics_verified", "prediction_pool_independent")


@dataclass
class PatientVolume:
    patient_id: str
    source_patient_id: str
    split: str
    volume: np.ndarray
    mask: object = None


def validate_pair(patient):
    validate_volume(patient.volume)
    if patient.mask is None or patient.mask.shape != patient.volume.shape:
        raise ValueError("A shape-matched full-volume mask is required for supervised data")
    if not np.isin(patient.mask, [0, 1, 2]).all():
        raise ValueError("Full mask labels must be 0=background, 1=kidney, 2=tumour")


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def volume_fingerprint(volume):
    normalized = np.ascontiguousarray(volume, dtype=np.float32)
    return hashlib.sha256(str(normalized.shape).encode() + normalized.tobytes()).hexdigest()


def _path(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Manifest asset is missing or is outside the approved data root")
    if Path(str(path) + ".aria2").exists():
        raise ValueError("Manifest asset is still downloading")
    return path


def read_array(path, slice_axis=0):
    path = Path(path)
    if type(slice_axis) is not int or slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must explicitly identify the verified axial array axis")
    affine = None
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.name.endswith((".nii", ".nii.gz")):
        image = nib.load(path)
        array = np.asanyarray(image.dataobj)
        affine = image.affine
    else:
        raise ValueError("Expected a full-volume HU NIfTI or npy file; selective PNG data is unsupported")
    if array.ndim != 3:
        raise ValueError("Expected a three-dimensional volume")
    # Preserve the reviewed KiTS axis convention; no implicit canonical reorientation.
    return np.moveaxis(array, slice_axis, 0), affine


def load_manifest(path, data_root=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("task") != "kidney_tumour_segmentation":
        raise ValueError("Expected version-1 kidney_tumour_segmentation manifest")
    if manifest.get("units") != "HU" or manifest.get("full_volume_coverage") is not True:
        raise ValueError("Real inputs require verified HU and full-volume coverage")
    if manifest.get("data_mode") != "real":
        raise ValueError("Real-data CLI cannot train synthetic manifests")
    rows = manifest.get("patients", [])
    if not rows:
        raise ValueError("No patient records supplied")
    for key in ("patient_id", "source_patient_id"):
        ids = [row.get(key) for row in rows]
        if any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids):
            raise ValueError("Patient and source identities must be non-empty and globally unique")
    counts = {split: sum(row.get("split") == split for row in rows) for split in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS or not any(row.get("split") == "prediction" for row in rows):
        raise ValueError("Requires canonical 20/10/10 patients plus separate prediction-only patient(s)")
    if any(row.get("split") not in {*SPLIT_COUNTS, "prediction"} for row in rows):
        raise ValueError("Unknown split")
    root = Path(data_root).resolve() if data_root else path.parent
    for row in rows:
        if type(row.get("slice_axis")) is not int or row.get("slice_axis") not in (0, 1, 2):
            raise ValueError("Every patient needs an audited slice_axis")
        _path(root, row["image"])
        if row["split"] != "prediction":
            _path(root, row["mask"])
    return manifest, root


def load_patient(row, root, supervised=True):
    volume, image_affine = read_array(_path(root, row["image"]), row["slice_axis"])
    validate_volume(volume)
    mask = None
    if supervised:
        mask, mask_affine = read_array(_path(root, row["mask"]), row["slice_axis"])
        if (image_affine is None) != (mask_affine is None):
            raise ValueError("Mixed mask/image geometry formats require a separately audited conversion")
        if image_affine is not None and not np.allclose(image_affine, mask_affine, rtol=1e-5, atol=1e-5):
            raise ValueError("Image and segmentation affines do not match")
    patient = PatientVolume(row["patient_id"], row["source_patient_id"], row["split"], volume, mask)
    if supervised:
        validate_pair(patient)
    return patient


def validate_real_training(path, audit_path, data_root=None):
    manifest, root = load_manifest(path, data_root)
    audit = json.loads(Path(audit_path).read_text())
    if audit.get("status") != "passed" or audit.get("manifest_sha256") != manifest_hash(path):
        raise ValueError("Requires a passed audit bound to this exact canonical manifest")
    if any(audit.get(flag) is not True for flag in AUDIT_FLAGS):
        raise ValueError("Incomplete data/geometry/label/prediction-pool audit; real training is blocked")
    expected_ids = {row["patient_id"] for row in manifest["patients"]}
    if set(audit.get("verified_patient_ids", [])) != expected_ids:
        raise ValueError("Audit does not cover every canonical and prediction patient")
    assets = {row["image"] for row in manifest["patients"]}
    assets.update(row["mask"] for row in manifest["patients"] if row["split"] != "prediction")
    checksums = audit.get("asset_sha256", {})
    if set(checksums) != assets or any(file_hash(_path(root, asset)) != checksums[asset] for asset in assets):
        raise ValueError("Audit asset checksums are missing or files have changed")
    fingerprints = set()
    for row in manifest["patients"]:
        patient = load_patient(row, root, supervised=row["split"] != "prediction")
        digest = volume_fingerprint(patient.volume)
        if digest in fingerprints:
            raise ValueError("Duplicate full volume across patient identities/splits")
        fingerprints.add(digest)
        row["volume_fingerprint"] = digest
    return manifest, root


class FullSliceDataset(Dataset):
    """All axial slices; one-patient cache avoids holding the full cohort in RAM."""
    def __init__(self, rows, root=None, config=PreprocessConfig()):
        self.rows, self.root, self.config = rows, root, config
        self.cache = None
        self.ends = []
        self.identities = []
        for row in rows:
            patient = row if isinstance(row, PatientVolume) else load_patient(row, root)
            validate_pair(patient)
            self.ends.append((self.ends[-1] if self.ends else 0) + len(patient.volume))
            self.identities.append((patient.patient_id, patient.source_patient_id, patient.split))
        if not self.ends:
            raise ValueError("Dataset must contain patients")
        for index in (0, 1):
            ids = [patient[index] for patient in self.identities]
            if any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids):
                raise ValueError("Dataset patients must have non-empty unique patient and source identities")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        patient_index = bisect_right(self.ends, index)
        if self.cache is None or self.cache[0] != patient_index:
            row = self.rows[patient_index]
            patient = row if isinstance(row, PatientVolume) else load_patient(row, self.root)
            self.cache = patient_index, patient
        patient = self.cache[1]
        z = index - (self.ends[patient_index - 1] if patient_index else 0)
        image, neighbours = preprocess_slice(patient.volume, z, self.config)
        return {"image": image, "mask": preprocess_mask(patient.mask[z], self.config),
                "patient_id": patient.patient_id, "source_patient_id": patient.source_patient_id,
                "split": patient.split, "slice_index": z, "context_indices": torch.tensor(neighbours)}


def synthetic_patients(seed=42, n_slices=3):
    """40+1 engineering fixtures. Shapes/bright regions are not medical examples."""
    rng = np.random.default_rng(seed)
    patients = []
    for split, count in [("train", 20), ("validation", 10), ("test", 10), ("prediction", 1)]:
        for index in range(count):
            volume = rng.normal(30, 15, (n_slices, 32, 32)).astype(np.float32)
            mask = np.zeros(volume.shape, dtype=np.uint8)
            mask[:, 8:24, 8:24] = 1
            mask[:, 14:20, 14:20] = 2
            volume[mask == 1] += 30
            volume[mask == 2] += 70
            identity = "synthetic-%s-%02d" % (split, index)
            patients.append(PatientVolume(identity, identity, split, volume, None if split == "prediction" else mask))
    return patients
