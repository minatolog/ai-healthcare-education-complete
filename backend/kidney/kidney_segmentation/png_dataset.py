"""Single-channel 2D PNG loading with explicit patient selection.

This adapter reads already processed PNG intensities, not HU or neighbouring
slices. It does not select a cohort, read splits.json, augment data, or train a
model. Constructing it does not establish a complete dataset audit: PNG pixels
and reference-mask metadata are checked only when each sample is requested.
"""
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PNGConfig:
    """The reviewed grayscale normalization, without another HU-window step."""
    input_channels: int = 1
    size: int = 256
    mean: float = 0.456
    std: float = 0.224
    version: str = "png-single-slice-gray8-normalized-v1"

    def __post_init__(self):
        if type(self.input_channels) is not int or self.input_channels != 1:
            raise ValueError("This pure 2D adapter supports exactly one input channel.")
        if type(self.size) is not int or self.size != 256:
            raise ValueError("This PNG adapter expects existing 256x256 images without resizing.")
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError("Normalization requires a finite mean and positive finite standard deviation.")
        if self.version != "png-single-slice-gray8-normalized-v1":
            raise ValueError("Unknown PNG preprocessing version.")

    def to_dict(self):
        return asdict(self)


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _integer(row, key):
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Manifest field '%s' must contain an integer." % key) from None
    if value < 0:
        raise ValueError("Manifest field '%s' must not be negative." % key)
    return value


def _asset_path(root, relative, folder, patient_id, slice_index):
    if not isinstance(relative, str) or not relative:
        raise ValueError("A manifest PNG path is missing.")
    relative = Path(relative)
    expected = "%s_z%03d.png" % (patient_id, slice_index)
    if relative.is_absolute() or relative.parts != (folder, expected):
        raise ValueError("Expected the selected %s path to be %s/%s." % (folder, folder, expected))
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("A PNG path resolves outside the approved data root.")
    return path


def _read_png(path, size):
    if not path.is_file():
        raise ValueError("The requested PNG file is missing: %s" % path)
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (size, size):
            raise ValueError("Expected an 8-bit grayscale L-mode %dx%d PNG: %s" % (size, size, path.name))
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated PNGs are not supported.")
        pixels = np.array(image, dtype=np.uint8)
    return pixels, _sha256(content)


class PNGPatientDataset(Dataset):
    """Load all manifest rows for explicitly supplied patients, one PNG at a time.

    Args:
        manifest_path: The existing data2d/manifest.csv file.
        patient_ids: Required non-empty sequence of unique case_XXXXX IDs.
        data_root: Directory containing images/ and masks/; defaults to the CSV
            directory. It must be explicitly approved by the caller.
        config: PNGConfig; exactly one input channel and no resizing.

    Rows are ordered by the supplied patient order and then slice index. Files
    are not silently dropped: missing files fail when their sample is loaded.
    Public rows, config_sha256 and manifest_sha256 support inspection tools;
    callers inspecting a subset must report that subset rather than a full audit.
    """
    def __init__(self, manifest_path, patient_ids, *, data_root=None, config=PNGConfig()):
        if isinstance(patient_ids, (str, bytes)):
            raise ValueError("patient_ids must be an explicit sequence of patient IDs, not one string.")
        try:
            patient_ids = tuple(patient_ids)
        except TypeError:
            raise ValueError("patient_ids must be a non-empty sequence.") from None
        if not patient_ids or any(not isinstance(patient, str) or not re.fullmatch(r"case_[0-9]{5}", patient) for patient in patient_ids):
            raise ValueError("Supply at least one valid patient ID such as case_00000.")
        if len(set(patient_ids)) != len(patient_ids):
            raise ValueError("Duplicate patient selection is not allowed.")
        if not isinstance(config, PNGConfig):
            raise ValueError("Use a validated PNGConfig for preprocessing.")
        self.config = config
        self.patient_ids = patient_ids
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if self.manifest_path.suffix != ".csv":
            raise ValueError("The PNG adapter requires a CSV manifest.")
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else self.manifest_path.parent
        if not self.data_root.is_dir():
            raise ValueError("The approved data root directory does not exist.")
        content = self.manifest_path.read_bytes()
        self.manifest_sha256 = _sha256(content)
        self.config_sha256 = _sha256(json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        self.selection_scope = "exported_2d_slices_for_explicit_patients; no split assignment or complete pixel audit"
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        required = {"case", "z", "image", "mask", "kidney_px", "tumor_px", "has_kidney", "has_tumor", "n_slices"}
        headers = reader.fieldnames or []
        if not required.issubset(headers) or len(set(headers)) != len(headers):
            raise ValueError("The manifest requires unique patient/slice/path/mask-metadata column names.")
        seen, available, selected = set(), set(), []
        for row in reader:
            patient = row["case"]
            if not isinstance(patient, str) or not re.fullmatch(r"case_[0-9]{5}", patient):
                raise ValueError("The manifest contains an invalid patient ID.")
            z = _integer(row, "z")
            identity = (patient, z)
            if identity in seen:
                raise ValueError("Duplicate patient/slice identity in the manifest: %s at z=%d." % identity)
            seen.add(identity)
            available.add(patient)
            if patient not in patient_ids:
                continue
            n_slices = _integer(row, "n_slices")
            if n_slices == 0 or z >= n_slices:
                raise ValueError("Selected slice index must satisfy 0 <= z < positive n_slices.")
            for field in ("kidney_px", "tumor_px", "has_kidney", "has_tumor"):
                _integer(row, field)
            if _integer(row, "has_kidney") not in (0, 1) or _integer(row, "has_tumor") not in (0, 1):
                raise ValueError("Manifest has_kidney and has_tumor must be 0 or 1.")
            _asset_path(self.data_root, row["image"], "images", patient, z)
            _asset_path(self.data_root, row["mask"], "masks", patient, z)
            selected.append({**row, "z": z})
        missing = set(patient_ids) - available
        if missing:
            raise ValueError("Requested patient IDs are absent from the manifest: %s." % ", ".join(sorted(missing)))
        order = {patient: index for index, patient in enumerate(patient_ids)}
        self.rows = tuple(sorted(selected, key=lambda row: (order[row["case"]], row["z"])))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or not 0 <= index < len(self):
            raise IndexError("PNG sample index is outside the explicitly selected records.")
        row = self.rows[index]
        patient, z = row["case"], row["z"]
        # Resolve again on access so a changed symlink cannot bypass containment.
        image_path = _asset_path(self.data_root, row["image"], "images", patient, z)
        mask_path = _asset_path(self.data_root, row["mask"], "masks", patient, z)
        image, image_hash = _read_png(image_path, self.config.size)
        mask, mask_hash = _read_png(mask_path, self.config.size)
        if not np.isin(mask, [0, 1, 2]).all():
            raise ValueError("Reference mask labels must be 0=background, 1=kidney, 2=tumour.")
        counts = {str(label): int((mask == label).sum()) for label in (0, 1, 2)}
        observed = {"kidney_px": counts["1"], "tumor_px": counts["2"],
                    "has_kidney": int(counts["1"] > 0), "has_tumor": int(counts["2"] > 0)}
        mismatches = [field for field, value in observed.items() if _integer(row, field) != value]
        if mismatches:
            raise ValueError("Reference-mask pixels disagree with manifest metadata: %s." % ", ".join(mismatches))
        # The target never affects the input: normalize the CT PNG only.
        normalized = (image.astype(np.float32) / 255.0 - self.config.mean) / self.config.std
        inputs = torch.from_numpy(np.ascontiguousarray(normalized[None]))
        target = torch.from_numpy(mask.astype(np.int64))
        return {"input": inputs, "target": target, "patient_id": patient, "slice_index": z,
                "sample_id": "%s:z%d" % (patient, z), "input_type": "single_2d_ct_png",
                "source_paths": {"image": str(image_path), "mask": str(mask_path), "manifest": str(self.manifest_path)},
                "source_sha256": {"image": image_hash, "mask": mask_hash},
                "manifest_sha256": self.manifest_sha256, "config_sha256": self.config_sha256,
                "mask_label_counts": counts, "reference_metadata_matches": True,
                "annotation_type": "ground_truth_reference_segmentation", "contains_model_prediction": False,
                "coordinate_space": "processed_png_256x256", "selection_scope": self.selection_scope}
