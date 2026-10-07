"""CT-only segmentation of every image in the frozen prediction pool.

Preflight reads the locked evidence and CT identities, never PNG pixels or a
model. Inference reads only audited CT PNGs. Reference paths, label columns and
reference pixels are not dataset inputs or example-selection criteria. Callers
create the new empty output directory and own failure/progress persistence.
"""
from collections import Counter
import copy
import csv
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Dataset

from . import CLASSES
from . import png_final_test as final_test
from . import png_training as training
from .png_dataset import _integer


PREDICTION_SCHEMA = "png_frozen_prediction_v1"
PREVIEW_RULE = "first and middle rows in ascending exported slice-index order; independent of reference labels and predictions"
OVERLAY_ALPHA = 0.4
OVERLAY_COLORS = np.array([[0, 0, 0], [50, 200, 100], [255, 80, 80]], dtype=np.float32)


def _ct_manifest_rows(content, patient_ids):
    """Project the CSV onto identity/CT fields; never parse label metadata."""
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    if not {"case", "z", "image", "n_slices"}.issubset(headers) or len(set(headers)) != len(headers):
        raise ValueError("CT prediction requires unique case/z/image/n_slices manifest columns.")
    selected, seen = [], set()
    for row in reader:
        patient, z = row["case"], _integer(row, "z")
        if not isinstance(patient, str) or not re.fullmatch(r"case_[0-9]{5}", patient):
            raise ValueError("The CT manifest contains an invalid patient identity.")
        identity = (patient, z)
        if identity in seen:
            raise ValueError("Duplicate CT patient/slice identity in the manifest.")
        seen.add(identity)
        if patient not in patient_ids:
            continue
        n_slices = _integer(row, "n_slices")
        if n_slices == 0 or z >= n_slices:
            raise ValueError("Selected CT slice index must be below positive n_slices.")
        selected.append({"case": patient, "z": z, "image": row["image"], "n_slices": n_slices})
    order = {patient: index for index, patient in enumerate(patient_ids)}
    return tuple(sorted(selected, key=lambda row: (order[row["case"]], row["z"])))


class FrozenPredictionPNGDataset(Dataset):
    """Dedicated image-only loader for the exact frozen prediction allowlist."""
    def __init__(self, inspected):
        final_test._check_snapshot(inspected, require_freeze=True)
        self.context = inspected["context"]
        self.config = inspected["config"]
        self.data_root = self.context["data_root"]
        self.manifest_path = self.context["paths"]["manifest"]
        records = self.context["cohort"]["pools"]["prediction"]
        self.patient_ids = tuple(record["patient_id"] for record in records)
        self.rows = _ct_manifest_rows(self.manifest_path.read_bytes(), self.patient_ids)
        audit_index = {patient["patient_id"]: patient for patient in self.context["audit"]["patients"]}
        counts = Counter(row["case"] for row in self.rows)
        for record in records:
            patient = record["patient_id"]
            audited = audit_index[patient]
            assets = [(row["slice_index"], row["image"]) for row in audited["slice_assets"]]
            observed = [(row["z"], row["image"]) for row in self.rows if row["case"] == patient]
            if (counts[patient] != record["selected_slice_count"]
                    or len(set(assets)) != len(assets) or set(observed) != set(assets)
                    or any(row["n_slices"] != audited["original_scan_slice_count"]
                           for row in self.rows if row["case"] == patient)):
                raise ValueError("Prediction CT rows/assets/counts differ from the exact locked audit.")
        if len(self) != inspected["freeze_record"]["selected_slice_counts"]["prediction"]:
            raise ValueError("Prediction CT slice total differs from the frozen pool.")
        for row in self.rows:
            self._image_path_and_facts(row)
        self.preview_sample_ids = {}
        for patient in self.patient_ids:
            rows = [row for row in self.rows if row["case"] == patient]
            if len(rows) < 2:
                raise ValueError("Each prediction patient needs two distinct exported CT rows for the fixed preview.")
            self.preview_sample_ids[patient] = [self._sample_id(rows[0]), self._sample_id(rows[len(rows) // 2])]

    @staticmethod
    def _sample_id(row):
        return "%s:z%d" % (row["case"], row["z"])

    def __len__(self):
        return len(self.rows)

    def _image_path_and_facts(self, row):
        if row["case"] not in self.patient_ids:
            raise ValueError("CT loading escaped the frozen prediction patient allowlist.")
        facts = self.context["audit"]["assets"].get(row["image"], {})
        if (facts.get("kind") != "image" or facts.get("patient_id") != row["case"]
                or facts.get("slice_index") != row["z"] or facts.get("mode") != "L"
                or (facts.get("width"), facts.get("height")) != (256, 256)):
            raise ValueError("Saved CT asset facts disagree with the selected prediction identity.")
        path = training._safe_asset(self.data_root, row["image"], "images", row["case"], row["z"])
        return path, facts

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError("CT prediction sample index is outside the frozen rows.")
        row = self.rows[index]
        path, facts = self._image_path_and_facts(row)
        content = path.read_bytes()
        byte_hash = final_test._hash(content)
        if byte_hash != facts["byte_sha256"]:
            raise ValueError("Prediction CT PNG bytes changed after the saved audit.")
        with Image.open(io.BytesIO(content)) as image:
            if (image.format != "PNG" or image.mode != "L" or image.size != (256, 256)
                    or getattr(image, "n_frames", 1) != 1):
                raise ValueError("Prediction CT must be a single 256x256 uint8 L-mode PNG, without resizing.")
            pixels = np.array(image, dtype=np.uint8)
        pixel_hash = final_test._hash(pixels.tobytes(order="C"))
        if pixel_hash != facts["decoded_pixel_sha256"]:
            raise ValueError("Prediction CT decoded-pixel hash differs from the saved audit.")
        normalized = (pixels.astype(np.float32) / 255.0 - self.config.mean) / self.config.std
        return {"input": torch.from_numpy(np.ascontiguousarray(normalized[None])),
                "ct_pixels": torch.from_numpy(pixels), "patient_id": row["case"],
                "slice_index": row["z"], "sample_id": self._sample_id(row),
                "source_image": str(path), "source_image_sha256": byte_hash,
                "source_decoded_pixel_sha256": pixel_hash, "pool": "prediction"}

    def metadata(self):
        return {"patient_ids": list(self.patient_ids), "patient_count": len(self.patient_ids),
                "slice_count": len(self), "patient_slice_counts": dict(Counter(row["case"] for row in self.rows)),
                "sample_ids": [self._sample_id(row) for row in self.rows],
                "preview_sample_ids": copy.deepcopy(self.preview_sample_ids), "preview_selection_rule": PREVIEW_RULE,
                "input_source": "CT PNG only", "mask_pixels_loaded": 0,
                "reference_metadata_used_for_selection": False}


def build_prediction_dataset(inspected):
    """Construct the CT-only fixed-pool loader without decoding PNG pixels."""
    dataset = FrozenPredictionPNGDataset(inspected)
    expected = inspected.get("prediction_inputs")
    if expected is not None and not final_test._json_equal(expected, dataset.metadata()):
        raise ValueError("Prediction dataset metadata differs from inspected CT identities.")
    return dataset


def inspect_prediction_inputs(root, cohort_path, audit_path, run_dir, freeze_path, *, allow_fixture=False):
    """Return frozen model evidence plus image-only preflight metadata."""
    inspected = final_test.inspect_frozen_model(root, cohort_path, audit_path, run_dir, freeze_path,
                                               allow_fixture=allow_fixture)
    inspected["prediction_inputs"] = build_prediction_dataset(inspected).metadata()
    final_test._check_snapshot(inspected, require_freeze=True)
    return inspected


def _empty_output_directory(inspected, output_dir):
    path = Path(output_dir).expanduser().absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Prediction output paths must not contain symlinks.")
    path = path.resolve()
    protected = (inspected["context"]["root"], inspected["run_dir"])
    if any(path == root or root in path.parents or path in root.parents for root in protected):
        raise ValueError("Prediction output must be outside the protected source data and completed run.")
    protected_files = [*inspected["context"]["paths"].values(), *inspected["artifact_paths"].values(),
                       inspected["run_summary_path"], inspected["freeze_path"]]
    if any(path == source or path in source.parents for source in protected_files):
        raise ValueError("Prediction output must not contain protected frozen evidence.")
    if not path.is_dir():
        raise ValueError("The caller must create a new empty prediction output directory.")
    if any(path.iterdir()):
        raise FileExistsError("Prediction output directory must be empty; existing outputs are never replaced.")
    return path


def _overlay(pixels, prediction):
    rgb = np.repeat(pixels[..., None], 3, axis=-1).astype(np.float32)
    active = prediction > 0
    rgb[active] = (1 - OVERLAY_ALPHA) * rgb[active] + OVERLAY_ALPHA * OVERLAY_COLORS[prediction[active]]
    return np.rint(rgb).astype(np.uint8)


def _save_png(path, pixels):
    with path.open("xb") as handle:
        Image.fromarray(pixels).save(handle, format="PNG")
    return {"path": str(path), "sha256": final_test._hash(path.read_bytes())}


def _save_preview(path, patient, selected):
    preview = Image.new("RGB", (536, 614), "white")
    draw = ImageDraw.Draw(preview)
    draw.text((12, 8), "%s - model segmentation preview" % patient, fill="black")
    draw.text((12, 24), "Kidney: green   Tumour: red   No reference annotation", fill="black")
    for index, (sample, pixels, overlay) in enumerate(selected):
        top = 58 + index * 278
        draw.text((12, top - 16), "CT %s" % sample, fill="black")
        draw.text((280, top - 16), "Model segmentation", fill="black")
        preview.paste(Image.fromarray(pixels).convert("RGB"), (12, top))
        preview.paste(Image.fromarray(overlay), (280, top))
    with path.open("xb") as handle:
        preview.save(handle, format="PNG")
    return {"path": str(path), "sha256": final_test._hash(path.read_bytes()),
            "sample_ids": [sample for sample, _, _ in selected], "selection_rule": PREVIEW_RULE,
            "content": "two CT images paired with their model segmentation overlays"}


def _check_model_evidence(model, inspected, expected_state):
    metadata = inspected["metadata"]
    if (training._state_hash(training._state(model)) != expected_state
            or expected_state != metadata["state_fingerprint_sha256"]):
        raise ValueError("Frozen model parameters or buffers changed during prediction inference.")
    if (not final_test._json_equal(model.png_metadata, metadata)
            or not final_test._json_equal(model.pretrained_proof, metadata["pretrained_proof"])
            or model.png_checkpoint_sha256 != inspected["checkpoint_sha256"]
            or model.png_cohort_fingerprint_sha256 != metadata["cohort"]["cohort_fingerprint_sha256"]
            or model.architecture != metadata["architecture"] or model.input_channels != 1
            or model.pretrained != metadata["pretrained_proof"]["pretrained_encoder"]
            or model.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())):
        raise ValueError("Frozen prediction model provenance or inference-only state changed.")
    training._check_encoder(model, training._state(model.network.encoder))


def predict_frozen_pool(inspected, output_dir, *, batch_size=2, num_threads=2, progress=None, model_builder=None):
    """Write every fixed CT prediction once and return its persisted JSON report."""
    for value, name in ((batch_size, "batch_size"), (num_threads, "num_threads")):
        if type(value) is not int or value < 1:
            raise ValueError("%s must be a positive integer." % name)
    final_test._check_snapshot(inspected, require_freeze=True)
    directory = _empty_output_directory(inspected, output_dir)
    metadata = inspected["metadata"]
    fixture = metadata["data_mode"] == "synthetic_fixture"
    if (fixture and inspected.get("allow_fixture") is not True) or (model_builder is not None and not fixture):
        raise ValueError("Injected prediction models require explicitly allowed synthetic fixtures.")
    if fixture and model_builder is None:
        raise ValueError("Synthetic fixture prediction requires its explicit test model builder.")
    dataset = build_prediction_dataset(inspected)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    reporter = training._Progress(progress, {"phase": "independent_prediction", "training_performed": False,
                                            "model_selection_performed": False, "optimizer_steps": 0})
    try:
        model, config, loaded_metadata = training.load_png_checkpoint(
            inspected["checkpoint_path"], device="cpu", model_builder=model_builder)
        if (config != inspected["config"] or not final_test._json_equal(loaded_metadata, metadata)
                or model.png_checkpoint_sha256 != inspected["checkpoint_sha256"]):
            raise ValueError("Loaded prediction model differs from the inspected frozen checkpoint.")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = training._state_hash(training._state(model))
        if before != metadata["state_fingerprint_sha256"]:
            raise ValueError("Loaded prediction state differs from the frozen state fingerprint.")
        _check_model_evidence(model, inspected, before)
        final_test._check_snapshot(inspected, require_freeze=True)
        patients = {patient: {"patient_id": patient, "exported_slice_count": 0, "slices": []}
                    for patient in dataset.patient_ids}
        previews = {patient: {} for patient in dataset.patient_ids}
        for patient in dataset.patient_ids:
            (directory / patient).mkdir()
        counts, sample_counts, seen = Counter(), Counter(), 0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        reporter.emit("initial", prediction_patient_count=len(dataset.patient_ids), total_slices=len(dataset),
                      batch_size=batch_size, num_threads=num_threads, checkpoint_sha256=inspected["checkpoint_sha256"])
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, 1):
                logits = model(batch["input"])
                if not isinstance(logits, torch.Tensor) or logits.shape != (len(batch["input"]), 3, 256, 256):
                    raise ValueError("Prediction model must return three-class 256x256 single-slice logits.")
                if not logits.is_floating_point() or not torch.isfinite(logits).all():
                    raise ValueError("Prediction logits must be finite floating-point values.")
                predictions = logits.argmax(1).cpu().numpy().astype(np.uint8)
                for index, prediction in enumerate(predictions):
                    patient, sample = batch["patient_id"][index], batch["sample_id"][index]
                    if patient not in patients:
                        raise ValueError("Prediction inference escaped the fixed patient allowlist.")
                    z = int(batch["slice_index"][index])
                    pixels = batch["ct_pixels"][index].numpy()
                    overlay = _overlay(pixels, prediction)
                    stem = "%s_z%03d" % (patient, z)
                    labels = _save_png(directory / patient / (stem + "_predicted_mask.png"), prediction)
                    colored = _save_png(directory / patient / (stem + "_model_overlay.png"), overlay)
                    item = {"sample_id": sample, "slice_index": z, "source_image": batch["source_image"][index],
                            "source_image_sha256": batch["source_image_sha256"][index],
                            "source_decoded_pixel_sha256": batch["source_decoded_pixel_sha256"][index],
                            "predicted_class_counts": {str(label): int((prediction == label).sum()) for label in CLASSES},
                            "predicted_mask": labels, "model_overlay": colored}
                    patients[patient]["slices"].append(item)
                    if sample in dataset.preview_sample_ids[patient]:
                        previews[patient][sample] = (sample, pixels.copy(), overlay.copy())
                    counts[patient] += 1
                    sample_counts[sample] += 1
                    seen += 1
                reporter.batch(batch_index, len(loader), slices_seen=seen, total_slices=len(dataset), progress=seen / len(dataset))
        expected = {dataset._sample_id(row): 1 for row in dataset.rows}
        if counts != Counter(row["case"] for row in dataset.rows) or dict(sample_counts) != expected or seen != len(dataset):
            raise ValueError("Prediction must use every frozen CT row exactly once.")
        after = training._state_hash(training._state(model))
        _check_model_evidence(model, inspected, before)
        final_test._check_snapshot(inspected, require_freeze=True)
        for patient, item in patients.items():
            item["exported_slice_count"] = counts[patient]
            selected = [previews[patient][sample] for sample in dataset.preview_sample_ids[patient]]
            item["preview"] = _save_preview(directory / patient / (patient + "_preview.png"), patient, selected)
        report = {
            "schema_version": PREDICTION_SCHEMA, "status": "completed", "split": "prediction",
            "data_mode": metadata["data_mode"], "scope": final_test.RUN_SCOPE,
            "patient_count": len(patients), "patient_ids": list(dataset.patient_ids), "slice_count": seen,
            "prediction_ct_count": seen, "patients": list(patients.values()), "sample_counts": dict(sample_counts),
            "patient_slice_counts": dict(counts), "all_rows_seen_once": True,
            "patient_counts": copy.deepcopy(inspected["freeze_record"]["patient_counts"]),
            "selected_slice_counts": copy.deepcopy(inspected["freeze_record"]["selected_slice_counts"]),
            "decoded_ct_images": {role: seen if role == "prediction" else 0 for role in final_test.ROLES},
            "test_pixels_loaded": 0, "mask_pixels_loaded": 0, "ground_truth_metrics_computed": False,
            "reference_metadata_used_for_selection": False, "ground_truth_used_for_inference": False,
            "preview_selection_rule": PREVIEW_RULE, "classes": {str(key): value for key, value in CLASSES.items()},
            "overlay": {"type": "model_segmentation", "alpha": OVERLAY_ALPHA,
                        "colors_rgb": {str(label): list(map(int, OVERLAY_COLORS[label])) for label in (1, 2)},
                        "legend": {"kidney": "green", "tumour": "red"}, "confidence_heatmap": False},
            "checkpoint_path": str(inspected["checkpoint_path"]), "checkpoint_sha256": inspected["checkpoint_sha256"],
            "frozen_model_sha256": inspected["checkpoint_sha256"], "selected_epoch": metadata["best_epoch"],
            "checkpoint_selection": copy.deepcopy(metadata["checkpoint_selection"]),
            "state_fingerprint_sha256": metadata["state_fingerprint_sha256"],
            "model_state_before_sha256": before, "model_state_after_sha256": after, "model_state_unchanged": True,
            "freeze_path": str(inspected["freeze_path"]), "freeze_sha256": inspected["freeze_sha256"],
            "freeze_fingerprint_sha256": inspected["freeze_record"][final_test.FINGERPRINT_FIELD],
            "run_summary_sha256": inspected["run_summary_sha256"],
            "training_artifact_sha256": copy.deepcopy(inspected["training_artifact_sha256"]),
            "bound_file_sha256": inspected["context"]["file_sha256"].copy(),
            "cohort_fingerprint_sha256": inspected["context"]["cohort"]["cohort_fingerprint_sha256"],
            "audit_fingerprint_sha256": inspected["context"]["audit"]["audit_fingerprint_sha256"],
            "preprocessing": config.to_dict(), "preprocessing_sha256": metadata["preprocessing_sha256"],
            "pretrained_encoder_proof": copy.deepcopy(metadata["pretrained_proof"]),
            "input_source": "single CT PNG only; no reference mask or neighbouring slice", "resizing_performed": False,
            "inference_rule": "hard_argmax_over_three_fixed_class_logits", "batch_size": batch_size,
            "num_threads": num_threads, "device": "cpu", "optimizer_steps": 0, "training_performed": False,
            "model_selection_performed": False, "threshold_tuning_performed": False,
            "full_scan_coverage": False, "clinical_performance_evidence": False, "patient_level_diagnosis": False,
            "confidence_estimate_provided": False, "anatomical_alignment_verified": False,
            "source_png_byte_and_decoded_pixel_hashes_verified": True, "bound_inputs_unchanged": True,
            "output_dir": str(directory), "created_at_utc": training._utc(),
            "limitations": ["Predictions cover only the locked exported 2D CT PNGs, not complete original scans.",
                            "A coloured hard-label model overlay is not a confidence estimate or clinical diagnosis.",
                            "No ground-truth segmentation metrics are computed for the independent prediction pool."]}
        reporter.emit("completed", slices_seen=seen, prediction_patient_count=len(patients), model_state_unchanged=True)
        # Completion callbacks remain inside the protected inference interval.
        final_test._check_snapshot(inspected, require_freeze=True)
        _check_model_evidence(model, inspected, before)
        with (directory / "prediction_report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return report
    finally:
        torch.set_num_threads(previous_threads)
