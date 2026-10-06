"""Audit-bound single-channel PNG training, separate from the 2.5D engine.

Every exported training slice is used once per epoch. Only validation is
evaluated here; test and prediction pools remain checkpoint metadata. Synthetic
model injection is a testing facility, never a pretrained fallback for real data.
"""
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import CLASSES
from .cohort import canonical_fingerprint, validate_cohort
from .metrics import confusion_matrix, patient_report, summarize_patients
from .model import KidneyUNet, segmentation_loss
from .png_dataset import PNGConfig, PNGPatientDataset
from .pretrained import (build_verified_model, CONFIG_SHA256, REVISION,
                         WEIGHTS_SHA256)

SCHEMA = "png_segmentation_v1"
METRIC_SCOPE = ("hard-argmax pixels in every exported reference-selected 256x256 "
                "2D PNG slice of the specified patient pool; no resize or full-volume evaluation")


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _check_audit_preprocessing(audit, config):
    recorded = audit.get("preprocessing", {})
    expected = {**config.to_dict(), "input_mode": "single_2d_slice",
                "units": "already_processed_uint8_not_HU",
                "mask_labels": {"0": "background", "1": "kidney", "2": "tumour"}}
    digest = _hash(json.dumps(recorded, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if (not isinstance(recorded, dict) or any(recorded.get(key) != value for key, value in expected.items()) or
            digest != audit.get("preprocessing_sha256")):
        raise ValueError("Saved audit preprocessing differs from the reviewed PNG configuration.")


def load_training_context(root, cohort_path, audit_path):
    """Validate saved locks/current CSV and old split, without decoding any PNG."""
    root = Path(root).expanduser().resolve()
    data = (root / "data2d").resolve()
    if not root.is_dir() or root not in data.parents:
        raise ValueError("A contained kits_project/data2d directory is required.")
    paths = {"cohort": Path(cohort_path).expanduser().resolve(),
             "audit": Path(audit_path).expanduser().resolve(),
             "manifest": (data / "manifest.csv").resolve(),
             "original_split": (data / "splits.json").resolve()}
    if any(data not in paths[key].parents for key in ("manifest", "original_split")):
        raise ValueError("Manifest and original split must remain inside data2d.")
    content = {key: path.read_bytes() for key, path in paths.items()}
    cohort, audit = json.loads(content["cohort"]), json.loads(content["audit"])
    validate_cohort(cohort, audit)
    if (_hash(content["manifest"]) != audit.get("manifest_sha256") or
            _hash(content["original_split"]) != audit.get("original_split_sha256")):
        raise ValueError("Current manifest/original split differs from the locked PNG audit.")
    config = PNGConfig(**cohort["preprocessing"]["config"])
    if config != PNGConfig():
        raise ValueError("Training requires the reviewed single-channel PNG preprocessing.")
    _check_audit_preprocessing(audit, config)
    return {"root": root, "data_root": data, "paths": paths,
            "file_sha256": {key: _hash(value) for key, value in content.items()},
            "cohort": cohort, "audit": audit, "config": config}


def _check_bindings(context):
    for key, path in context["paths"].items():
        if _hash(path.read_bytes()) != context["file_sha256"][key]:
            raise ValueError("A bound %s file changed during training." % key)


def _safe_asset(data, relative, folder, patient_id, z):
    expected = Path(folder) / ("%s_z%03d.png" % (patient_id, z))
    if not isinstance(relative, str) or Path(relative) != expected:
        raise ValueError("Audited asset filename must match its selected patient/slice.")
    path = data / expected
    # Reject an alias to another patient before opening or decoding its pixels.
    if any(part.is_symlink() for part in (path, path.parent, data)):
        raise ValueError("Audited PNG assets/directories must not be symlinks.")
    if path.resolve() != path or not path.is_file():
        raise ValueError("Audited PNG must be a regular file at its exact contained path.")
    return path


class AuditedPNGDataset(PNGPatientDataset):
    """All rows for one locked train/validation pool, checked on every access."""
    def __init__(self, context, role="train"):
        if role not in ("train", "validation"):
            raise ValueError("This training module never constructs test/prediction pixel datasets.")
        _check_bindings(context)
        self.context, self.role = context, role
        records = context["cohort"]["pools"][role]
        super().__init__(context["paths"]["manifest"], [r["patient_id"] for r in records],
                         data_root=context["data_root"], config=context["config"])
        if self.manifest_sha256 != context["audit"]["manifest_sha256"]:
            raise ValueError("PNG loader manifest differs from the saved audit.")
        if self.config_sha256 != context["cohort"]["preprocessing"]["config_sha256"]:
            raise ValueError("PNG loader preprocessing differs from the cohort lock.")
        index = {r["patient_id"]: r for r in context["audit"]["patients"]}
        counts = Counter(row["case"] for row in self.rows)
        for record in records:
            patient = record["patient_id"]
            if counts[patient] != record["selected_slice_count"]:
                raise ValueError("Selected patient row count differs from the cohort lock.")
            audited = {(r["slice_index"], r["image"], r["mask"])
                       for r in index[patient]["slice_assets"]}
            observed = {(r["z"], r["image"], r["mask"]) for r in self.rows if r["case"] == patient}
            if observed != audited:
                raise ValueError("Selected patient slice assets differ from the saved audit.")
        # Preflight every selected path before the first batch can be decoded.
        for row in self.rows:
            self._paths_and_facts(row)

    def _paths_and_facts(self, row):
        if row["case"] not in self.patient_ids:
            raise ValueError("PNG loading is restricted to the locked patient allowlist.")
        result = {}
        for kind, folder in (("image", "images"), ("mask", "masks")):
            facts = self.context["audit"]["assets"][row[kind]]
            if (facts["kind"] != kind or facts["patient_id"] != row["case"] or
                    facts["slice_index"] != row["z"] or facts["mode"] != "L" or
                    (facts["width"], facts["height"]) != (256, 256)):
                raise ValueError("Saved asset facts disagree with the selected PNG identity.")
            path = _safe_asset(self.data_root, row[kind], folder, row["case"], row["z"])
            result[kind] = (path, facts)
        return result

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(self):
            raise IndexError("Audited PNG sample index is outside the selected rows.")
        row = self.rows[index]
        assets = self._paths_and_facts(row)
        # Check raw bytes before the base adapter opens a PNG decoder.
        for path, facts in assets.values():
            if _hash(path.read_bytes()) != facts["byte_sha256"]:
                raise ValueError("Selected PNG bytes changed after the saved audit.")
        self._paths_and_facts(row)
        item = super().__getitem__(index)
        # Inverting the reviewed normalization and rounding recovers uint8 pixels
        # exactly; this avoids a second PNG decode solely for content hashing.
        ct = np.rint((item["input"][0].numpy() * self.config.std + self.config.mean) * 255).astype(np.uint8)
        mask = item["target"].numpy().astype(np.uint8)
        for kind, pixels in (("image", ct), ("mask", mask)):
            facts = assets[kind][1]
            if (item["source_sha256"][kind] != facts["byte_sha256"] or
                    _hash(pixels.tobytes(order="C")) != facts["decoded_pixel_sha256"]):
                raise ValueError("Loaded PNG byte/decoded-pixel hashes differ from the saved audit.")
        if item["mask_label_counts"] != assets["mask"][1]["label_counts"]:
            raise ValueError("Loaded reference counts differ from the saved audit.")
        item["pool"] = self.role
        item["audit_fingerprint_sha256"] = self.context["audit"]["audit_fingerprint_sha256"]
        item["selection_scope"] = METRIC_SCOPE
        return item


def _state(model_or_module):
    return {key: value.detach().cpu().clone() for key, value in model_or_module.state_dict().items()}


def _state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _same_state(module, expected):
    observed = module.state_dict()
    return set(observed) == set(expected) and all(torch.equal(observed[k].detach().cpu(), v) for k, v in expected.items())


def _proof_valid(proof, data_mode):
    if not isinstance(proof, dict):
        raise ValueError("Explicit model initialization proof is required.")
    if data_mode == "synthetic_fixture":
        if proof.get("initialization_kind") != "synthetic_fixture" or proof.get("pretrained_encoder") is not False:
            raise ValueError("Synthetic model injection must explicitly report an unpretrained fixture.")
        return
    required = {"offline_only": True, "download_attempted": False, "random_encoder_fallback": False,
                "pretrained_encoder": True, "all_official_encoder_tensors_exact": True,
                "encoder_frozen": True, "encoder_batch_norm_eval": True, "input_channels": 1,
                "first_conv_adaptation": "sum_rgb_weights", "adapted_first_conv_equals_rgb_sum": True,
                "source_revision": REVISION, "weights_sha256": WEIGHTS_SHA256, "config_sha256": CONFIG_SHA256}
    if any(proof.get(key) != value for key, value in required.items()):
        raise ValueError("Real PNG training requires complete pinned offline pretrained encoder proof.")


def _check_encoder(model, expected):
    encoder = model.network.encoder
    if (encoder.training or any(p.requires_grad or p.grad is not None for p in encoder.parameters()) or
            not _same_state(encoder, expected)):
        raise ValueError("Frozen encoder parameters or BatchNorm state changed during training.")


def _trainable_groups(model):
    groups = {"decoder": [], "head": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name.startswith("network.decoder."):
                groups["decoder"].append((name, parameter))
            elif name.startswith("network.segmentation_head."):
                groups["head"].append((name, parameter))
            else:
                raise ValueError("Only decoder and segmentation-head parameters may be trainable.")
    if any(not group for group in groups.values()):
        raise ValueError("Both decoder and segmentation head must contain trainable parameters.")
    return groups


def _check_gradients(groups):
    for name, entries in groups.items():
        gradients = [p.grad for _, p in entries]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise ValueError("Missing or non-finite %s gradients." % name)
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise ValueError("All %s gradients are zero." % name)


class _Progress:
    def __init__(self, callback, common):
        self.callback, self.common, self.last = callback, common, time.monotonic()

    def emit(self, event, **fields):
        if self.callback:
            self.callback({**self.common, "event": event, "timestamp_utc": _utc(), **copy.deepcopy(fields)})
        self.last = time.monotonic()

    def batch(self, batch_index, total_batches, **fields):
        if batch_index % 10 == 0 or batch_index == total_batches or time.monotonic() - self.last >= 30:
            phase = fields.get("phase", self.common.get("phase"))
            self.emit("%s_batch" % phase, batch_index=batch_index, total_batches=total_batches, **fields)


def evaluate_png_validation(model, dataset, *, batch_size=2, device="cpu", progress=None,
                            epoch=None, epochs=None):
    """Hard predictions accumulated per validation patient, without optimizer work."""
    if not isinstance(dataset, AuditedPNGDataset) or dataset.role != "validation":
        raise ValueError("Only a locked audited validation dataset can be evaluated here.")
    _check_bindings(dataset.context)
    if getattr(model, "png_cohort_fingerprint_sha256", None) != dataset.context["cohort"]["cohort_fingerprint_sha256"]:
        raise ValueError("Validation membership differs from the fitted model's cohort.")
    model.eval()
    matrices = {patient: np.zeros((3, 3), dtype=np.int64) for patient in dataset.patient_ids}
    counts, seen, loss_sum = Counter(), 0, 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    reporter = _Progress(progress, {"phase": "validation", "epoch": epoch, "epochs": epochs,
                                    "metric_scope": METRIC_SCOPE})
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            logits = model(batch["input"].to(device))
            target = batch["target"].to(device)
            loss = segmentation_loss(logits, target)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise ValueError("Validation logits/loss became non-finite.")
            predicted = logits.argmax(1).cpu().numpy()
            for patient, actual, prediction in zip(batch["patient_id"], target.cpu().numpy(), predicted):
                matrices[patient] += confusion_matrix(actual, prediction)
                counts[patient] += 1
            seen += len(target)
            loss_sum += float(loss) * len(target)
            reporter.batch(batch_index, len(loader), slices_seen=seen, total_slices=len(dataset),
                           loss=loss_sum / seen, progress=seen / len(dataset))
    expected = Counter(row["case"] for row in dataset.rows)
    if counts != expected or seen != len(dataset):
        raise ValueError("Validation must evaluate every selected slice exactly once.")
    reports = []
    for patient, matrix in matrices.items():
        item = patient_report(patient, matrix)
        item["exported_slice_count"] = counts[patient]
        reports.append(item)
    report = summarize_patients(reports)
    report.update(loss=loss_sum / seen, split="validation", slice_count=seen, metric_scope=METRIC_SCOPE,
                  loss_scope="slice-weighted mean of minibatch cross entropy plus foreground soft-Dice",
                  model_selection_only=True, clinical_performance_evidence=False,
                  full_scan_coverage=False, created_at_utc=_utc())
    return report


def train_png(root, cohort_path, audit_path, *, epochs=1, batch_size=2, lr=1e-3,
              seed=42, device="cpu", num_threads=2, progress=None, cache_root=None,
              model_builder=None, data_mode="real", visual_review_note=None):
    """Train a new PNG model and return its best-validation state and evidence.

    No files are written here. The caller persists reports and a new checkpoint.
    A visual_review_note records a human statement; it does not certify alignment.
    """
    for value, name in ((epochs, "epochs"), (batch_size, "batch_size"), (num_threads, "num_threads")):
        if type(value) is not int or value < 1:
            raise ValueError("%s must be a positive integer." % name)
    if type(seed) is not int or not isinstance(lr, (float, int)) or isinstance(lr, bool) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Use an integer seed and a positive finite learning rate.")
    if str(device) != "cpu":
        raise ValueError("This first PNG training module supports CPU execution only.")
    if data_mode not in ("real", "synthetic_fixture"):
        raise ValueError("Unknown PNG training data mode.")
    if model_builder is not None and data_mode != "synthetic_fixture":
        raise ValueError("Injected models are permitted only for synthetic fixture tests.")
    if visual_review_note is not None and not isinstance(visual_review_note, str):
        raise ValueError("visual_review_note must be text or None.")
    context = load_training_context(root, cohort_path, audit_path)
    fixture_audit = context["audit"].get("data_origin") == "synthetic_fixture"
    if fixture_audit != (data_mode == "synthetic_fixture"):
        raise ValueError("Synthetic fixture mode requires an explicitly marked fixture audit; real mode rejects it.")
    if data_mode == "synthetic_fixture" and model_builder is None:
        raise ValueError("Synthetic fixture training requires an explicit test model builder.")
    training, validation = AuditedPNGDataset(context, "train"), AuditedPNGDataset(context, "validation")
    reporter = _Progress(progress, {"data_mode": data_mode, "metric_scope": METRIC_SCOPE})
    reporter.emit("initial", phase="initialization", train_patient_count=len(training.patient_ids),
                  validation_patient_count=len(validation.patient_ids), train_slice_count=len(training),
                  validation_slice_count=len(validation), test_prediction_pixels_loaded=False,
                  epochs=epochs, device="cpu", num_threads=num_threads, batch_size=batch_size,
                  learning_rate=lr, seed=seed)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    started = _utc()
    try:
        torch.manual_seed(seed)
        builder = model_builder if model_builder is not None else build_verified_model
        model, proof = builder(input_channels=1, cache_root=cache_root)
        _proof_valid(proof, data_mode)
        if getattr(model, "input_channels", None) != 1:
            raise ValueError("The PNG model must have exactly one input channel.")
        if data_mode == "real" and getattr(model, "architecture", None) != "smp_unet_resnet18":
            raise ValueError("Real PNG training requires the verified ResNet-18 U-Net architecture.")
        model.to(device)
        model.png_cohort_fingerprint_sha256 = context["cohort"]["cohort_fingerprint_sha256"]
        groups = _trainable_groups(model)
        encoder_before = _state(model.network.encoder)
        _check_encoder(model, encoder_before)
        initial_trainable = {name: p.detach().cpu().clone() for group in groups.values() for name, p in group}
        optimizer = torch.optim.Adam([p for group in groups.values() for _, p in group], lr=lr)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator,
                            drop_last=False, num_workers=0)
        best_loss, best_epoch, best_state, best_report = float("inf"), None, None, None
        history, optimizer_steps, last_report = [], 0, None
        for epoch in range(1, epochs + 1):
            _check_bindings(context)
            model.train()
            _check_encoder(model, encoder_before)
            seen, loss_sum, sample_counts = 0, 0.0, Counter()
            epoch_started = _utc()
            reporter.emit("epoch_start", phase="training", epoch=epoch, epochs=epochs,
                          total_batches=len(loader), total_slices=len(training))
            for batch_index, batch in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["input"].to(device))
                target = batch["target"].to(device)
                loss = segmentation_loss(logits, target)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise ValueError("Training logits/loss became non-finite.")
                loss.backward()
                _check_gradients(groups)
                optimizer.step()
                if any(not torch.isfinite(p).all() for group in groups.values() for _, p in group):
                    raise ValueError("Trainable parameters became non-finite after an optimizer step.")
                optimizer_steps += 1
                seen += len(target)
                sample_counts.update(batch["sample_id"])
                loss_sum += float(loss.detach()) * len(target)
                reporter.batch(batch_index, len(loader), phase="training", epoch=epoch, epochs=epochs,
                               slices_seen=seen, total_slices=len(training), loss=loss_sum / seen,
                               optimizer_steps_completed=optimizer_steps, progress=seen / len(training))
            expected_samples = {"%s:z%d" % (r["case"], r["z"]) for r in training.rows}
            if seen != len(training) or set(sample_counts) != expected_samples or any(n != 1 for n in sample_counts.values()):
                raise ValueError("Every exported training slice must be used exactly once per epoch.")
            _check_encoder(model, encoder_before)
            last_report = evaluate_png_validation(model, validation, batch_size=batch_size, device=device,
                                                  progress=progress, epoch=epoch, epochs=epochs)
            _check_encoder(model, encoder_before)
            record = {"epoch": epoch, "epochs": epochs, "train_loss": loss_sum / seen,
                      "validation_loss": last_report["loss"], "train_slices_seen": seen,
                      "validation_slices_seen": last_report["slice_count"],
                      "validation_patient_mean": last_report["patient_mean"],
                      "optimizer_steps_completed": optimizer_steps, "started_at_utc": epoch_started,
                      "completed_at_utc": _utc(), "metric_scope": METRIC_SCOPE}
            history.append(record)
            if record["validation_loss"] < best_loss:
                best_loss, best_epoch, best_state = record["validation_loss"], epoch, _state(model)
                best_report = copy.deepcopy(last_report)
            reporter.emit("epoch_completed", phase="epoch_summary", **record)
        _check_bindings(context)
        model.load_state_dict(best_state, strict=True)
        model.eval()
        _check_encoder(model, encoder_before)
        changed = {name: any(not torch.equal(p.detach().cpu(), initial_trainable[key]) for key, p in entries)
                   for name, entries in groups.items()}
        if not all(changed.values()):
            raise ValueError("Both decoder and segmentation head must update during training.")
        metadata = {
            "schema_version": SCHEMA, "architecture": model.architecture, "input_channels": 1,
            "input_mode": "single_2d_slice", "classes": CLASSES, "data_mode": data_mode,
            "preprocessing": context["config"].to_dict(), "preprocessing_sha256": training.config_sha256,
            "audit": copy.deepcopy(context["audit"]), "cohort": copy.deepcopy(context["cohort"]),
            "source_root": str(context["root"]),
            "bound_file_paths": {key: str(path) for key, path in context["paths"].items()},
            "bound_file_sha256": context["file_sha256"].copy(), "pretrained_proof": copy.deepcopy(proof),
            "encoder_frozen": True, "encoder_state_unchanged": True,
            "encoder_initial_state_sha256": _state_hash(encoder_before),
            "encoder_saved_state_sha256": _state_hash(_state(model.network.encoder)),
            "finite_nonzero_decoder_head_gradients": True, "decoder_updated": changed["decoder"],
            "segmentation_head_updated": changed["head"], "gradient_checks_completed": optimizer_steps,
            "state_fingerprint_sha256": _state_hash(best_state),
            "epochs": epochs, "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "batch_size": batch_size, "learning_rate": lr, "seed": seed, "device": "cpu", "num_threads": num_threads,
            "optimizer": "Adam", "optimizer_steps_completed": optimizer_steps,
            "sampling": "all exported training slices once per epoch; shuffled without replacement; no label-based selection/oversampling",
            "drop_last": False, "train_slice_count": len(training), "validation_slice_count": len(validation),
            "history": history, "best_validation_report": best_report, "metric_scope": METRIC_SCOPE,
            "test_evaluated": False, "prediction_run": False, "test_prediction_pixels_loaded": False,
            "clinical_performance_evidence": False, "anatomical_alignment_verified": False,
            "manual_review_status": "pending", "visual_review_note": visual_review_note,
            "full_scan_coverage": False, "source_patient_aliases_verified": False,
            "library_versions": {name: version(name) for name in
                                 ("torch", "torchvision", "segmentation-models-pytorch", "numpy", "Pillow", "safetensors")},
            "started_at_utc": started, "completed_at_utc": _utc(),
            "limitations": ["Reference-selected exports do not establish full-scan segmentation performance.",
                            "Validation selects the checkpoint and is not an untouched final test.",
                            "One epoch is a baseline execution, not established clinical performance.",
                            "Machine hashes/counts and training do not certify anatomical annotation alignment."]}
        metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
        model.png_metadata = copy.deepcopy(metadata)
        model.pretrained_proof = copy.deepcopy(proof)
        reporter.emit("completed", phase="completed", best_epoch=best_epoch, best_validation_loss=best_loss,
                      optimizer_steps_completed=optimizer_steps, test_prediction_pixels_loaded=False)
        return {"model": model, "history": history, "metadata": metadata, "proof": proof,
                "validation_report": best_report, "best_validation_report": best_report,
                "last_validation_report": last_report}
    finally:
        torch.set_num_threads(previous_threads)


def _validate_metadata(metadata):
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA or
            metadata.get("input_channels") != 1 or metadata.get("input_mode") != "single_2d_slice" or
            metadata.get("classes") != CLASSES):
        raise ValueError("Unsupported single-channel PNG checkpoint metadata.")
    if canonical_fingerprint(metadata, "metadata_fingerprint_sha256") != metadata.get("metadata_fingerprint_sha256"):
        raise ValueError("PNG checkpoint metadata fingerprint is stale.")
    validate_cohort(metadata["cohort"], metadata["audit"])
    if metadata.get("data_mode") not in ("real", "synthetic_fixture"):
        raise ValueError("Invalid PNG checkpoint data mode.")
    fixture = metadata["audit"].get("data_origin") == "synthetic_fixture"
    if fixture != (metadata["data_mode"] == "synthetic_fixture"):
        raise ValueError("Checkpoint fixture/real audit origin is inconsistent.")
    _proof_valid(metadata["pretrained_proof"], metadata["data_mode"])
    config = PNGConfig(**metadata["preprocessing"])
    _check_audit_preprocessing(metadata["audit"], config)
    if (config.to_dict() != metadata["cohort"]["preprocessing"]["config"] or
            metadata["preprocessing_sha256"] != metadata["cohort"]["preprocessing"]["config_sha256"]):
        raise ValueError("Checkpoint preprocessing differs from the locked cohort.")
    if metadata["encoder_initial_state_sha256"] != metadata["encoder_saved_state_sha256"]:
        raise ValueError("Checkpoint encoder provenance changed during training.")
    if metadata["data_mode"] == "real" and metadata["architecture"] != "smp_unet_resnet18":
        raise ValueError("Unsupported real PNG checkpoint architecture.")
    return config


def save_png_checkpoint(path, model, metadata):
    """Atomically create a new PNG checkpoint; existing paths are never replaced."""
    _validate_metadata(metadata)
    if getattr(model, "png_metadata", None) != metadata:
        raise ValueError("PNG checkpoint metadata differs from the fitted model.")
    state = _state(model)
    if _state_hash(state) != metadata["state_fingerprint_sha256"]:
        raise ValueError("Model state differs from the selected best-validation checkpoint.")
    path = Path(os.path.abspath(str(Path(path).expanduser())))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("PNG checkpoint output paths must not contain symlinks.")
    source_root = Path(metadata["source_root"])
    if path == source_root or source_root in path.parents:
        raise ValueError("Save checkpoints outside the original data project.")
    if path.exists():
        raise FileExistsError("PNG checkpoint path already exists; use a new run directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".png-checkpoint.tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save({"schema_version": SCHEMA, "metadata": metadata, "state_dict": state}, temporary)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    digest = _hash(path.read_bytes())
    model.png_checkpoint_sha256 = digest
    return {"path": str(path), "sha256": digest, "schema_version": SCHEMA}


def load_png_checkpoint(path, device="cpu", model_builder=None):
    """Restore complete weights without downloading or calling pretrained init."""
    if str(device) != "cpu":
        raise ValueError("This PNG checkpoint loader currently supports CPU only.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported PNG checkpoint schema; the 2.5D schema is separate.")
    metadata = payload.get("metadata")
    config = _validate_metadata(metadata)
    if _state_hash(payload["state_dict"]) != metadata["state_fingerprint_sha256"]:
        raise ValueError("PNG checkpoint tensors differ from their saved state fingerprint.")
    if metadata["data_mode"] == "synthetic_fixture":
        if model_builder is None:
            raise ValueError("Synthetic fixture checkpoint loading requires its test model builder.")
        model, _ = model_builder(input_channels=1, cache_root=None)
    else:
        if model_builder is not None:
            raise ValueError("Real checkpoint loading cannot inject a substitute model.")
        model = KidneyUNet(pretrained=False, input_channels=1)
    if model.architecture != metadata["architecture"] or model.input_channels != 1:
        raise ValueError("Checkpoint/model architecture or input channel mismatch.")
    model.load_state_dict(payload["state_dict"], strict=True)
    if _state_hash(_state(model.network.encoder)) != metadata["encoder_saved_state_sha256"]:
        raise ValueError("Restored encoder differs from the frozen provenance fingerprint.")
    model.pretrained = metadata["pretrained_proof"]["pretrained_encoder"]
    model.pretrained_proof = copy.deepcopy(metadata["pretrained_proof"])
    model.png_metadata = copy.deepcopy(metadata)
    model.png_cohort_fingerprint_sha256 = metadata["cohort"]["cohort_fingerprint_sha256"]
    model.png_checkpoint_sha256 = _hash(Path(path).read_bytes())
    model.eval()
    _check_encoder(model, _state(model.network.encoder))
    return model, config, metadata
