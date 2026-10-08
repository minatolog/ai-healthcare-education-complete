"""Freeze a completed PNG run and evaluate only its locked test patients.

Inspection and freezing read metadata and saved tensors, never initialize a
model or decode a PNG. Evaluation is a separate, read-only operation: the
checkpoint is already selected by validation, and test metrics cannot select
weights, a threshold, a loss setting or a patient pool. Callers own write-once
freeze/output persistence and the final-test access ledger.
"""
from collections import Counter
import copy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import CLASSES
from .cohort import canonical_fingerprint, ROLES
from .metrics import confusion_matrix, patient_report, summarize_patients
from .model import segmentation_loss
from .png_dataset import PNGPatientDataset
from . import png_training as training

FREEZE_SCHEMA = "frozen_png_model_v1"
TEST_SCHEMA = "png_frozen_test_v1"
RUN_SCOPE = "selected_exported_2d_png_segmentation"
ARTIFACT_NAMES = ("best_model.pt", "training_history.json", "validation_report.json")
FINGERPRINT_FIELD = "freeze_fingerprint_sha256"


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _json_object(content, description):
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError):
        raise ValueError("%s must be valid JSON." % description) from None
    if not isinstance(value, dict):
        raise ValueError("%s must be a JSON object." % description)
    return value


def _json_equal(first, second):
    """Compare JSON evidence without conflating booleans with numeric values."""
    try:
        return (json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False)
                == json.dumps(second, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError):
        return False


def _regular_path(path, description):
    original = Path(path).expanduser().absolute()
    if any(part.is_symlink() for part in (original, *original.parents)):
        raise ValueError("%s paths must not contain symlinks." % description)
    resolved = original.resolve()
    if not resolved.is_file():
        raise ValueError("%s must be an existing regular file." % description)
    return resolved


def _pool_evidence(context):
    pools = context["cohort"]["pools"]
    ids = {role: [row["patient_id"] for row in pools[role]] for role in ROLES}
    counts = {role: len(ids[role]) for role in ROLES}
    if counts != {"train": 20, "validation": 10, "test": 10, "prediction": 5}:
        raise ValueError("Final testing requires the locked 20/10/10 plus 5 prediction patient pools.")
    slices = {role: sum(row["selected_slice_count"] for row in pools[role]) for role in ROLES}
    return ids, counts, slices


def _validate_run_evidence(context, config, metadata, summary, history_artifact,
                           validation_artifact, checkpoint_path, artifact_hashes):
    """Require the completed run, selected report and current lock to agree."""
    ids, counts, slices = _pool_evidence(context)
    if (summary.get("schema_version") != "png_training_run_v1"
            or summary.get("status") != "completed"
            or summary.get("formal_training_completed") is not True
            or summary.get("scope") != RUN_SCOPE):
        raise ValueError("A completed PNG training run with the selected-exported-slice scope is required.")
    checkpoint = summary.get("checkpoint")
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("sha256") != artifact_hashes["best_model.pt"]
            or checkpoint.get("path") != str(checkpoint_path)
            or checkpoint.get("schema_version") != training.SCHEMA
            or not _json_equal(summary.get("artifact_sha256"), artifact_hashes)):
        raise ValueError("Completed run checkpoint/artifact hashes or paths differ from the actual artifacts.")
    bound_paths = {name: str(path) for name, path in context["paths"].items()}
    if (not _json_equal(metadata.get("bound_file_sha256"), context["file_sha256"])
            or not _json_equal(metadata.get("bound_file_paths"), bound_paths)
            or metadata.get("source_root") != str(context["root"])
            or config != context["config"]
            or not _json_equal(metadata.get("cohort"), context["cohort"])
            or not _json_equal(metadata.get("audit"), context["audit"])
            or summary.get("cohort_fingerprint_sha256") != context["cohort"]["cohort_fingerprint_sha256"]
            or summary.get("audit_fingerprint_sha256") != context["audit"]["audit_fingerprint_sha256"]):
        raise ValueError("Checkpoint/run metadata differs from the current bound cohort/audit/CSV/split inputs.")
    if (not _json_equal(summary.get("patient_counts"), counts)
            or not _json_equal(summary.get("selected_slice_counts"), slices)
            or metadata.get("train_slice_count") != slices["train"]
            or metadata.get("validation_slice_count") != slices["validation"]):
        raise ValueError("Completed run patient/slice counts differ from the locked pools.")
    if (any(metadata.get(field) is not False for field in
            ("test_evaluated", "prediction_run", "test_prediction_pixels_loaded"))
            or summary.get("final_test_performed") is not False
            or summary.get("independent_prediction_performed") is not False
            or metadata.get("metric_scope") != training.METRIC_SCOPE):
        raise ValueError("Freeze requires an untested run with the unchanged exported PNG metric scope.")
    selection = metadata.get("checkpoint_selection")
    loss_config = metadata.get("loss_configuration")
    if not isinstance(selection, dict) or not isinstance(loss_config, dict) or "tumour_ce_weight" not in metadata:
        raise ValueError("Freeze requires explicit validation selection and loss-configuration evidence.")
    # _validate_metadata has already checked the ranking over the whole history,
    # including the tie rule and the selected report/epoch.
    expected_config = {
        "epochs": metadata.get("epochs"), "batch_size": metadata.get("batch_size"),
        "learning_rate": metadata.get("learning_rate"), "seed": metadata.get("seed"),
        "device": metadata.get("device"), "cpu_threads": metadata.get("num_threads"),
        "selection_metric": selection["metric"], "tumour_ce_weight": metadata["tumour_ce_weight"],
        "training_sampling": "all exported training rows, shuffled once per epoch; no label-based sampling",
        "resume": False,
    }
    if (not _json_equal(summary.get("configuration"), expected_config)
            or summary.get("epochs_completed") != metadata.get("epochs")
            or summary.get("best_epoch") != metadata.get("best_epoch")
            or not _json_equal(summary.get("checkpoint_selection"), selection)
            or not _json_equal(summary.get("loss_configuration"), loss_config)
            or not _json_equal(summary.get("pretrained_encoder_proof"), metadata["pretrained_proof"])):
        raise ValueError("Completed run configuration or selected-checkpoint evidence differs from metadata.")
    if (history_artifact.get("schema_version") != "png_training_history_v1"
            or history_artifact.get("scope") != RUN_SCOPE
            or not _json_equal(history_artifact.get("configuration"), expected_config)
            or not _json_equal(history_artifact.get("history"), metadata["history"])):
        raise ValueError("Training history artifact differs from the selected checkpoint history/configuration.")
    for row in metadata["history"]:
        if (row.get("train_slices_seen") != slices["train"]
                or row.get("validation_slices_seen") != slices["validation"]
                or row.get("metric_scope") != training.METRIC_SCOPE):
            raise ValueError("Epoch history must preserve all exported train/validation rows and metric scope.")
    report = metadata["best_validation_report"]
    if (not _json_equal(validation_artifact, report)
            or not _json_equal(summary.get("validation_patient_mean"), report.get("patient_mean"))):
        raise ValueError("Selected validation report artifact or summary differs from checkpoint evidence.")
    patients = report.get("patients")
    if (report.get("split") != "validation"
            or report.get("slice_count") != slices["validation"]
            or report.get("patient_count") != counts["validation"]
            or report.get("metric_scope") != training.METRIC_SCOPE
            or report.get("model_selection_only") is not True
            or not isinstance(patients, list)
            or [row.get("patient_id") for row in patients] != ids["validation"]):
        raise ValueError("Selected validation report patient membership/counts/scope are inconsistent.")
    expected_slices = {row["patient_id"]: row["selected_slice_count"]
                       for row in context["cohort"]["pools"]["validation"]}
    if any(row.get("exported_slice_count") != expected_slices[row["patient_id"]] for row in patients):
        raise ValueError("Selected validation report does not cover each patient's exported slices.")


def inspect_completed_run(root, cohort_path, audit_path, run_dir, *, allow_fixture=False):
    """Validate metadata, artifact bytes and saved state; no model or PNG decode."""
    if type(allow_fixture) is not bool:
        raise ValueError("allow_fixture must be an explicit boolean.")
    context = training.load_training_context(root, cohort_path, audit_path)
    run_dir = Path(run_dir).expanduser().absolute()
    if any(part.is_symlink() for part in (run_dir, *run_dir.parents)) or not run_dir.is_dir():
        raise ValueError("The completed run directory must exist without symlinks.")
    run_dir = run_dir.resolve()
    paths = {name: _regular_path(run_dir / name, "Training artifact") for name in ARTIFACT_NAMES}
    summary_path = _regular_path(run_dir / "run_summary.json", "Run summary")
    # Read each artifact once for this snapshot. weights_only rejects executable
    # pickle objects; BytesIO binds the inspected tensor payload to these bytes.
    content = {name: path.read_bytes() for name, path in paths.items()}
    summary_bytes = summary_path.read_bytes()
    hashes = {name: _hash(value) for name, value in content.items()}
    summary = _json_object(summary_bytes, "Run summary")
    history = _json_object(content["training_history.json"], "Training history")
    validation = _json_object(content["validation_report.json"], "Validation report")
    payload = torch.load(io.BytesIO(content["best_model.pt"]), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != training.SCHEMA:
        raise ValueError("A single-channel PNG checkpoint is required; volume/2.5D checkpoints are separate.")
    metadata = payload.get("metadata")
    config = training._validate_metadata(metadata)
    state = payload.get("state_dict")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items())
            or training._state_hash(state) != metadata.get("state_fingerprint_sha256")):
        raise ValueError("PNG checkpoint tensors differ from their saved state fingerprint.")
    prefix = "network.encoder."
    encoder_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if (not encoder_state
            or training._state_hash(encoder_state) != metadata["encoder_saved_state_sha256"]):
        raise ValueError("PNG checkpoint encoder tensors differ from the frozen provenance fingerprint.")
    if metadata["data_mode"] != "real" and not allow_fixture:
        raise ValueError("This final-test entry rejects synthetic fixtures unless allow_fixture=True is explicit.")
    _validate_run_evidence(context, config, metadata, summary, history, validation,
                           paths["best_model.pt"], hashes)
    inspected = {
        "context": context, "config": config, "metadata": metadata, "summary": summary,
        "run_dir": run_dir, "checkpoint_path": paths["best_model.pt"],
        "checkpoint_sha256": hashes["best_model.pt"], "run_summary_path": summary_path,
        "run_summary_sha256": _hash(summary_bytes), "training_artifact_sha256": hashes.copy(),
        "artifact_paths": paths, "artifact_sha256": hashes.copy(), "allow_fixture": allow_fixture,
    }
    _check_snapshot(inspected)
    return inspected


def _check_snapshot(inspected, *, require_freeze=False):
    """Detect binding/byte drift before and after any pixel/model operation."""
    training._check_bindings(inspected["context"])
    for name, path in inspected["artifact_paths"].items():
        checked = _regular_path(path, "Training artifact")
        if _hash(checked.read_bytes()) != inspected["artifact_sha256"][name]:
            raise ValueError("A completed training artifact changed after inspection: %s." % name)
    summary = _regular_path(inspected["run_summary_path"], "Run summary")
    if _hash(summary.read_bytes()) != inspected["run_summary_sha256"]:
        raise ValueError("The completed run summary changed after inspection.")
    if require_freeze:
        if "freeze_record" not in inspected or "freeze_path" not in inspected or "freeze_sha256" not in inspected:
            raise ValueError("Final test evaluation requires an inspected persisted freeze record.")
        freeze_path = _regular_path(inspected["freeze_path"], "Freeze record")
        content = freeze_path.read_bytes()
        if _hash(content) != inspected["freeze_sha256"]:
            raise ValueError("The persisted freeze record changed after inspection.")
        record = _json_object(content, "Freeze record")
        if not _json_equal(record, inspected["freeze_record"]):
            raise ValueError("The inspected freeze record differs from its persisted snapshot.")
        validate_freeze_record(record, inspected)


def make_freeze_record(inspected):
    """Return a JSON-safe snapshot referencing the original checkpoint bytes."""
    context, metadata = inspected["context"], inspected["metadata"]
    ids, counts, slices = _pool_evidence(context)
    record = {
        "schema_version": FREEZE_SCHEMA, "source_run_dir": str(inspected["run_dir"]),
        "checkpoint_path": str(inspected["checkpoint_path"]),
        "checkpoint_sha256": inspected["checkpoint_sha256"],
        "run_summary_sha256": inspected["run_summary_sha256"],
        "training_artifact_sha256": copy.deepcopy(inspected["training_artifact_sha256"]),
        "metadata_fingerprint_sha256": metadata["metadata_fingerprint_sha256"],
        "state_fingerprint_sha256": metadata["state_fingerprint_sha256"],
        "selected_epoch": metadata["best_epoch"],
        "checkpoint_selection": copy.deepcopy(metadata["checkpoint_selection"]),
        "tumour_ce_weight": metadata["tumour_ce_weight"],
        "loss_configuration": copy.deepcopy(metadata["loss_configuration"]),
        "architecture": metadata["architecture"], "input_channels": metadata["input_channels"],
        "input_mode": metadata["input_mode"], "classes": {str(key): value for key, value in CLASSES.items()},
        "data_mode": metadata["data_mode"], "pretrained_proof": copy.deepcopy(metadata["pretrained_proof"]),
        "patient_ids": ids, "patient_counts": counts, "selected_slice_counts": slices,
        "pools": copy.deepcopy(context["cohort"]["pools"]),
        "bound_file_paths": {name: str(path) for name, path in context["paths"].items()},
        "bound_file_sha256": context["file_sha256"].copy(),
        "preprocessing": context["config"].to_dict(),
        "preprocessing_sha256": metadata["preprocessing_sha256"],
        "cohort_fingerprint_sha256": context["cohort"]["cohort_fingerprint_sha256"],
        "audit_fingerprint_sha256": context["audit"]["audit_fingerprint_sha256"],
        "metric_scope": training.METRIC_SCOPE, "frozen_at_utc": training._utc(),
    }
    record[FINGERPRINT_FIELD] = canonical_fingerprint(record, FINGERPRINT_FIELD)
    return record


def validate_freeze_record(record, inspected):
    """Reject stale or rehashed drift against the inspected training snapshot."""
    if not isinstance(record, dict) or record.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("Unsupported frozen PNG model record.")
    try:
        valid_fingerprint = canonical_fingerprint(record, FINGERPRINT_FIELD) == record.get(FINGERPRINT_FIELD)
    except (TypeError, ValueError):
        valid_fingerprint = False
    if not valid_fingerprint:
        raise ValueError("Frozen model record fingerprint is stale.")
    timestamp = record.get("frozen_at_utc")
    try:
        parsed = datetime.fromisoformat(timestamp)
        valid_timestamp = isinstance(timestamp, str) and parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0
    except (TypeError, ValueError):
        valid_timestamp = False
    if not valid_timestamp:
        raise ValueError("Frozen model record requires an explicit UTC freeze timestamp.")
    expected = make_freeze_record(inspected)
    # A timestamp is an event rather than a property of the trained model.
    # All other keys (including unknown additions) must match exactly, even if
    # someone has recomputed the record's self-referential fingerprint.
    expected["frozen_at_utc"] = timestamp
    expected[FINGERPRINT_FIELD] = canonical_fingerprint(expected, FINGERPRINT_FIELD)
    if not _json_equal(record, expected):
        raise ValueError("Frozen model record differs from the exact inspected checkpoint/configuration/pool snapshot.")


def inspect_frozen_model(root, cohort_path, audit_path, run_dir, freeze_path, *, allow_fixture=False):
    """Check persisted freezing and current bindings without model/PNG loading."""
    inspected = inspect_completed_run(root, cohort_path, audit_path, run_dir, allow_fixture=allow_fixture)
    path = _regular_path(freeze_path, "Freeze record")
    content = path.read_bytes()
    record = _json_object(content, "Freeze record")
    validate_freeze_record(record, inspected)
    inspected.update(freeze_record=record, freeze_path=path, freeze_sha256=_hash(content))
    _check_snapshot(inspected, require_freeze=True)
    return inspected


class FrozenTestPNGDataset(training.AuditedPNGDataset):
    """Dedicated test allowlist; the training dataset's role gate stays intact."""
    def __init__(self, inspected):
        _check_snapshot(inspected, require_freeze=True)
        context = inspected["context"]
        self.context, self.role = context, "test"
        records = context["cohort"]["pools"]["test"]
        # Intentionally bypass only the training constructor's train/validation
        # role restriction. Its per-access audit, byte/pixel, label and path
        # checks are inherited unchanged.
        PNGPatientDataset.__init__(self, context["paths"]["manifest"],
                                   [row["patient_id"] for row in records],
                                   data_root=context["data_root"], config=context["config"])
        if (self.manifest_sha256 != context["audit"]["manifest_sha256"]
                or self.config_sha256 != context["cohort"]["preprocessing"]["config_sha256"]):
            raise ValueError("Frozen test loader manifest/preprocessing differs from the locked audit.")
        index = {row["patient_id"]: row for row in context["audit"]["patients"]}
        counts = Counter(row["case"] for row in self.rows)
        for record in records:
            patient = record["patient_id"]
            audited = {(row["slice_index"], row["image"], row["mask"])
                       for row in index[patient]["slice_assets"]}
            observed = {(row["z"], row["image"], row["mask"])
                        for row in self.rows if row["case"] == patient}
            if counts[patient] != record["selected_slice_count"] or observed != audited:
                raise ValueError("Frozen test patient rows/assets differ from the exact locked cohort/audit.")
        if len(self) != inspected["freeze_record"]["selected_slice_counts"]["test"]:
            raise ValueError("Frozen test slice total differs from the freeze record.")
        for row in self.rows:
            self._paths_and_facts(row)


def evaluate_frozen_test(inspected, *, batch_size=2, num_threads=2, progress=None, model_builder=None):
    """Score every fixed test row exactly once, with no training or selection."""
    for value, name in ((batch_size, "batch_size"), (num_threads, "num_threads")):
        if type(value) is not int or value < 1:
            raise ValueError("%s must be a positive integer." % name)
    _check_snapshot(inspected, require_freeze=True)
    metadata = inspected["metadata"]
    fixture = metadata["data_mode"] == "synthetic_fixture"
    if (fixture and inspected.get("allow_fixture") is not True) or (model_builder is not None and not fixture):
        raise ValueError("Injected final-test models are permitted only for explicitly allowed synthetic fixtures.")
    if fixture and model_builder is None:
        raise ValueError("Synthetic fixture final testing requires its explicit test model builder.")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    reporter = training._Progress(progress, {"phase": "final_test", "metric_scope": training.METRIC_SCOPE,
                                            "training_performed": False, "model_selection_performed": False})
    try:
        model, config, loaded_metadata = training.load_png_checkpoint(
            inspected["checkpoint_path"], device="cpu", model_builder=model_builder)
        if (config != inspected["config"] or not _json_equal(loaded_metadata, metadata)
                or model.png_checkpoint_sha256 != inspected["checkpoint_sha256"]):
            raise ValueError("Loaded model differs from the inspected frozen checkpoint.")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = training._state_hash(training._state(model))
        if before != metadata["state_fingerprint_sha256"]:
            raise ValueError("Loaded model state differs from the original frozen state fingerprint.")
        _check_snapshot(inspected, require_freeze=True)
        dataset = FrozenTestPNGDataset(inspected)
        matrices = {patient: np.zeros((3, 3), dtype=np.int64) for patient in dataset.patient_ids}
        counts, sample_counts, seen, loss_sum = Counter(), Counter(), 0, 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        weight = metadata["tumour_ce_weight"]
        reporter.emit("initial", test_patient_count=len(dataset.patient_ids), total_slices=len(dataset),
                      batch_size=batch_size, num_threads=num_threads, optimizer_steps=0,
                      checkpoint_sha256=inspected["checkpoint_sha256"])
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, 1):
                # Only CT-derived inputs go to the model. Reference masks are
                # used after inference for the fixed loss and pixel scores.
                logits = model(batch["input"])
                target = batch["target"]
                if logits.shape != (len(target), 3, 256, 256):
                    raise ValueError("Frozen test model must return three-class 256x256 single-slice logits.")
                loss = segmentation_loss(logits, target, tumour_ce_weight=weight)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise ValueError("Frozen test logits/loss became non-finite.")
                predicted = logits.argmax(1).cpu().numpy()
                for patient, actual, prediction in zip(batch["patient_id"], target.cpu().numpy(), predicted):
                    if patient not in matrices:
                        raise ValueError("Final test loading escaped the fixed test patient allowlist.")
                    matrices[patient] += confusion_matrix(actual, prediction)
                    counts[patient] += 1
                sample_counts.update(batch["sample_id"])
                seen += len(target)
                loss_sum += float(loss) * len(target)
                reporter.batch(batch_index, len(loader), slices_seen=seen, total_slices=len(dataset),
                               loss=loss_sum / seen, progress=seen / len(dataset), optimizer_steps=0)
        expected_counts = Counter(row["case"] for row in dataset.rows)
        expected_samples = {"%s:z%d" % (row["case"], row["z"]) for row in dataset.rows}
        if (counts != expected_counts or seen != len(dataset) or set(sample_counts) != expected_samples
                or any(count != 1 for count in sample_counts.values())):
            raise ValueError("Final testing must use every frozen test slice exactly once.")
        reports = []
        for patient, matrix in matrices.items():
            item = patient_report(patient, matrix)
            item["exported_slice_count"] = counts[patient]
            item["confusion_matrix_total_pixels"] = int(matrix.sum())
            if item["confusion_matrix_total_pixels"] != counts[patient] * 256 * 256:
                raise ValueError("Patient confusion matrix does not account for every exported pixel.")
            reports.append(item)
        after = training._state_hash(training._state(model))
        if after != before or after != metadata["state_fingerprint_sha256"]:
            raise ValueError("Frozen model parameters or buffers changed during final test inference.")
        training._check_encoder(model, training._state(model.network.encoder))
        _check_snapshot(inspected, require_freeze=True)
        report = summarize_patients(reports)
        report.update(
            schema_version=TEST_SCHEMA, status="completed", split="test", data_mode=metadata["data_mode"],
            scope=RUN_SCOPE, metric_scope=training.METRIC_SCOPE, slice_count=seen,
            patient_ids=list(dataset.patient_ids), patient_counts=copy.deepcopy(inspected["freeze_record"]["patient_counts"]),
            selected_slice_counts=copy.deepcopy(inspected["freeze_record"]["selected_slice_counts"]),
            decoded_pairs={role: seen if role == "test" else 0 for role in ROLES},
            sample_counts=dict(sample_counts), patient_slice_counts=dict(counts),
            confusion_matrix_total_pixels=sum(int(matrix.sum()) for matrix in matrices.values()),
            loss=loss_sum / seen, tumour_ce_weight=weight,
            loss_configuration=copy.deepcopy(metadata["loss_configuration"]),
            loss_scope="slice-weighted mean of minibatch frozen class-weighted cross entropy plus equally averaged kidney/tumour soft-Dice; descriptive only, never checkpoint selection",
            checkpoint_path=str(inspected["checkpoint_path"]), checkpoint_sha256=inspected["checkpoint_sha256"],
            frozen_model_sha256=inspected["checkpoint_sha256"],
            state_fingerprint_sha256=metadata["state_fingerprint_sha256"],
            model_state_before_sha256=before, model_state_after_sha256=after, model_state_unchanged=True,
            freeze_path=str(inspected["freeze_path"]), freeze_sha256=inspected["freeze_sha256"],
            freeze_fingerprint_sha256=inspected["freeze_record"][FINGERPRINT_FIELD],
            run_summary_sha256=inspected["run_summary_sha256"],
            training_artifact_sha256=copy.deepcopy(inspected["training_artifact_sha256"]),
            bound_file_sha256=inspected["context"]["file_sha256"].copy(),
            cohort_fingerprint_sha256=inspected["context"]["cohort"]["cohort_fingerprint_sha256"],
            audit_fingerprint_sha256=inspected["context"]["audit"]["audit_fingerprint_sha256"],
            selected_epoch=metadata["best_epoch"], checkpoint_selection=copy.deepcopy(metadata["checkpoint_selection"]),
            preprocessing=inspected["config"].to_dict(), preprocessing_sha256=metadata["preprocessing_sha256"],
            input_source="CT PNG only; reference segmentation is used only for scoring",
            inference_rule="hard_argmax_over_three_fixed_class_logits", batch_size=batch_size,
            num_threads=num_threads, device="cpu", optimizer_steps=0, training_performed=False,
            model_selection_performed=False, model_selection_only=False, threshold_tuning_performed=False,
            full_scan_coverage=False, clinical_performance_evidence=False, anatomical_alignment_verified=False,
            patient_level_diagnosis=False, prediction_pool_pixels_loaded=False,
            source_png_byte_and_decoded_pixel_hashes_verified=True, bound_inputs_unchanged=True,
            created_at_utc=training._utc(),
            limitations=["Metrics cover only the locked reference-selected exported 2D PNG slices.",
                         "This educational baseline does not establish full-volume or clinical performance.",
                         "Pixel segmentation metrics do not establish a patient-level diagnosis or healthy-patient label.",
                         "Machine hashes/counts do not certify anatomical annotation alignment or original patient aliases."])
        reporter.emit("completed", slices_seen=seen, test_patient_count=len(dataset.patient_ids),
                      model_state_unchanged=True, optimizer_steps=0)
        # Progress callbacks may touch files: include the completion callback in
        # the protected interval as well.
        _check_snapshot(inspected, require_freeze=True)
        if training._state_hash(training._state(model)) != before:
            raise ValueError("Frozen model parameters or buffers changed during final test inference.")
        return report
    finally:
        torch.set_num_threads(previous_threads)
