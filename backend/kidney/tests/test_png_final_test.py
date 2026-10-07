"""Frozen final-test engineering checks with synthetic PNGs and tiny weights.

The completed-run fixture is assembled from declared synthetic evidence.  It
never calls training, downloads pretrained weights, or reads a real CT dataset.
Only explicit final-test calls below run the tiny model on synthetic test PNGs.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from kidney_segmentation import CLASSES
from kidney_segmentation import png_dataset
from kidney_segmentation import png_final_test as final_test
from kidney_segmentation import png_training as training
from kidney_segmentation.cohort import canonical_fingerprint
from kidney_segmentation.metrics import METRIC_NAMES, patient_report, summarize_patients
from test_png_training import TinyFixtureModel, fixture_data, tiny_builder


STAMP = "2026-10-07T00:00:00+00:00"
SCOPE = "selected_exported_2d_png_segmentation"
ARTIFACTS = ("best_model.pt", "training_history.json", "validation_report.json")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prohibit(*args, **kwargs):
    pytest.fail("This metadata-only operation decoded pixels or initialized a model")


def prohibit_model_and_pixels(monkeypatch):
    monkeypatch.setattr(png_dataset, "_read_png", prohibit)
    monkeypatch.setattr(Image, "open", prohibit)
    monkeypatch.setattr(TinyFixtureModel, "__init__", prohibit)
    monkeypatch.setattr(training, "build_verified_model", prohibit)
    monkeypatch.setattr(training, "KidneyUNet", prohibit)


def inspected(run, *, allow_fixture=True):
    return final_test.inspect_completed_run(run["root"], run["cohort_path"], run["audit_path"],
                                            run["run_dir"], allow_fixture=allow_fixture)


def frozen_inspected(run, *, allow_fixture=True):
    return final_test.inspect_frozen_model(run["root"], run["cohort_path"], run["audit_path"],
                                           run["run_dir"], run["freeze_path"], allow_fixture=allow_fixture)


def save_freeze(run):
    loaded = inspected(run)
    record = final_test.make_freeze_record(loaded)
    write_json(run["freeze_path"], record)
    return loaded, record


def declared_validation_report(data, context):
    """Build fixture metrics from audited counts, without opening PNGs."""
    reports = []
    for record in context["cohort"]["pools"]["validation"]:
        rows = [row for row in data["rows"] if row["case"] == record["patient_id"]]
        matrix = np.zeros((3, 3), dtype=np.int64)
        for row in rows:
            matrix[:, 0] += [256 * 256 - row["kidney_px"] - row["tumor_px"],
                             row["kidney_px"], row["tumor_px"]]
        item = patient_report(record["patient_id"], matrix)
        item["exported_slice_count"] = len(rows)
        reports.append(item)
    report = summarize_patients(reports)
    report.update(loss=0.5, split="validation", slice_count=sum(p["exported_slice_count"] for p in reports),
                  metric_scope=training.METRIC_SCOPE, tumour_ce_weight=1.0,
                  loss_configuration=training.png_loss_configuration(),
                  loss_scope="declared synthetic-fixture validation objective; no training was run",
                  model_selection_only=True, clinical_performance_evidence=False,
                  full_scan_coverage=False, created_at_utc=STAMP)
    return report


@pytest.fixture
def completed_run(fixture_data, monkeypatch):
    # Enforce the fixture's promise even if its setup is changed later.
    monkeypatch.setattr(training, "train_png", lambda *args, **kwargs: pytest.fail("Fixture must not train"))
    data = fixture_data
    context = training.load_training_context(data["root"], data["cohort_path"], data["audit_path"])
    pools = context["cohort"]["pools"]
    counts = {role: len(records) for role, records in pools.items()}
    slices = {role: sum(record["selected_slice_count"] for record in records) for role, records in pools.items()}
    model, proof = tiny_builder()
    # A deterministic complete tiny checkpoint makes actual restoration visible.
    with torch.no_grad():
        model.network.segmentation_head.weight.fill_(0.125)
        model.network.segmentation_head.bias.copy_(torch.tensor([0.2, 0.1, -0.1]))
    model.eval()
    state = training._state(model)
    encoder_hash = training._state_hash(training._state(model.network.encoder))
    report = declared_validation_report(data, context)
    optimizer_steps = (slices["train"] + 1) // 2
    selection = {**training._selection_rules("validation_loss"), "best_score": report["loss"],
                 "selected_validation_loss": report["loss"], "lowest_validation_loss_observed": report["loss"]}
    history = [{"epoch": 1, "epochs": 1, "train_loss": 0.75, "tumour_ce_weight": 1.0,
                "validation_loss": report["loss"], "train_slices_seen": slices["train"],
                "validation_slices_seen": slices["validation"],
                "validation_patient_mean": copy.deepcopy(report["patient_mean"]),
                "validation_contributing_patient_counts": copy.deepcopy(report["contributing_patient_counts"]),
                "selection_metric": "validation_loss", "selection_score": report["loss"],
                "checkpoint_selected": True, "optimizer_steps_completed": optimizer_steps,
                "started_at_utc": STAMP, "completed_at_utc": STAMP, "metric_scope": training.METRIC_SCOPE}]
    metadata = {
        "schema_version": training.SCHEMA, "architecture": model.architecture,
        "input_channels": 1, "input_mode": "single_2d_slice", "classes": CLASSES,
        "data_mode": "synthetic_fixture", "tumour_ce_weight": 1.0,
        "loss_configuration": training.png_loss_configuration(),
        "preprocessing": context["config"].to_dict(),
        "preprocessing_sha256": context["cohort"]["preprocessing"]["config_sha256"],
        "audit": copy.deepcopy(context["audit"]), "cohort": copy.deepcopy(context["cohort"]),
        "source_root": str(context["root"]),
        "bound_file_paths": {key: str(path) for key, path in context["paths"].items()},
        "bound_file_sha256": context["file_sha256"].copy(), "pretrained_proof": proof,
        "encoder_frozen": True, "encoder_state_unchanged": True,
        "encoder_initial_state_sha256": encoder_hash, "encoder_saved_state_sha256": encoder_hash,
        "finite_nonzero_decoder_head_gradients": True, "decoder_updated": True,
        "segmentation_head_updated": True, "gradient_checks_completed": optimizer_steps,
        "state_fingerprint_sha256": training._state_hash(state),
        "epochs": 1, "best_epoch": 1, "best_validation_loss": report["loss"],
        "lowest_validation_loss_observed": report["loss"], "checkpoint_selection": selection,
        "batch_size": 2, "learning_rate": 1e-3, "seed": 42, "device": "cpu", "num_threads": 2,
        "optimizer": "Adam", "optimizer_steps_completed": optimizer_steps,
        "sampling": "all exported training slices once per epoch; shuffled without replacement; no label-based selection/oversampling",
        "drop_last": False, "train_slice_count": slices["train"], "validation_slice_count": slices["validation"],
        "history": history, "best_validation_report": report, "metric_scope": training.METRIC_SCOPE,
        "test_evaluated": False, "prediction_run": False, "test_prediction_pixels_loaded": False,
        "clinical_performance_evidence": False, "anatomical_alignment_verified": False,
        "manual_review_status": "pending", "visual_review_note": "Synthetic engineering fixture only.",
        "full_scan_coverage": False, "source_patient_aliases_verified": False,
        "library_versions": {}, "started_at_utc": STAMP, "completed_at_utc": STAMP,
        "limitations": ["Declared synthetic fixture evidence; no actual training or clinical evaluation."],
    }
    metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
    training._validate_metadata(metadata)
    run_dir = data["tmp"] / "completed_fixture_run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "best_model.pt"
    torch.save({"schema_version": training.SCHEMA, "metadata": metadata, "state_dict": state}, checkpoint_path)
    configuration = {
        "epochs": 1, "batch_size": 2, "learning_rate": 1e-3, "seed": 42, "device": "cpu", "cpu_threads": 2,
        "selection_metric": "validation_loss", "tumour_ce_weight": 1.0,
        "training_sampling": "all exported training rows, shuffled once per epoch; no label-based sampling", "resume": False,
    }
    write_json(run_dir / "training_history.json", {"schema_version": "png_training_history_v1", "scope": SCOPE,
               "configuration": configuration, "history": history})
    write_json(run_dir / "validation_report.json", report)
    hashes = {name: digest((run_dir / name).read_bytes()) for name in ARTIFACTS}
    summary = {
        "schema_version": "png_training_run_v1", "scope": SCOPE, "status": "completed",
        "formal_training_completed": True, "patient_counts": counts, "selected_slice_counts": slices,
        "configuration": configuration, "epochs_completed": 1, "best_epoch": 1,
        "checkpoint": {"path": str(checkpoint_path), "sha256": hashes["best_model.pt"], "schema_version": training.SCHEMA},
        "artifact_sha256": hashes, "checkpoint_selection": selection,
        "loss_configuration": training.png_loss_configuration(), "validation_patient_mean": report["patient_mean"],
        "pretrained_encoder_proof": proof,
        "cohort_fingerprint_sha256": context["cohort"]["cohort_fingerprint_sha256"],
        "audit_fingerprint_sha256": context["audit"]["audit_fingerprint_sha256"],
        "reference_review": {"report_available": False, "manual_review_status": "not_recorded"},
        "visual_review_note": metadata["visual_review_note"], "clinical_alignment_verified": False,
        "full_scan_coverage_verified": False, "final_test_performed": False,
        "independent_prediction_performed": False, "started_at_utc": STAMP, "finished_at_utc": STAMP,
        "last_reported_progress": None,
    }
    write_json(run_dir / "run_summary.json", summary)
    return {**data, "run_dir": run_dir, "freeze_path": data["tmp"] / "frozen_model.json",
            "checkpoint_path": checkpoint_path, "metadata": metadata, "summary": summary,
            "state": state, "slices": slices, "counts": counts}


def test_inspect_freeze_and_check_only_never_decode_or_initialize(completed_run, monkeypatch):
    prohibit_model_and_pixels(monkeypatch)
    loaded, record = save_freeze(completed_run)
    final_test.validate_freeze_record(record, loaded)
    checked = frozen_inspected(completed_run)
    assert set(loaded) >= {"context", "config", "metadata", "summary", "checkpoint_path", "checkpoint_sha256",
                          "run_dir", "run_summary_path", "run_summary_sha256", "training_artifact_sha256",
                          "artifact_paths", "artifact_sha256", "allow_fixture"}
    assert checked["checkpoint_sha256"] == digest(completed_run["checkpoint_path"].read_bytes())
    assert record["schema_version"] == "frozen_png_model_v1"
    assert record["patient_counts"] == completed_run["counts"]
    assert record["selected_slice_counts"] == completed_run["slices"]
    assert record["patient_ids"]["test"] == [r["patient_id"] for r in completed_run["cohort"]["pools"]["test"]]
    assert record["freeze_fingerprint_sha256"] == canonical_fingerprint(record, "freeze_fingerprint_sha256")


def test_synthetic_run_requires_explicit_fixture_gate(completed_run, monkeypatch):
    save_freeze(completed_run)
    prohibit_model_and_pixels(monkeypatch)
    with pytest.raises(ValueError, match="[Ff]ixture|synthetic"):
        inspected(completed_run, allow_fixture=False)
    with pytest.raises(ValueError, match="[Ff]ixture|synthetic"):
        frozen_inspected(completed_run, allow_fixture=False)
    loaded = frozen_inspected(completed_run)
    with pytest.raises(ValueError, match="builder|fixture"):
        final_test.evaluate_frozen_test(loaded)


@pytest.mark.parametrize("bound", ["manifest", "original_split", "cohort", "audit"])
def test_bound_input_drift_fails_before_pixels_and_model(completed_run, monkeypatch, bound):
    save_freeze(completed_run)
    loaded = frozen_inspected(completed_run)
    path = loaded["context"]["paths"][bound]
    path.write_bytes(path.read_bytes() + b" ")
    prohibit_model_and_pixels(monkeypatch)
    with pytest.raises(ValueError):
        frozen_inspected(completed_run)
    with pytest.raises(ValueError):
        final_test.evaluate_frozen_test(loaded, model_builder=tiny_builder)


@pytest.mark.parametrize("name", ["run_summary.json", *ARTIFACTS])
def test_saved_run_artifact_byte_drift_rejected_before_pixels(completed_run, monkeypatch, name):
    save_freeze(completed_run)
    loaded = frozen_inspected(completed_run)
    path = completed_run["run_dir"] / name
    path.write_bytes(path.read_bytes() + b" ")
    prohibit_model_and_pixels(monkeypatch)
    with pytest.raises(ValueError):
        frozen_inspected(completed_run)
    with pytest.raises(ValueError):
        final_test.evaluate_frozen_test(loaded, model_builder=tiny_builder)


@pytest.mark.parametrize("name", ["run_summary.json", *ARTIFACTS])
def test_rehashed_artifacts_and_freeze_still_reject_semantic_run_drift(completed_run, monkeypatch, name):
    _, record = save_freeze(completed_run)
    run_dir = completed_run["run_dir"]
    path = run_dir / name
    if name == "best_model.pt":
        payload = torch.load(path, weights_only=True)
        payload["metadata"]["best_epoch"] = 2
        payload["metadata"]["metadata_fingerprint_sha256"] = canonical_fingerprint(payload["metadata"], "metadata_fingerprint_sha256")
        torch.save(payload, path)
        record["selected_epoch"] = 2
        record["metadata_fingerprint_sha256"] = payload["metadata"]["metadata_fingerprint_sha256"]
    else:
        artifact = read_json(path)
        if name == "run_summary.json":
            artifact["best_epoch"] = 2
            record["selected_epoch"] = 2
        elif name == "training_history.json":
            artifact["history"][0]["train_slices_seen"] -= 1
        else:
            artifact["slice_count"] -= 1
        write_json(path, artifact)
    # All byte hashes in the summary and persisted freeze now agree with the
    # altered files.  Semantic agreement with checkpoint history/pools is still
    # required; self-referential rehashing must not authorize changed evidence.
    hashes = {artifact: digest((run_dir / artifact).read_bytes()) for artifact in ARTIFACTS}
    summary = read_json(run_dir / "run_summary.json")
    summary["artifact_sha256"] = hashes
    summary["checkpoint"]["sha256"] = hashes["best_model.pt"]
    if name == "best_model.pt":
        summary["best_epoch"] = 2
    write_json(run_dir / "run_summary.json", summary)
    record["training_artifact_sha256"] = hashes
    record["checkpoint_sha256"] = hashes["best_model.pt"]
    record["run_summary_sha256"] = digest((run_dir / "run_summary.json").read_bytes())
    record["freeze_fingerprint_sha256"] = canonical_fingerprint(record, "freeze_fingerprint_sha256")
    write_json(completed_run["freeze_path"], record)
    prohibit_model_and_pixels(monkeypatch)
    with pytest.raises(ValueError):
        inspected(completed_run)
    with pytest.raises(ValueError):
        frozen_inspected(completed_run)


@pytest.mark.parametrize("field", ["selected_epoch", "checkpoint_selection", "tumour_ce_weight", "patient_ids",
                                   "pools", "preprocessing", "state_fingerprint_sha256"])
def test_rehashed_freeze_cannot_change_selected_model_or_cohort(completed_run, monkeypatch, field):
    loaded, record = save_freeze(completed_run)
    if field == "selected_epoch":
        record[field] += 1
    elif field == "checkpoint_selection":
        record[field]["best_score"] += 0.1
    elif field == "tumour_ce_weight":
        record[field] = 2.0
    elif field == "patient_ids":
        record[field]["test"][0] = record[field]["prediction"][0]
    elif field == "pools":
        record[field]["test"][0]["selected_slice_count"] += 1
    elif field == "preprocessing":
        record[field]["mean"] = 0.5
    else:
        record[field] = "0" * 64
    record["freeze_fingerprint_sha256"] = canonical_fingerprint(record, "freeze_fingerprint_sha256")
    write_json(completed_run["freeze_path"], record)
    prohibit_model_and_pixels(monkeypatch)
    with pytest.raises(ValueError):
        final_test.validate_freeze_record(record, loaded)
    with pytest.raises(ValueError):
        frozen_inspected(completed_run)


def test_training_dataset_still_refuses_test_and_prediction(completed_run, monkeypatch):
    loaded = inspected(completed_run)
    prohibit_model_and_pixels(monkeypatch)
    for role in ("test", "prediction"):
        with pytest.raises(ValueError, match="never constructs"):
            training.AuditedPNGDataset(loaded["context"], role)


def test_fixed_test_rows_once_actual_restore_macro_metrics_and_unchanged_weights(completed_run, monkeypatch):
    save_freeze(completed_run)
    loaded = frozen_inspected(completed_run)
    expected_patients = [r["patient_id"] for r in completed_run["cohort"]["pools"]["test"]]
    expected_rows = [r for r in completed_run["rows"] if r["case"] in expected_patients]
    expected_samples = {"%s:z%d" % (r["case"], r["z"]): 1 for r in expected_rows}
    decodes, forwards, restores, built = Counter(), [], [], []
    original_decode = png_dataset._read_png

    def guarded_decode(path, size):
        path = Path(path)
        patient = path.stem.split("_z")[0]
        assert patient in expected_patients, "A non-test patient was decoded"
        decodes[(patient, path.parent.name, path.name)] += 1
        return original_decode(path, size)

    def restoring_builder(**kwargs):
        model, proof = tiny_builder(**kwargs)
        built.append(model)
        actual_load = model.load_state_dict
        actual_forward = model.forward

        def load_state(state, *args, **options):
            restores.append(copy.deepcopy(state))
            return actual_load(state, *args, **options)

        def forward(inputs):
            assert not model.training and not torch.is_grad_enabled()
            assert inputs.shape[1:] == (1, 256, 256)
            assert all(torch.equal(model.state_dict()[key], tensor) for key, tensor in completed_run["state"].items())
            forwards.append(len(inputs))
            return actual_forward(inputs)

        model.load_state_dict = load_state
        model.forward = forward
        return model, proof

    monkeypatch.setattr(png_dataset, "_read_png", guarded_decode)
    monkeypatch.setattr(torch.optim, "Adam", lambda *args, **kwargs: pytest.fail("Final test constructed an optimizer"))
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", lambda *args, **kwargs: pytest.fail("Final test constructed an optimizer"))
    monkeypatch.setattr(torch.optim.Optimizer, "step", lambda *args, **kwargs: pytest.fail("Final test stepped an optimizer"))
    report = final_test.evaluate_frozen_test(loaded, batch_size=3, model_builder=restoring_builder)
    assert len(built) == 1 and len(restores) == 1
    assert all(torch.equal(restores[0][key], tensor) for key, tensor in completed_run["state"].items())
    assert sum(forwards) == len(expected_rows)
    assert decodes == Counter({(r["case"], folder, Path(r[kind]).name): 1 for r in expected_rows
                              for kind, folder in (("image", "images"), ("mask", "masks"))})
    assert report["schema_version"] == "png_frozen_test_v1"
    assert report["patient_count"] == 10 and report["unit"] == "patient"
    assert [p["patient_id"] for p in report["patients"]] == expected_patients
    assert report["slice_count"] == len(expected_rows) == completed_run["slices"]["test"]
    assert report["decoded_pairs"] == {"train": 0, "validation": 0, "test": len(expected_rows), "prediction": 0}
    assert report["sample_counts"] == expected_samples
    assert report["confusion_matrix_total_pixels"] == len(expected_rows) * 256 * 256
    assert sum(sum(sum(row) for row in p["confusion_matrix"]) for p in report["patients"]) == report["confusion_matrix_total_pixels"]
    assert report["model_state_before_sha256"] == report["model_state_after_sha256"] == training._state_hash(completed_run["state"])
    assert report["model_state_unchanged"] is True
    assert report["frozen_model_sha256"] == digest(completed_run["checkpoint_path"].read_bytes())
    assert report["training_performed"] is False and report["model_selection_performed"] is False
    assert report["optimizer_steps"] == 0
    assert all(torch.equal(built[0].state_dict()[key], tensor) for key, tensor in completed_run["state"].items())
    for name in CLASSES.values():
        for metric in METRIC_NAMES:
            values = [p["per_class"][name][metric] for p in report["patients"] if p["per_class"][name][metric] is not None]
            assert report["contributing_patient_counts"][name][metric] == len(values)
            assert report["patient_mean"][name][metric] == (pytest.approx(float(np.mean(values))) if values else None)
    absent = [p for p in report["patients"] if p["per_class"]["tumour"]["support_pixels"] == 0]
    assert absent and all(p["per_class"]["tumour"]["dice"] is None for p in absent)


def test_test_png_byte_drift_rejected_before_decoder(completed_run, monkeypatch):
    save_freeze(completed_run)
    loaded = frozen_inspected(completed_run)
    patient = completed_run["cohort"]["pools"]["test"][0]["patient_id"]
    row = next(r for r in completed_run["rows"] if r["case"] == patient)
    source = completed_run["data"] / row["image"]
    source.write_bytes(source.read_bytes() + b"changed")
    monkeypatch.setattr(png_dataset, "_read_png", lambda *args: pytest.fail("Changed PNG bytes were decoded"))
    with pytest.raises(ValueError, match="bytes|hash|changed"):
        final_test.evaluate_frozen_test(loaded, model_builder=tiny_builder)


@pytest.mark.parametrize("fact", ["decoded_pixel_sha256", "label_counts", "patient_id"])
def test_test_asset_pixel_and_identity_facts_are_enforced(completed_run, monkeypatch, fact):
    save_freeze(completed_run)
    loaded = frozen_inspected(completed_run)
    patient = completed_run["cohort"]["pools"]["test"][0]["patient_id"]
    row = next(r for r in completed_run["rows"] if r["case"] == patient)
    asset = loaded["context"]["audit"]["assets"][row["mask"]]
    if fact == "decoded_pixel_sha256":
        asset[fact] = "0" * 64
    elif fact == "label_counts":
        asset[fact]["0"] -= 1
    else:
        asset[fact] = completed_run["cohort"]["pools"]["prediction"][0]["patient_id"]
    with pytest.raises(ValueError):
        final_test.evaluate_frozen_test(loaded, model_builder=tiny_builder)


def test_inference_model_mutation_is_rejected(completed_run):
    save_freeze(completed_run)
    loaded = frozen_inspected(completed_run)

    def mutating_builder(**kwargs):
        model, proof = tiny_builder(**kwargs)
        actual_forward = model.forward

        def forward(inputs):
            output = actual_forward(inputs)
            model.network.segmentation_head.bias.add_(0.01)
            return output

        model.forward = forward
        return model, proof

    with pytest.raises(ValueError, match="state|changed|mutat"):
        final_test.evaluate_frozen_test(loaded, model_builder=mutating_builder)
