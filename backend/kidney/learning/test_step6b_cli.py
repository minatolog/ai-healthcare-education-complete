"""Step 6B orchestration tests with stub training, no model or patient PNGs."""
from argparse import Namespace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "step6b_cli_under_test", WORKSPACE / "learning" / "06b_train_png.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def snapshot(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "original_data"
    data = root / "data2d"
    data.mkdir(parents=True)
    args = Namespace(root=root, cohort=tmp_path / "cohort.json", audit=tmp_path / "audit.json",
                     reference_review=tmp_path / "absent_reference_review.json",
                     output_dir=tmp_path / "runs" / "run01", epochs=1, batch_size=2,
                     learning_rate=0.001, seed=42, device="cpu", cpu_threads=2,
                     visual_review_note=None, check_only=False)
    pools, offset = {}, 0
    for role, count in (("train", 20), ("validation", 10), ("test", 10), ("prediction", 5)):
        pools[role] = [{"patient_id": "case_%05d" % index, "selected_slice_count": 2}
                       for index in range(offset, offset + count)]
        offset += count
    context = {
        "root": root.resolve(), "data_root": data.resolve(),
        "paths": {"cohort": args.cohort, "audit": args.audit,
                  "manifest": data / "manifest.csv", "original_split": data / "splits.json"},
        "file_sha256": {key: digest(("fake " + key).encode())
                        for key in ("cohort", "audit", "manifest", "original_split")},
        "cohort": {"pools": pools, "cohort_fingerprint_sha256": digest(b"fake cohort")},
        "audit": {"audit_fingerprint_sha256": digest(b"fake audit")},
        "config": SimpleNamespace(input_channels=1, size=256),
    }
    event = {"event": "batch", "phase": "training", "epoch": 1,
             "batch_index": 1, "total_batches": 20, "slices_seen": 2, "loss": 1.25}
    model = object()
    fitted = {"model": model, "history": [{"epoch": 1, "train_loss": 1.25}],
              "metadata": {"best_epoch": 1, "fixture": True},
              "proof": {"offline_only": True, "fixture": "not actual weight verification"},
              "best_validation_report": {"patient_count": 10, "split": "validation",
                                         "patient_mean": {"kidney": {"dice": 0.4},
                                                          "tumour": {"dice": 0.2}}}}
    state = {"context_calls": [], "train_calls": [], "save_calls": [], "cache_calls": [],
             "runtime_load_calls": [], "event": event, "fitted": fitted,
             "train_error": None, "checkpoint_error": None, "context": context}

    def load_context(actual_root, cohort_path, audit_path):
        assert actual_root == root.resolve()
        assert (cohort_path, audit_path) == (args.cohort, args.audit)
        state["context_calls"].append((actual_root, cohort_path, audit_path))
        return context

    def train_stub(actual_root, cohort_path, audit_path, **kwargs):
        state["train_calls"].append(kwargs)
        assert (actual_root, cohort_path, audit_path) == (root.resolve(), args.cohort, args.audit)
        initial = read_json(args.output_dir / "run_summary.json")
        assert initial["status"] == "running" and initial["formal_training_completed"] is False
        assert initial["last_reported_progress"] is None
        kwargs["progress"](copy.deepcopy(event))
        # The callback must persist a complete event before training can fail.
        lines = (args.output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["event"] == event["event"]
        running = read_json(args.output_dir / "run_summary.json")
        assert running["status"] == "running"
        assert running["last_reported_progress"] == json.loads(lines[0])
        state["persisted_running_summary"] = running
        if state["train_error"] is not None:
            raise state["train_error"]
        return fitted

    def save_stub(path, actual_model, metadata):
        assert actual_model is model and metadata is fitted["metadata"]
        state["save_calls"].append(path)
        if state["checkpoint_error"] is not None:
            raise state["checkpoint_error"]
        content = b"Synthetic checkpoint bytes; no Torch serialization.\n"
        path.write_bytes(content)
        return {"checkpoint_sha256": digest(content), "schema_version": "fixture_only"}

    runtime = SimpleNamespace(load_training_context=load_context, train_png=train_stub,
                              save_png_checkpoint=save_stub)

    def runtime_stub():
        state["runtime_load_calls"].append(True)
        return runtime

    def cache_stub():
        state["cache_calls"].append(True)
        return {"download_attempted": False, "model_initialized": False,
                "source_revision": "synthetic fixture; no cache files were read"}

    monkeypatch.setattr(CLI, "load_runtime", runtime_stub)
    monkeypatch.setattr(CLI, "verify_offline_cache", cache_stub)
    return args, state


def valid_reference_report(state):
    return {"schema_version": "training_reference_review_v1",
            "cohort_fingerprint_sha256": state["context"]["cohort"]["cohort_fingerprint_sha256"],
            "audit_fingerprint_sha256": state["context"]["audit"]["audit_fingerprint_sha256"],
            "contains_model_prediction": False, "manual_review_status": "pending",
            "anatomical_alignment_verified": False}


def test_check_only_validates_context_without_training_or_creating_run_directory(environment):
    args, state = environment
    args.check_only = True
    report = CLI.run(args)
    assert report["status"] == "preflight_completed"
    assert report["training_performed"] is False and report["model_initialized"] is False
    assert report["ct_mask_pngs_decoded"] == 0 and report["optimizer_steps"] == 0
    assert report["patient_counts"] == {"train": 20, "validation": 10, "test": 10, "prediction": 5}
    assert report["reference_review"] == {"report_available": False, "manual_review_status": "not_recorded"}
    assert not args.output_dir.exists() and not args.output_dir.parent.exists()
    assert not state["train_calls"] and not state["save_calls"]
    assert len(state["context_calls"]) == 1 and len(state["cache_calls"]) == 1
    assert snapshot(args.root) == {}


def test_completed_run_records_progress_checkpoint_and_artifact_hashes(environment):
    args, state = environment
    report = CLI.run(args)
    assert report["status"] == "completed" and report["formal_training_completed"] is True
    assert report["epochs_completed"] == 1 and report["best_epoch"] == 1
    assert report["final_test_performed"] is False and report["independent_prediction_performed"] is False
    assert report["clinical_alignment_verified"] is False
    assert report["pretrained_encoder_proof"] == state["fitted"]["proof"]
    assert report["validation_patient_mean"] == state["fitted"]["best_validation_report"]["patient_mean"]
    assert read_json(args.output_dir / "run_summary.json") == report
    assert read_json(args.output_dir / "training_history.json")["history"] == state["fitted"]["history"]
    assert read_json(args.output_dir / "validation_report.json") == state["fitted"]["best_validation_report"]
    for name in ("best_model.pt", "training_history.json", "validation_report.json"):
        assert report["artifact_sha256"][name] == digest((args.output_dir / name).read_bytes())
    assert state["persisted_running_summary"]["formal_training_completed"] is False
    assert len(state["train_calls"]) == 1 and len(state["save_calls"]) == 1
    call = state["train_calls"][0]
    assert (call["epochs"], call["batch_size"], call["lr"], call["seed"], call["device"],
            call["num_threads"], call["visual_review_note"]) == (1, 2, 0.001, 42, "cpu", 2, None)
    assert snapshot(args.root) == {}


def test_existing_run01_is_never_overwritten_or_trained_again(environment):
    args, state = environment
    CLI.run(args)
    before = snapshot(args.output_dir)
    with pytest.raises(ValueError, match="already exists and will not be replaced"):
        CLI.run(args)
    assert snapshot(args.output_dir) == before
    assert len(state["train_calls"]) == 1 and len(state["save_calls"]) == 1


@pytest.mark.parametrize("error, status", [(RuntimeError("Synthetic training failure"), "failed"),
                                         (KeyboardInterrupt(), "interrupted")])
def test_failure_or_interruption_preserves_running_progress_without_completion(environment, error, status):
    args, state = environment
    state["train_error"] = error
    with pytest.raises(type(error)):
        CLI.run(args)
    report = read_json(args.output_dir / "run_summary.json")
    assert report["status"] == status and report["formal_training_completed"] is False
    assert report["last_reported_progress"] == state["persisted_running_summary"]["last_reported_progress"]
    line = json.loads((args.output_dir / "progress.jsonl").read_text(encoding="utf-8"))
    assert line == report["last_reported_progress"] and line["slices_seen"] == 2
    assert "artifact_sha256" not in report and "epochs_completed" not in report
    assert not (args.output_dir / "best_model.pt").exists()
    assert not (args.output_dir / "training_history.json").exists()
    assert not state["save_calls"]
    if status == "failed":
        assert report["error_type"] == "RuntimeError" and "Synthetic" in report["error_message"]
    else:
        assert "does not resume" in report["interruption_note"]


def test_checkpoint_save_failure_does_not_claim_formal_training_completed(environment):
    args, state = environment
    state["checkpoint_error"] = RuntimeError("Synthetic checkpoint failure")
    with pytest.raises(RuntimeError, match="checkpoint failure"):
        CLI.run(args)
    report = read_json(args.output_dir / "run_summary.json")
    assert report["status"] == "failed" and report["formal_training_completed"] is False
    assert report["error_message"] == "Synthetic checkpoint failure"
    assert (args.output_dir / "progress.jsonl").is_file()
    assert "artifact_sha256" not in report


@pytest.mark.parametrize("field, value", [
    ("epochs", 0), ("batch_size", 0), ("learning_rate", 0), ("learning_rate", -0.1),
    ("learning_rate", float("nan")), ("learning_rate", float("inf")),
    ("seed", -1), ("seed", 2**63), ("cpu_threads", 0), ("cpu_threads", 17), ("device", "cuda"),
])
def test_invalid_settings_rejected_before_runtime_or_run_directory(environment, field, value):
    args, state = environment
    setattr(args, field, value)
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not state["runtime_load_calls"] and not state["context_calls"] and not state["cache_calls"]
    assert not state["train_calls"] and not args.output_dir.exists()


def test_argument_parser_rejects_unsupported_batch_size_before_runtime(environment):
    _, state = environment
    with pytest.raises(SystemExit) as error:
        CLI.main(["--batch-size", "3"])
    assert error.value.code == 2 and not state["runtime_load_calls"]


@pytest.mark.parametrize("changed", ["cohort_fingerprint_sha256", "audit_fingerprint_sha256",
                                    "schema_version", "contains_model_prediction", "non_object"])
def test_reference_review_binding_mismatch_rejected_before_reserving_run(environment, changed):
    args, state = environment
    report = valid_reference_report(state)
    if changed == "non_object":
        report = None
    elif changed == "contains_model_prediction":
        report[changed] = True
    else:
        report[changed] = "Different saved evidence."
    args.reference_review.write_text(json.dumps(report), encoding="utf-8")
    before = args.reference_review.read_bytes()
    with pytest.raises(ValueError, match="reference-review report differs"):
        CLI.run(args)
    assert not args.output_dir.exists() and not state["train_calls"]
    assert args.reference_review.read_bytes() == before


def test_valid_pending_reference_report_and_note_are_recorded_without_clinical_claim(environment):
    args, state = environment
    args.check_only = True
    args.visual_review_note = "Synthetic note: only the training reference previews were inspected."
    args.reference_review.write_text(json.dumps(valid_reference_report(state)), encoding="utf-8")
    result = CLI.run(args)
    record = result["reference_review"]
    assert record["report_available"] is True
    assert record["report_sha256"] == digest(args.reference_review.read_bytes())
    assert record["manual_review_status"] == "pending" and record["anatomical_alignment_verified"] is False
    assert result["visual_review_note"] == args.visual_review_note
    assert result["clinical_alignment_verified"] is False
    assert not args.output_dir.exists() and not state["train_calls"]


def test_default_selection_retains_loss_policy_in_run_evidence_and_training(environment):
    args, state = environment
    result = CLI.run(args)
    assert result['configuration']['selection_metric'] == 'validation_loss'
    assert state['train_calls'][0]['selection_metric'] == 'validation_loss'


def test_tumour_selection_is_passed_to_training_and_saved_in_run_summary(environment):
    args, state = environment
    args.selection_metric = 'tumour_dice'
    state['fitted']['metadata']['checkpoint_selection'] = {
        'metric': 'tumour_dice', 'best_score': 0.2, 'direction': 'maximize'}
    result = CLI.run(args)
    assert result['configuration']['selection_metric'] == 'tumour_dice'
    assert state['train_calls'][0]['selection_metric'] == 'tumour_dice'
    assert result['checkpoint_selection'] == state['fitted']['metadata']['checkpoint_selection']
    assert read_json(args.output_dir / 'run_summary.json') == result


def test_check_only_records_tumour_selection_without_running_training(environment):
    args, state = environment
    args.selection_metric, args.check_only = 'tumour_dice', True
    result = CLI.run(args)
    assert result['configuration']['selection_metric'] == 'tumour_dice'
    assert not state['train_calls'] and not args.output_dir.exists()


def test_invalid_selection_rejected_before_reading_runtime_or_reserving_output(environment):
    args, state = environment
    args.selection_metric = 'test_dice'
    with pytest.raises(ValueError, match='checkpoint selection'):
        CLI.run(args)
    assert not state['runtime_load_calls'] and not args.output_dir.exists()


def test_parser_rejects_unknown_selection_before_loading_runtime(environment):
    _, state = environment
    with pytest.raises(SystemExit) as error:
        CLI.main(['--selection-metric', 'test_dice'])
    assert error.value.code == 2 and not state['runtime_load_calls']


def test_default_ce_weight_passed_without_changing_sampling_or_selection(environment):
    args, state = environment
    result = CLI.run(args)
    assert result['configuration']['tumour_ce_weight'] == 1.0
    assert state['train_calls'][0]['tumour_ce_weight'] == 1.0
    assert state['train_calls'][0]['selection_metric'] == 'validation_loss'
    assert result['configuration']['training_sampling'] == 'all exported training rows, shuffled once per epoch; no label-based sampling'


def test_weighted_ce_configuration_and_metadata_are_persisted(environment, capsys):
    args, state = environment
    args.tumour_ce_weight, args.selection_metric = 2.0, 'tumour_dice'
    state['fitted']['metadata']['loss_configuration'] = {'fixture': 'loss metadata passed through; not a real training result'}
    result = CLI.run(args)
    assert result['configuration']['tumour_ce_weight'] == 2.0
    assert state['train_calls'][0]['tumour_ce_weight'] == 2.0
    assert result['loss_configuration'] == state['fitted']['metadata']['loss_configuration']
    assert read_json(args.output_dir / 'run_summary.json')['configuration']['tumour_ce_weight'] == 2.0
    assert 'Relative CE class weights [background, kidney, tumour]: [1.0, 1.0, 2.0]' in capsys.readouterr().out
    assert snapshot(args.root) == {}


def test_weighted_ce_check_only_creates_no_run_or_model(environment):
    args, state = environment
    args.tumour_ce_weight, args.check_only = 2.0, True
    result = CLI.run(args)
    assert result['configuration']['tumour_ce_weight'] == 2.0
    assert result['ct_mask_pngs_decoded'] == 0 and result['optimizer_steps'] == 0
    assert not state['train_calls'] and not state['save_calls'] and not args.output_dir.exists()


@pytest.mark.parametrize('weight', [0, -1, None, True, False, '2', float('nan'), float('inf')])
def test_invalid_ce_weight_fails_before_loading_runtime_or_writing_outputs(environment, weight):
    args, state = environment
    args.tumour_ce_weight = weight
    with pytest.raises(ValueError, match='CE weight'):
        CLI.run(args)
    assert not state['runtime_load_calls'] and not state['context_calls'] and not args.output_dir.exists()


def test_parser_rejects_text_ce_weight_before_loading_runtime(environment):
    _, state = environment
    with pytest.raises(SystemExit) as error:
        CLI.main(['--tumour-ce-weight', 'invalid'])
    assert error.value.code == 2 and not state['runtime_load_calls']
