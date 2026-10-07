"""Step 8 orchestration tests using synthetic receipts and a CT-inference stub."""
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
    "step8_cli_under_test", WORKSPACE / "learning" / "08_predict_new_patients.py"
)
PREDICT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREDICT)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def argv_for(args):
    values = []
    for field, value in vars(args).items():
        flag = "--" + field.replace("_", "-")
        if isinstance(value, bool):
            if value:
                values.append(flag)
        else:
            values.extend((flag, str(value)))
    return values


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root, run_dir = tmp_path / "original_data", tmp_path / "source_run"
    root.mkdir()
    run_dir.mkdir()
    (root / "original_ct_placeholder.txt").write_bytes(b"Synthetic CT placeholder; never decoded.\n")
    checkpoint = run_dir / "best_model.pt"
    checkpoint.write_bytes(b"Synthetic checkpoint placeholder; never loaded as weights.\n")
    cohort, audit = tmp_path / "cohort.json", tmp_path / "audit.json"
    write_json(cohort, {"fixture_only": "cohort"})
    write_json(audit, {"fixture_only": "audit"})
    freeze_dir, test_dir = tmp_path / "source_freeze", tmp_path / "completed_test"
    freeze_dir.mkdir()
    test_dir.mkdir()
    freeze_path = freeze_dir / "frozen_model.json"
    write_json(freeze_path, {"schema_version": "png_frozen_model_v1", "fixture_only": True})
    checkpoint_hash, freeze_hash = digest(checkpoint.read_bytes()), digest(freeze_path.read_bytes())
    test_report_path = test_dir / "test_report.json"
    test_report = {
        "schema_version": "png_frozen_test_v1", "status": "completed",
        "checkpoint_sha256": checkpoint_hash, "freeze_sha256": freeze_hash,
        "model_state_unchanged": True, "training_performed": False,
        "model_selection_performed": False, "optimizer_steps": 0,
    }
    # Scores are deliberately absent: the receipt proves completion and binding,
    # and does not provide a numerical threshold for allowing prediction.
    write_json(test_report_path, test_report)
    receipt = {
        "schema_version": "png_final_test_run_v1", "status": "completed",
        "final_test_completed": True, "checkpoint_sha256": checkpoint_hash,
        "freeze_sha256": freeze_hash, "test_report_sha256": digest(test_report_path.read_bytes()),
        "output_dir": str(test_dir.resolve()),
    }
    summary_path, access_path = test_dir / "evaluation_summary.json", freeze_dir / "test_access.json"
    write_json(summary_path, receipt)
    write_json(access_path, receipt)
    args = Namespace(root=root, cohort=cohort, audit=audit, run_dir=run_dir, freeze=freeze_path,
                     test_dir=test_dir, output_dir=tmp_path / "new_predictions" / "prediction01",
                     batch_size=2, cpu_threads=2, check_only=False)
    inspected = {
        "context": {"root": root.resolve()}, "run_dir": run_dir.resolve(),
        "freeze_path": str(freeze_path.resolve()), "checkpoint_sha256": checkpoint_hash,
        "freeze_sha256": freeze_hash,
        "prediction_inputs": {"patient_count": 2, "slice_count": 3},
    }
    events = [
        {"event": "started", "phase": "prediction", "patient_count": 2, "slice_count": 3},
        {"event": "batch", "phase": "prediction", "batch_index": 1, "slices_seen": 2},
    ]
    report = {
        "schema_version": "png_frozen_prediction_v1", "status": "completed",
        "patient_count": 2, "slice_count": 3, "reference_mask_pngs_read": 0,
        "training_performed": False, "model_selection_performed": False,
        "aggregate_performance_evaluation": False,
    }
    observed = {
        "runtime_load_calls": [], "inspect_calls": [], "predict_calls": [],
        "inspection_error": None, "prediction_error": None, "after_prediction": None,
        "saved_report": None, "report": report, "events": events,
        "inspected": inspected, "access_path": access_path,
        "receipt_paths": {"summary": summary_path, "report": test_report_path, "access": access_path},
    }

    def inspect_stub(actual_root, actual_cohort, actual_audit, actual_run, actual_freeze):
        observed["inspect_calls"].append((actual_root, actual_cohort, actual_audit, actual_run, actual_freeze))
        assert (actual_root, actual_cohort, actual_audit, actual_run, actual_freeze) == (
            root, cohort, audit, run_dir, freeze_path)
        if observed["inspection_error"] is not None:
            raise observed["inspection_error"]
        return inspected

    def predict_stub(actual_inspected, output, *, batch_size, num_threads, progress):
        observed["predict_calls"].append({"output": output, "batch_size": batch_size,
                                          "num_threads": num_threads})
        assert actual_inspected is inspected
        # The core checks emptiness before its first callback. The caller must
        # not pre-create its summary or progress files inside this directory.
        assert output.is_dir() and list(output.iterdir()) == []
        partial = output / "predictions" / "case_0900_mask.png"
        partial.parent.mkdir()
        partial.write_bytes(b"Synthetic partial prediction; not a real PNG.\n")
        for event in events:
            progress(copy.deepcopy(event))
            saved = json.loads((output / "progress.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            running = read_json(output / "prediction_summary.json")
            assert {key: saved[key] for key in event} == event
            assert running["status"] == "running" and running["last_reported_progress"] == saved
            observed["persisted_progress"] = saved
        if observed["prediction_error"] is not None:
            raise observed["prediction_error"]
        write_json(output / "prediction_report.json",
                   report if observed["saved_report"] is None else observed["saved_report"])
        if observed["after_prediction"] is not None:
            observed["after_prediction"]()
        return copy.deepcopy(report)

    runtime = SimpleNamespace(inspect_prediction_inputs=inspect_stub, predict_frozen_pool=predict_stub)

    def load_runtime_stub():
        observed["runtime_load_calls"].append(True)
        return runtime

    monkeypatch.setattr(PREDICT, "load_runtime", load_runtime_stub)
    return args, observed


def source_snapshot(args):
    return (snapshot(args.root), snapshot(args.run_dir), snapshot(args.freeze.parent),
            snapshot(args.test_dir), args.cohort.read_bytes(), args.audit.read_bytes())


def test_check_only_validates_receipt_without_outputs_or_model_prediction(environment, tmp_path):
    args, observed = environment
    args.check_only = True
    before = snapshot(tmp_path)
    result = PREDICT.run(args)
    assert result["status"] == "preflight_completed" and result["model_initialized"] is False
    assert result["ct_pngs_decoded"] == result["reference_mask_pngs_read"] == 0
    assert result["training_performed"] is False
    assert len(observed["inspect_calls"]) == 1 and not observed["predict_calls"]
    assert not args.output_dir.parent.exists() and snapshot(tmp_path) == before


def test_completion_saves_progress_report_hash_and_preserves_all_input_evidence(environment):
    args, observed = environment
    before = source_snapshot(args)
    receipt_hashes = {name: digest(path.read_bytes()) for name, path in observed["receipt_paths"].items()}
    result = PREDICT.run(args)
    report_path = args.output_dir / "prediction_report.json"
    assert result["schema_version"] == "png_prediction_run_v1" and result["status"] == "completed"
    assert read_json(args.output_dir / "prediction_summary.json") == result
    assert read_json(report_path) == observed["report"]
    assert result["artifact_sha256"] == {"prediction_report.json": digest(report_path.read_bytes())}
    assert result["completed_test_receipt_sha256"] == receipt_hashes
    assert result["checkpoint_sha256"] == observed["inspected"]["checkpoint_sha256"]
    assert result["freeze_sha256"] == observed["inspected"]["freeze_sha256"]
    assert result["original_test_evidence_unchanged"] is True
    assert result["training_performed"] is result["model_selection_performed"] is False
    assert result["aggregate_performance_evaluation"] is False and result["reference_mask_pngs_read"] == 0
    assert result["output_dir"] == str(args.output_dir.resolve())
    assert result["started_at_utc"] and result["finished_at_utc"]
    progress = [json.loads(line) for line in (args.output_dir / "progress.jsonl").read_text().splitlines()]
    assert len(progress) == len(observed["events"])
    assert [{key: saved[key] for key in event} for saved, event in zip(progress, observed["events"])] == observed["events"]
    assert all(event["recorded_at_utc"] for event in progress)
    assert result["last_reported_progress"] == progress[-1]
    assert observed["predict_calls"] == [{"output": args.output_dir.resolve(), "batch_size": 2, "num_threads": 2}]
    assert source_snapshot(args) == before


@pytest.mark.parametrize("error, status", [
    (RuntimeError("Synthetic CT inference failure"), "failed"), (KeyboardInterrupt(), "interrupted"),
])
def test_failure_or_interrupt_preserves_partial_predictions_and_progress(environment, error, status):
    args, observed = environment
    observed["prediction_error"] = error
    before = source_snapshot(args)
    with pytest.raises(type(error)) as raised:
        PREDICT.run(args)
    assert raised.value is error
    summary = read_json(args.output_dir / "prediction_summary.json")
    assert summary["status"] == status and summary["error_type"] == type(error).__name__
    assert summary["error_message"] == str(error) and summary["finished_at_utc"]
    assert summary["last_reported_progress"] == observed["persisted_progress"]
    progress = [json.loads(line) for line in (args.output_dir / "progress.jsonl").read_text().splitlines()]
    assert len(progress) == 2 and progress[-1] == summary["last_reported_progress"]
    assert (args.output_dir / "predictions" / "case_0900_mask.png").read_bytes().startswith(b"Synthetic partial")
    assert not (args.output_dir / "prediction_report.json").exists()
    assert "artifact_sha256" not in summary and "original_test_evidence_unchanged" not in summary
    assert source_snapshot(args) == before


@pytest.mark.parametrize("check_only", [False, True])
def test_existing_output_is_preserved_before_prediction(environment, check_only):
    args, observed = environment
    args.check_only = check_only
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "human_notes.txt").write_bytes(b"Keep this existing evidence.\n")
    before = snapshot(args.output_dir)
    with pytest.raises(ValueError, match="already exists"):
        PREDICT.run(args)
    assert snapshot(args.output_dir) == before and not observed["predict_calls"]


@pytest.mark.parametrize("location", ["data", "run", "freeze", "test"])
def test_output_inside_protected_inputs_or_receipts_is_rejected_before_writes(environment, tmp_path, location):
    args, observed = environment
    protected = {"data": args.root, "run": args.run_dir, "freeze": args.freeze.parent, "test": args.test_dir}
    args.output_dir = protected[location] / "unsafe_output"
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        PREDICT.run(args)
    assert not args.output_dir.exists() and not observed["predict_calls"]
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("symlink_kind", ["output", "parent"])
def test_symlink_output_or_parent_is_rejected_before_writes(environment, tmp_path, symlink_kind):
    args, observed = environment
    target = tmp_path / "symlink_target"
    target.mkdir()
    alias = tmp_path / "output_alias"
    alias.symlink_to(target, target_is_directory=True)
    args.output_dir = alias if symlink_kind == "output" else alias / "new_output"
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        PREDICT.run(args)
    assert snapshot(tmp_path) == before and snapshot(target) == {} and not observed["predict_calls"]


@pytest.mark.parametrize("symlink_kind", ["directory", "parent", "report", "access"])
def test_symlink_receipt_paths_cannot_bypass_completed_test_guard(environment, tmp_path, symlink_kind):
    args, observed = environment
    if symlink_kind in ("directory", "parent"):
        alias = tmp_path / "test_alias"
        target = args.test_dir if symlink_kind == "directory" else args.test_dir.parent
        alias.symlink_to(target, target_is_directory=True)
        args.test_dir = alias if symlink_kind == "directory" else alias / args.test_dir.name
    else:
        path = observed["receipt_paths"][symlink_kind]
        saved = tmp_path / (symlink_kind + "_saved.json")
        saved.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(saved)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="symlink|Complete Step 7B"):
        PREDICT.run(args)
    assert not args.output_dir.exists() and not observed["predict_calls"]
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("invalid_receipt", [
    "missing_report", "incomplete", "wrong_checkpoint", "wrong_freeze", "wrong_report_hash",
    "model_changed", "non_object", "invalid_json",
])
def test_bad_receipt_stops_before_output_or_prediction(environment, tmp_path, invalid_receipt):
    args, observed = environment
    paths = observed["receipt_paths"]
    if invalid_receipt == "missing_report":
        paths["report"].unlink()
    elif invalid_receipt == "invalid_json":
        paths["access"].write_text("{", encoding="utf-8")
    elif invalid_receipt == "non_object":
        write_json(paths["summary"], [])
    elif invalid_receipt == "model_changed":
        report = read_json(paths["report"])
        report["model_state_unchanged"] = False
        write_json(paths["report"], report)
        for name in ("summary", "access"):
            record = read_json(paths[name])
            record["test_report_sha256"] = digest(paths["report"].read_bytes())
            write_json(paths[name], record)
    else:
        record = read_json(paths["access"])
        key = {"incomplete": "status", "wrong_checkpoint": "checkpoint_sha256",
               "wrong_freeze": "freeze_sha256", "wrong_report_hash": "test_report_sha256"}[invalid_receipt]
        record[key] = "running" if invalid_receipt == "incomplete" else "0" * 64
        write_json(paths["access"], record)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError):
        PREDICT.run(args)
    assert not args.output_dir.exists() and not observed["predict_calls"]
    assert snapshot(tmp_path) == before


def test_saved_and_returned_report_mismatch_cannot_mark_completion(environment):
    args, observed = environment
    observed["saved_report"] = {**observed["report"], "slice_count": 999}
    before = source_snapshot(args)
    with pytest.raises(ValueError, match="Saved prediction report differs"):
        PREDICT.run(args)
    summary = read_json(args.output_dir / "prediction_summary.json")
    assert summary["status"] == "failed" and summary["error_type"] == "ValueError"
    assert read_json(args.output_dir / "prediction_report.json") == observed["saved_report"]
    assert summary["last_reported_progress"] == observed["persisted_progress"]
    assert "artifact_sha256" not in summary and "original_test_evidence_unchanged" not in summary
    assert source_snapshot(args) == before


@pytest.mark.parametrize("which", ["summary", "report", "access"])
def test_changed_test_evidence_after_inference_prevents_completion(environment, which):
    args, observed = environment
    path = observed["receipt_paths"][which]
    original = path.read_bytes()
    observed["after_prediction"] = lambda: path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="evidence changed during prediction"):
        PREDICT.run(args)
    summary = read_json(args.output_dir / "prediction_summary.json")
    assert summary["status"] == "failed" and summary["error_type"] == "ValueError"
    assert summary["last_reported_progress"] == observed["persisted_progress"]
    assert "artifact_sha256" not in summary and "original_test_evidence_unchanged" not in summary
    assert read_json(args.output_dir / "prediction_report.json") == observed["report"]
    # The CLI must detect the changed source, not overwrite it with its old copy.
    assert path.read_bytes() == original + b"\n"


@pytest.mark.parametrize("field, value", [("batch_size", 0), ("batch_size", 3),
                                         ("cpu_threads", 0), ("cpu_threads", 17)])
def test_invalid_settings_fail_before_runtime_loading_or_writes(environment, tmp_path, field, value):
    args, observed = environment
    setattr(args, field, value)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="batch size"):
        PREDICT.run(args)
    assert not observed["runtime_load_calls"] and not observed["inspect_calls"]
    assert not observed["predict_calls"] and snapshot(tmp_path) == before


def test_input_binding_error_stops_before_writes_or_prediction(environment, tmp_path):
    args, observed = environment
    observed["inspection_error"] = ValueError("Synthetic frozen-input binding mismatch")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="binding mismatch"):
        PREDICT.run(args)
    assert not observed["predict_calls"] and snapshot(tmp_path) == before


def test_failure_summary_write_does_not_replace_original_inference_error(environment, monkeypatch, capsys):
    args, observed = environment
    original_error = RuntimeError("Synthetic original inference failure")
    observed["prediction_error"] = original_error
    atomic = PREDICT.atomic_json

    def unwritable_failure(path, record):
        if record["status"] == "failed":
            raise OSError("Synthetic failure-summary write error")
        return atomic(path, record)

    monkeypatch.setattr(PREDICT, "atomic_json", unwritable_failure)
    with pytest.raises(RuntimeError) as raised:
        PREDICT.run(args)
    assert raised.value is original_error
    assert read_json(args.output_dir / "prediction_summary.json")["status"] == "running"
    assert (args.output_dir / "predictions" / "case_0900_mask.png").is_file()
    assert "Could not record prediction failure" in capsys.readouterr().err


def test_progress_summary_write_error_remains_original_when_failure_write_also_breaks(environment, monkeypatch):
    args, observed = environment
    original_error = OSError("Synthetic original progress-summary write error")
    writes = []

    def unwritable_summary(path, record):
        writes.append(record["status"])
        if len(writes) == 1:
            raise original_error
        raise OSError("Synthetic secondary failure-summary write error")

    monkeypatch.setattr(PREDICT, "atomic_json", unwritable_summary)
    with pytest.raises(OSError) as raised:
        PREDICT.run(args)
    assert raised.value is original_error and writes == ["running", "failed"]
    assert len(observed["predict_calls"]) == 1
    assert len((args.output_dir / "progress.jsonl").read_text().splitlines()) == 1
    assert not (args.output_dir / "prediction_report.json").exists()


def test_concurrently_created_output_is_preserved_when_creation_loses_race(environment, monkeypatch):
    args, observed = environment
    original_mkdir = Path.mkdir
    winner = {"owner": "another process", "status": "completed"}

    def competing_mkdir(path, *values, **kwargs):
        if path == args.output_dir:
            original_mkdir(path, parents=True)
            write_json(path / "prediction_summary.json", winner)
            raise FileExistsError("Synthetic concurrent output creation")
        return original_mkdir(path, *values, **kwargs)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)
    with pytest.raises(FileExistsError, match="concurrent output"):
        PREDICT.run(args)
    assert read_json(args.output_dir / "prediction_summary.json") == winner
    assert set(snapshot(args.output_dir)) == {"prediction_summary.json"}
    assert not observed["predict_calls"]


@pytest.mark.parametrize("arguments", [["--unexpected-option"], ["--batch-size", "3"],
                                        ["--cpu-threads", "two"]])
def test_parser_rejects_invalid_arguments_before_runtime(environment, arguments):
    _, observed = environment
    with pytest.raises(SystemExit) as raised:
        PREDICT.main(arguments)
    assert raised.value.code == 2 and not observed["runtime_load_calls"]


def test_main_returns_interrupt_status_after_recording_partial_progress(environment):
    args, observed = environment
    observed["prediction_error"] = KeyboardInterrupt()
    assert PREDICT.main(argv_for(args)) == 130
    summary = read_json(args.output_dir / "prediction_summary.json")
    assert summary["status"] == "interrupted" and summary["last_reported_progress"] == observed["persisted_progress"]


def test_main_reports_incomplete_receipt_without_output_or_prediction(environment):
    args, observed = environment
    access = read_json(observed["access_path"])
    access["final_test_completed"] = False
    write_json(observed["access_path"], access)
    with pytest.raises(SystemExit) as raised:
        PREDICT.main(argv_for(args))
    assert raised.value.code == 2 and not args.output_dir.exists() and not observed["predict_calls"]
