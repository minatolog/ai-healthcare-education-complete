"""Step 7 orchestration tests with stub evaluation and tiny fake evidence files."""
from argparse import Namespace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKSPACE = Path(__file__).resolve().parents[1]


def load_cli(name, filename):
    spec = importlib.util.spec_from_file_location(name, WORKSPACE / "learning" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FREEZE = load_cli("step7a_cli_under_test", "07a_freeze_model.py")
EVALUATE = load_cli("step7b_cli_under_test", "07b_evaluate_test.py")
COMMON = load_cli("step7_cli_helpers_under_test", "png_final_cli_common.py")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


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
    root = tmp_path / "original_data"
    root.mkdir()
    run_dir = tmp_path / "source_run"
    run_dir.mkdir()
    checkpoint = run_dir / "best_model.pt"
    checkpoint.write_bytes(b"Synthetic checkpoint placeholder; never loaded as weights.\n")
    (run_dir / "run_summary.json").write_text('{"fixture_only": true}\n', encoding="utf-8")
    cohort, audit = tmp_path / "cohort.json", tmp_path / "audit.json"
    cohort.write_text('{"fixture_only": "cohort"}\n', encoding="utf-8")
    audit.write_text('{"fixture_only": "audit"}\n', encoding="utf-8")
    common_args = dict(root=root, cohort=cohort, audit=audit, run_dir=run_dir, check_only=False)
    freeze_args = Namespace(**common_args, output_dir=tmp_path / "new_freezes" / "freeze01")
    freeze_dir = tmp_path / "source_freeze"
    freeze_dir.mkdir()
    freeze_path = freeze_dir / "frozen_model.json"
    evaluate_args = Namespace(**common_args, freeze=freeze_path,
                              output_dir=tmp_path / "evaluations" / "test01",
                              batch_size=2, cpu_threads=2)
    record = {
        "schema_version": "png_frozen_model_v1", "fixture_only": True,
        "selected_epoch": 3, "checkpoint_sha256": digest(checkpoint.read_bytes()),
        "checkpoint_path": str(checkpoint),
        "freeze_fingerprint_sha256": digest(b"synthetic freeze fingerprint"),
        "patient_counts": {"train": 20, "validation": 10, "test": 10, "prediction": 5},
        "selected_slice_counts": {"train": 40, "validation": 20, "test": 20, "prediction": 10},
        "bound_file_sha256": {"cohort": digest(cohort.read_bytes()), "audit": digest(audit.read_bytes())},
        "checkpoint_selection": {"metric": "tumour_dice", "direction": "maximize", "best_score": 0.2},
        "loss_configuration": {"ce_class_weights": [1.0, 1.0, 2.0]},
        "final_test_performed": False, "independent_prediction_performed": False,
    }
    COMMON.write_new_json(freeze_path, record)
    context = {"root": root.resolve()}
    completed = {"context": context, "run_dir": run_dir.resolve(), "fixture_only": True}
    events = [
        {"event": "batch", "phase": "test", "batch_index": 1, "total_batches": 10, "slices_seen": 2},
        {"event": "batch", "phase": "test", "batch_index": 2, "total_batches": 10, "slices_seen": 4},
    ]
    report = {
        "schema_version": "png_final_test_report_v1", "fixture_only": True,
        "split": "test", "patient_count": 10, "slice_count": 20,
        "patient_mean": {"kidney": {"dice": 0.6, "iou": 0.45},
                         "tumour": {"dice": 0.3, "iou": 0.2}},
    }
    observed = {
        "runtime_load_calls": [], "completed_calls": [], "make_calls": [], "validate_calls": [],
        "frozen_calls": [], "evaluate_calls": [], "binding_error": None, "validation_error": None,
        "evaluation_error": None, "record": record, "events": events, "report": report,
        "access_path": freeze_dir / "test_access.json", "completed": completed,
    }

    def inspect_completed(actual_root, actual_cohort, actual_audit, actual_run):
        observed["completed_calls"].append((actual_root, actual_cohort, actual_audit, actual_run))
        assert (actual_root, actual_cohort, actual_audit, actual_run) == (root, cohort, audit, run_dir)
        if observed["binding_error"] is not None:
            raise observed["binding_error"]
        return completed

    def make_record(inspected):
        assert inspected is completed
        observed["make_calls"].append(inspected)
        return copy.deepcopy(record)

    def validate_record(actual_record, inspected):
        assert actual_record == record and inspected is completed
        observed["validate_calls"].append(actual_record)
        if observed["validation_error"] is not None:
            raise observed["validation_error"]

    def inspect_frozen(actual_root, actual_cohort, actual_audit, actual_run, path):
        observed["frozen_calls"].append(path)
        assert (actual_root, actual_cohort, actual_audit, actual_run) == (root, cohort, audit, run_dir)
        if observed["binding_error"] is not None:
            raise observed["binding_error"]
        assert read_json(path) == record
        return {"context": context, "run_dir": run_dir.resolve(), "freeze_path": str(path.resolve()),
                "freeze_record": read_json(path), "freeze_sha256": digest(path.read_bytes())}

    def evaluate_stub(inspected, **kwargs):
        observed["evaluate_calls"].append(kwargs)
        assert inspected["freeze_path"] == str(freeze_path.resolve())
        access = read_json(observed["access_path"])
        summary_path = evaluate_args.output_dir / "evaluation_summary.json"
        running = read_json(summary_path)
        assert access == running
        assert running["status"] == "running" and running["final_test_opened"] is True
        assert running["final_test_completed"] is False and running["last_reported_progress"] is None
        assert running["output_dir"] == str(evaluate_args.output_dir.resolve())
        observed["initial_running_state"] = running
        for event in events:
            kwargs["progress"](copy.deepcopy(event))
            # Evidence must already be on disk when the evaluator reports its next event or fails.
            lines = (evaluate_args.output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            saved = json.loads(lines[-1])
            assert {key: saved[key] for key in event} == event
            assert read_json(summary_path)["last_reported_progress"] == saved
            observed["persisted_progress"] = saved
        if observed["evaluation_error"] is not None:
            raise observed["evaluation_error"]
        return copy.deepcopy(report)

    runtime = SimpleNamespace(inspect_completed_run=inspect_completed, make_freeze_record=make_record,
                              validate_freeze_record=validate_record, inspect_frozen_model=inspect_frozen,
                              evaluate_frozen_test=evaluate_stub)

    def load_runtime_stub():
        observed["runtime_load_calls"].append(True)
        return runtime

    monkeypatch.setattr(FREEZE, "load_runtime", load_runtime_stub)
    monkeypatch.setattr(EVALUATE, "load_runtime", load_runtime_stub)
    return freeze_args, evaluate_args, observed


def test_freeze_check_only_validates_metadata_without_writes_or_evaluation(environment, tmp_path):
    args, _, observed = environment
    args.check_only = True
    before = snapshot(tmp_path)
    result = FREEZE.run(args)
    assert result["status"] == "preflight_completed" and result["freeze_record"] == observed["record"]
    assert result["model_initialized"] is False and result["pngs_decoded"] == 0
    assert result["training_performed"] is False and result["final_test_performed"] is False
    assert len(observed["completed_calls"]) == len(observed["validate_calls"]) == 1
    assert not observed["frozen_calls"] and not observed["evaluate_calls"]
    assert not args.output_dir.parent.exists() and snapshot(tmp_path) == before


def test_freeze_creates_and_rechecks_immutable_metadata_without_copying_checkpoint(environment, tmp_path):
    args, _, observed = environment
    before_sources = snapshot(args.root), snapshot(args.run_dir), snapshot(tmp_path / "source_freeze")
    result = FREEZE.run(args)
    path = args.output_dir / "frozen_model.json"
    assert result["status"] == "frozen" and result["freeze_path"] == str(path)
    assert read_json(path) == result["freeze_record"] == observed["record"]
    assert result["freeze_sha256"] == digest(path.read_bytes())
    assert set(snapshot(args.output_dir)) == {"frozen_model.json"}
    assert observed["frozen_calls"] == [path] and not observed["evaluate_calls"]
    assert result["model_initialized"] is False and result["pngs_decoded"] == 0
    assert result["training_performed"] is False and result["final_test_performed"] is False
    assert (snapshot(args.root), snapshot(args.run_dir), snapshot(tmp_path / "source_freeze")) == before_sources
    saved = snapshot(args.output_dir)
    with pytest.raises(ValueError, match="already exists"):
        FREEZE.run(args)
    assert snapshot(args.output_dir) == saved and len(observed["frozen_calls"]) == 1


@pytest.mark.parametrize("which", ["freeze", "evaluate"])
@pytest.mark.parametrize("check_only", [False, True])
def test_existing_output_is_preserved_before_freezing_or_test_access(environment, which, check_only):
    freeze_args, evaluate_args, observed = environment
    cli, args = (FREEZE, freeze_args) if which == "freeze" else (EVALUATE, evaluate_args)
    args.check_only = check_only
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "human_notes.txt").write_text("Preserve this existing evidence.\n", encoding="utf-8")
    before = snapshot(args.output_dir)
    with pytest.raises(ValueError, match="already exists"):
        cli.run(args)
    assert snapshot(args.output_dir) == before
    assert not observed["make_calls"] and not observed["evaluate_calls"]
    assert not observed["access_path"].exists()


@pytest.mark.parametrize("which, location", [
    ("freeze", "data"), ("freeze", "run"),
    ("evaluate", "data"), ("evaluate", "run"), ("evaluate", "freeze"),
])
def test_output_inside_protected_source_is_rejected_before_writes(environment, which, location, tmp_path):
    freeze_args, evaluate_args, observed = environment
    cli, args = (FREEZE, freeze_args) if which == "freeze" else (EVALUATE, evaluate_args)
    protected = {"data": args.root, "run": args.run_dir, "freeze": evaluate_args.freeze.parent}[location]
    args.output_dir = protected / "unsafe_output"
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        cli.run(args)
    assert not observed["make_calls"] and not observed["evaluate_calls"]
    assert snapshot(tmp_path) == before and not args.output_dir.exists()


@pytest.mark.parametrize("which", ["freeze", "evaluate"])
@pytest.mark.parametrize("symlink_kind", ["output", "parent"])
def test_symlink_output_or_parent_is_rejected_before_writes(environment, which, symlink_kind, tmp_path):
    freeze_args, evaluate_args, observed = environment
    cli, args = (FREEZE, freeze_args) if which == "freeze" else (EVALUATE, evaluate_args)
    target = tmp_path / "symlink_target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    args.output_dir = alias if symlink_kind == "output" else alias / "new_output"
    with pytest.raises(ValueError, match="symlink"):
        cli.run(args)
    assert snapshot(target) == {} and not observed["access_path"].exists()
    assert not observed["make_calls"] and not observed["evaluate_calls"]


def test_evaluation_check_only_opens_no_test_and_creates_no_output_or_access_lock(environment, tmp_path):
    _, args, observed = environment
    args.check_only = True
    before = snapshot(tmp_path)
    result = EVALUATE.run(args)
    assert result["status"] == "preflight_completed" and result["test_opened"] is False
    assert result["model_initialized"] is False and result["pngs_decoded"] == 0
    assert result["training_performed"] is False
    assert observed["frozen_calls"] == [args.freeze] and not observed["evaluate_calls"]
    assert not args.output_dir.parent.exists() and not observed["access_path"].exists()
    assert snapshot(tmp_path) == before


def test_completed_evaluation_persists_progress_report_access_lock_and_sha_evidence(environment):
    _, args, observed = environment
    sources_before = snapshot(args.root), snapshot(args.run_dir), args.freeze.read_bytes()
    result = EVALUATE.run(args)
    report_path = args.output_dir / "test_report.json"
    assert result["schema_version"] == "png_final_test_run_v1" and result["status"] == "completed"
    assert result["final_test_opened"] is True and result["final_test_completed"] is True
    assert result["training_performed"] is False and result["model_selection_performed"] is False
    assert result["independent_prediction_performed"] is False
    assert result["checkpoint_sha256"] == observed["record"]["checkpoint_sha256"]
    assert result["freeze_sha256"] == digest(args.freeze.read_bytes())
    assert result["freeze_fingerprint_sha256"] == observed["record"]["freeze_fingerprint_sha256"]
    assert result["batch_size"] == args.batch_size and result["cpu_threads"] == args.cpu_threads
    assert result["output_dir"] == str(args.output_dir.resolve())
    assert read_json(report_path) == observed["report"]
    assert result["patient_mean"] == observed["report"]["patient_mean"]
    assert result["test_report_sha256"] == digest(report_path.read_bytes())
    assert result["artifact_sha256"] == {"test_report.json": digest(report_path.read_bytes())}
    assert read_json(args.output_dir / "evaluation_summary.json") == read_json(observed["access_path"]) == result
    progress = [json.loads(line) for line in (args.output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [{key: saved[key] for key in original} for saved, original in zip(progress, observed["events"])] == observed["events"]
    assert len(progress) == 2 and result["last_reported_progress"] == progress[-1]
    assert all(event["recorded_at_utc"] for event in progress)
    assert result["started_at_utc"] and result["finished_at_utc"]
    assert len(observed["evaluate_calls"]) == 1
    assert observed["evaluate_calls"][0]["batch_size"] == 2
    assert observed["evaluate_calls"][0]["num_threads"] == 2
    assert (snapshot(args.root), snapshot(args.run_dir), args.freeze.read_bytes()) == sources_before


@pytest.mark.parametrize("check_only", [False, True])
def test_rerun_with_different_output_is_rejected_without_evaluation(environment, tmp_path, check_only):
    _, args, observed = environment
    EVALUATE.run(args)
    original_output, original_access = snapshot(args.output_dir), observed["access_path"].read_bytes()
    args.output_dir, args.check_only = tmp_path / "another_output", check_only
    with pytest.raises(ValueError, match="already opened"):
        EVALUATE.run(args)
    assert len(observed["evaluate_calls"]) == 1 and not args.output_dir.exists()
    assert observed["access_path"].read_bytes() == original_access
    assert snapshot(tmp_path / "evaluations" / "test01") == original_output


@pytest.mark.parametrize("error, status", [
    (RuntimeError("Synthetic final-test failure"), "failed"), (KeyboardInterrupt(), "interrupted"),
])
def test_failure_or_interruption_keeps_progress_and_access_lock_without_completion(environment, tmp_path, error, status):
    _, args, observed = environment
    observed["evaluation_error"] = error
    with pytest.raises(type(error)):
        EVALUATE.run(args)
    summary = read_json(args.output_dir / "evaluation_summary.json")
    assert summary == read_json(observed["access_path"])
    assert summary["status"] == status and summary["final_test_opened"] is True
    assert summary["final_test_completed"] is False and summary["error_type"] == type(error).__name__
    assert summary["last_reported_progress"] == observed["persisted_progress"]
    progress = [json.loads(line) for line in (args.output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(progress) == 2 and progress[-1] == summary["last_reported_progress"]
    assert "test_report_sha256" not in summary and "artifact_sha256" not in summary
    assert not (args.output_dir / "test_report.json").exists()
    assert summary["finished_at_utc"]
    if status == "failed":
        assert summary["error_message"] == "Synthetic final-test failure"
    failed_output = snapshot(args.output_dir)
    args.output_dir = tmp_path / "retry_after_failure"
    with pytest.raises(ValueError, match="already opened"):
        EVALUATE.run(args)
    assert len(observed["evaluate_calls"]) == 1 and not args.output_dir.exists()
    assert snapshot(tmp_path / "evaluations" / "test01") == failed_output


def test_evaluation_output_creation_failure_still_preserves_reserved_access_lock(environment, monkeypatch):
    _, args, observed = environment
    original_mkdir = Path.mkdir

    def failing_mkdir(path, *values, **kwargs):
        if path == args.output_dir:
            raise OSError("Synthetic output creation failure")
        return original_mkdir(path, *values, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)
    with pytest.raises(OSError, match="output creation"):
        EVALUATE.run(args)
    access = read_json(observed["access_path"])
    assert access["status"] == "failed" and access["final_test_opened"] is True
    assert access["final_test_completed"] is False and access["last_reported_progress"] is None
    assert access["error_type"] == "OSError" and not args.output_dir.exists()
    assert not observed["evaluate_calls"]


def test_concurrently_created_output_is_preserved_when_mkdir_loses_race(environment, monkeypatch):
    _, args, observed = environment
    original_mkdir = Path.mkdir
    competing_summary = {"fixture_only": True, "status": "completed", "owner": "another process"}

    def competing_mkdir(path, *values, **kwargs):
        if path == args.output_dir:
            original_mkdir(path, parents=True)
            COMMON.write_new_json(path / "evaluation_summary.json", competing_summary)
            (path / "human_notes.txt").write_text("Preserve the winning process.\n", encoding="utf-8")
            raise FileExistsError("Synthetic concurrent output creation")
        return original_mkdir(path, *values, **kwargs)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)
    with pytest.raises(FileExistsError, match="concurrent output"):
        EVALUATE.run(args)
    assert read_json(args.output_dir / "evaluation_summary.json") == competing_summary
    assert (args.output_dir / "human_notes.txt").read_text(encoding="utf-8") == "Preserve the winning process.\n"
    assert set(snapshot(args.output_dir)) == {"evaluation_summary.json", "human_notes.txt"}
    access = read_json(observed["access_path"])
    assert access["status"] == "failed" and access["error_type"] == "FileExistsError"
    assert access["final_test_completed"] is False and not observed["evaluate_calls"]


def test_persistent_summary_write_error_still_records_failed_access_and_original_exception(environment, monkeypatch):
    _, args, observed = environment
    original_error = OSError("Synthetic original summary write error")
    writes, atomic = [], EVALUATE.atomic_json

    def unwritable_summary(path, record):
        if path == args.output_dir / "evaluation_summary.json":
            writes.append(record["status"])
            if len(writes) == 1:
                raise original_error
            raise OSError("Synthetic secondary failure-record write error")
        return atomic(path, record)

    monkeypatch.setattr(EVALUATE, "atomic_json", unwritable_summary)
    with pytest.raises(OSError) as error:
        EVALUATE.run(args)
    assert error.value is original_error and writes == ["running", "failed"]
    access = read_json(observed["access_path"])
    assert access["status"] == "failed" and access["error_message"] == str(original_error)
    assert access["error_type"] == "OSError" and access["final_test_completed"] is False
    assert not (args.output_dir / "evaluation_summary.json").exists()
    assert not observed["evaluate_calls"]


def test_failed_access_write_does_not_prevent_failed_summary_or_replace_evaluation_error(environment, monkeypatch):
    _, args, observed = environment
    original_error = RuntimeError("Synthetic original evaluator error")
    observed["evaluation_error"] = original_error
    atomic = EVALUATE.atomic_json

    def unwritable_failure_access(path, record):
        if path == observed["access_path"] and record["status"] == "failed":
            raise OSError("Synthetic access record write error")
        return atomic(path, record)

    monkeypatch.setattr(EVALUATE, "atomic_json", unwritable_failure_access)
    with pytest.raises(RuntimeError) as error:
        EVALUATE.run(args)
    assert error.value is original_error
    summary = read_json(args.output_dir / "evaluation_summary.json")
    assert summary["status"] == "failed" and summary["error_message"] == str(original_error)
    assert summary["last_reported_progress"] == observed["persisted_progress"]
    assert read_json(observed["access_path"])["status"] == "running"
    assert observed["access_path"].is_file() and summary["final_test_completed"] is False


def test_exclusive_access_creation_prevents_competing_process_from_opening_same_freeze(environment, monkeypatch):
    _, args, observed = environment
    competing = {"fixture_only": True, "status": "reserved", "output_dir": "another process"}
    write = EVALUATE.write_new_json

    def competing_lock(path, record):
        assert path == observed["access_path"]
        COMMON.write_new_json(path, competing)
        return write(path, record)

    monkeypatch.setattr(EVALUATE, "write_new_json", competing_lock)
    with pytest.raises(FileExistsError):
        EVALUATE.run(args)
    assert read_json(observed["access_path"]) == competing
    assert not args.output_dir.exists() and not observed["evaluate_calls"]


@pytest.mark.parametrize("dangling", [False, True])
def test_existing_access_symlink_cannot_bypass_one_time_test_guard(environment, tmp_path, dangling):
    _, args, observed = environment
    target = tmp_path / "old_access.json"
    if not dangling:
        target.write_text('{"preserve": true}\n', encoding="utf-8")
    observed["access_path"].symlink_to(target)
    with pytest.raises(ValueError, match="already opened"):
        EVALUATE.run(args)
    assert observed["access_path"].is_symlink() and not args.output_dir.exists()
    assert not observed["evaluate_calls"]
    if not dangling:
        assert read_json(target) == {"preserve": True}


@pytest.mark.parametrize("which", ["freeze", "evaluate"])
@pytest.mark.parametrize("check_only", [False, True])
def test_binding_failure_stops_before_any_writes_or_evaluation(environment, tmp_path, which, check_only):
    freeze_args, evaluate_args, observed = environment
    cli, args = (FREEZE, freeze_args) if which == "freeze" else (EVALUATE, evaluate_args)
    args.check_only = check_only
    observed["binding_error"] = ValueError("Synthetic current cohort binding mismatch")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="binding mismatch"):
        cli.run(args)
    assert snapshot(tmp_path) == before and not args.output_dir.exists()
    assert not observed["make_calls"] and not observed["evaluate_calls"]


def test_invalid_freeze_record_stops_before_output_creation(environment, tmp_path):
    args, _, observed = environment
    observed["validation_error"] = ValueError("Synthetic freeze fingerprint mismatch")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        FREEZE.run(args)
    assert snapshot(tmp_path) == before and not args.output_dir.exists()
    assert not observed["frozen_calls"] and not observed["evaluate_calls"]


@pytest.mark.parametrize("field, value", [("batch_size", 0), ("batch_size", 3),
                                         ("cpu_threads", 0), ("cpu_threads", 17)])
def test_invalid_evaluation_settings_fail_before_loading_runtime_or_writes(environment, tmp_path, field, value):
    _, args, observed = environment
    setattr(args, field, value)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="batch size"):
        EVALUATE.run(args)
    assert not observed["runtime_load_calls"] and not observed["frozen_calls"]
    assert not observed["evaluate_calls"] and snapshot(tmp_path) == before


@pytest.mark.parametrize("cli, arguments", [
    (FREEZE, ["--unexpected-option"]), (EVALUATE, ["--batch-size", "3"]),
    (EVALUATE, ["--cpu-threads", "two"]),
])
def test_parser_rejects_invalid_arguments_before_runtime(environment, cli, arguments):
    _, _, observed = environment
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2 and not observed["runtime_load_calls"]


def test_main_returns_interrupt_status_after_recording_interrupted_access(environment):
    _, args, observed = environment
    observed["evaluation_error"] = KeyboardInterrupt()
    assert EVALUATE.main(argv_for(args)) == 130
    assert read_json(observed["access_path"])["status"] == "interrupted"
    assert read_json(args.output_dir / "evaluation_summary.json")["final_test_completed"] is False


def test_main_reports_binding_error_without_creating_freeze(environment):
    args, _, observed = environment
    observed["binding_error"] = ValueError("Synthetic binding mismatch")
    with pytest.raises(SystemExit) as error:
        FREEZE.main(argv_for(args))
    assert error.value.code == 2 and not args.output_dir.exists()
    assert not observed["evaluate_calls"]


def test_json_helper_exclusive_creation_preserves_existing_evidence(tmp_path):
    path = tmp_path / "frozen_model.json"
    COMMON.write_new_json(path, {"selected_epoch": 3})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        COMMON.write_new_json(path, {"selected_epoch": 99})
    assert path.read_bytes() == before and COMMON.digest_file(path) == digest(before)


def test_atomic_json_preserves_previous_state_when_serialization_fails(tmp_path):
    path = tmp_path / "test_access.json"
    COMMON.write_new_json(path, {"status": "running"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        COMMON.atomic_json(path, {"status": "completed", "metric": float("nan")})
    assert path.read_bytes() == before and {item.name for item in tmp_path.iterdir()} == {path.name}
