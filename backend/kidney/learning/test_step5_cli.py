"""Step 5 lock preservation checks with fake audits, not original PNG data."""
from argparse import Namespace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from kidney_segmentation import cohort

SPEC = importlib.util.spec_from_file_location(
    "step5_cli_under_test", WORKSPACE / "learning" / "05_audit_and_lock_cohort.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def fake_audit():
    """Seventy unique synthetic metadata records; no source files are needed."""
    audit = {
        "schema_version": "png_dataset_audit_v1", "status": "passed",
        "manifest_sha256": hashlib.sha256(b"test manifest only").hexdigest(),
        "original_split_sha256": hashlib.sha256(b"test old splits only").hexdigest(),
        "scope": "selected_manifest_slices", "issues": [], "warnings": [],
        "duplicate_patient_components": [], "assets": {}, "patients": [],
    }
    for index in range(70):
        patient = "case_%05d" % index
        audit["patients"].append({
            "patient_id": patient, "source_patient_id": "kits19:" + patient,
            "eligible": True, "selected_slice_count": index + 1,
            "content_fingerprint_sha256": hashlib.sha256(patient.encode()).hexdigest(),
        })
    audit["audit_fingerprint_sha256"] = cohort.canonical_fingerprint(
        audit, "audit_fingerprint_sha256")
    return audit


@pytest.fixture
def environment(tmp_path, monkeypatch):
    """Stub only auditing/exposure discovery; retain actual cohort validation."""
    original = tmp_path / "original_data"
    original.mkdir()
    args = Namespace(root=original, output_dir=tmp_path / "outputs", seed=42,
                     exposed_case=[])
    state = {"audit": fake_audit(), "calls": []}

    def audit_stub(root, progress=None):
        assert root == original.resolve()
        state["calls"].append(root)
        if progress:
            progress({"completed_slice_count": 70, "total_slice_count": 70})
        return copy.deepcopy(state["audit"])

    monkeypatch.setattr(CLI, "load_helpers", lambda: (
        SimpleNamespace(audit_png_dataset=audit_stub), cohort))
    monkeypatch.setattr(CLI, "registered_exposures", lambda starter, additional:
                        sorted({"case_00000", *additional}))
    return args, state


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bound_files(args):
    return args.output_dir / "project40_splits.json", args.output_dir / "png_data_audit.json"


def test_create_lock_cannot_overwrite_an_existing_file(tmp_path):
    path = tmp_path / "project40_splits.json"
    original = b"Existing immutable evidence.\n"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="already exists"):
        CLI.create_lock(path, {"replacement": "must never be written"})
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_first_run_creates_valid_locked_cohort_and_bound_audit(environment):
    args, state = environment
    summary = CLI.run(args)
    lock_path, audit_path = bound_files(args)
    lock, audit = read_json(lock_path), read_json(audit_path)
    assert summary["lock_action"] == "created_new_lock"
    assert summary["patient_counts"] == {"train": 20, "validation": 10, "test": 10, "prediction": 5}
    assert summary["model_run"] is False
    assert summary["formal_training_performed"] is False
    assert audit == state["audit"]
    assert lock["audit_fingerprint_sha256"] == audit["audit_fingerprint_sha256"]
    assert "case_00000" in {record["patient_id"] for record in lock["pools"]["train"]}
    cohort.validate_cohort(lock, audit)
    assert list(args.root.iterdir()) == []
    assert len(state["calls"]) == 1


def test_identical_repeat_reuses_lock_without_changing_evidence_bytes(environment):
    args, state = environment
    CLI.run(args)
    lock_path, audit_path = bound_files(args)
    before_lock, before_audit = lock_path.read_bytes(), audit_path.read_bytes()
    summary = CLI.run(args)
    assert summary["lock_action"] == "verified_existing_lock"
    assert lock_path.read_bytes() == before_lock
    assert audit_path.read_bytes() == before_audit
    assert len(state["calls"]) == 2
    assert not (args.output_dir / "latest_audit_attempt.json").exists()


def test_changed_seed_rejected_preserving_lock_and_bound_audit(environment):
    args, state = environment
    CLI.run(args)
    lock_path, audit_path = bound_files(args)
    before_lock, before_audit = lock_path.read_bytes(), audit_path.read_bytes()
    args.seed = 43
    with pytest.raises(ValueError, match="Existing cohort lock was preserved"):
        CLI.run(args)
    assert lock_path.read_bytes() == before_lock
    assert audit_path.read_bytes() == before_audit
    assert read_json(args.output_dir / "latest_audit_attempt.json") == state["audit"]
    summary = read_json(args.output_dir / "step5_summary.json")
    assert summary["status"] == "existing_lock_rejected"
    assert summary["existing_lock_preserved"] is True
    assert summary["bound_audit_preserved"] is True
    assert "seed" in summary["reason"]


def test_failed_audit_preserves_existing_lock_and_bound_audit(environment):
    args, state = environment
    CLI.run(args)
    lock_path, audit_path = bound_files(args)
    before_lock, before_audit = lock_path.read_bytes(), audit_path.read_bytes()
    state["audit"]["status"] = "failed"
    state["audit"]["issues"] = [{"code": "fixture_corrupt_png", "severity": "error"}]
    state["audit"]["audit_fingerprint_sha256"] = cohort.canonical_fingerprint(
        state["audit"], "audit_fingerprint_sha256")
    with pytest.raises(ValueError, match="PNG data checks failed"):
        CLI.run(args)
    assert lock_path.read_bytes() == before_lock
    assert audit_path.read_bytes() == before_audit
    assert read_json(args.output_dir / "latest_audit_attempt.json") == state["audit"]
    summary = read_json(args.output_dir / "step5_summary.json")
    assert summary["status"] == "data_check_failed"
    assert summary["existing_lock_preserved"] is True
    assert summary["cohort_lock_created"] is False
    assert list(args.root.iterdir()) == []


@pytest.mark.parametrize("invalid_record, message", [
    (None, "must contain a JSON object"),
    ([], "must contain a JSON object"),
    ({"development_inspection": None}, "Invalid development_inspection object"),
    ({"development_inspection": []}, "Invalid development_inspection object"),
    ({"development_exposure": None}, "Invalid development_exposure object"),
    ({"development_exposure": []}, "Invalid development_exposure object"),
    ({"checkpoint_metadata": None}, "Invalid checkpoint exposure object"),
    ({"checkpoint_metadata": []}, "Invalid checkpoint exposure object"),
    ({"checkpoint_metadata": {"development_exposure": None}},
     "Invalid checkpoint exposure object"),
    ({"checkpoint_metadata": {"development_exposure": []}},
     "Invalid checkpoint exposure object"),
])
def test_malformed_development_reports_fail_without_losing_saved_registration(
        tmp_path, invalid_record, message):
    starter = tmp_path / "starter"
    reports = starter / "learning_outputs"
    valid_path = reports / "earlier" / "inspection.json"
    invalid_path = reports / "later" / "training_step.json"
    valid_path.parent.mkdir(parents=True)
    invalid_path.parent.mkdir(parents=True)
    valid_path.write_text(json.dumps({
        "development_inspection": {"case_inspected_for_development": "case_00007"},
        "development_exposure": {"case_inspected_for_development": "case_00008"},
        "checkpoint_metadata": {"development_exposure": {"patient_id": "case_00009"}},
    }), encoding="utf-8")
    invalid_path.write_text(json.dumps(invalid_record), encoding="utf-8")
    valid_bytes, invalid_bytes = valid_path.read_bytes(), invalid_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        CLI.registered_exposures(starter, ["case_00123"])

    # Invalid saved evidence is rejected, rather than silently dropping a case.
    assert valid_path.read_bytes() == valid_bytes
    assert invalid_path.read_bytes() == invalid_bytes
    assert sorted(path.name for path in reports.rglob("*.json")) == [
        "inspection.json", "training_step.json"]
    invalid_path.unlink()
    assert CLI.registered_exposures(starter, ["case_00123"]) == [
        "case_00000", "case_00007", "case_00008", "case_00009", "case_00123"]


def test_lock_edit_during_audit_rejected_preserving_bound_audit(environment, monkeypatch):
    args, state = environment
    CLI.run(args)
    lock_path, audit_path = bound_files(args)
    before_lock, before_audit = lock_path.read_bytes(), audit_path.read_bytes()
    changed_lock = before_lock + b"\n"

    def audit_with_concurrent_edit(root, progress=None):
        assert root == args.root.resolve()
        state["calls"].append(root)
        # Whitespace preserves JSON semantics, so only the byte-change guard catches it.
        lock_path.write_bytes(changed_lock)
        if progress:
            progress({"completed_slice_count": 70, "total_slice_count": 70})
        return copy.deepcopy(state["audit"])

    monkeypatch.setattr(CLI, "load_helpers", lambda: (
        SimpleNamespace(audit_png_dataset=audit_with_concurrent_edit), cohort))
    with pytest.raises(ValueError, match="lock changed while the audit was running"):
        CLI.run(args)

    # The CLI must not rewrite an externally edited lock or its original bound audit.
    assert lock_path.read_bytes() == changed_lock
    assert audit_path.read_bytes() == before_audit
    assert read_json(args.output_dir / "latest_audit_attempt.json") == state["audit"]
    summary = read_json(args.output_dir / "step5_summary.json")
    assert summary["status"] == "existing_lock_rejected"
    assert summary["existing_lock_preserved"] is True
    assert summary["bound_audit_preserved"] is True
    assert "lock changed while the audit was running" in summary["reason"]
    assert len(state["calls"]) == 2
    assert list(args.root.iterdir()) == []
