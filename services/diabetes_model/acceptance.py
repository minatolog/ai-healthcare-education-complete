"""Final delivery acceptance checks for the diabetes ML package."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .artifact_manifest import verify_artifact_manifest
from .explanations import LOCAL_EXPLANATION_METHOD
from .features import MODEL_FEATURES
from .model_artifact import load_model_bundle
from .prediction import predict_patient
from .schema import EXPECTED_PATIENT_COUNT, load_reference_cohort
from .workflow import DEFAULT_COHORT_PATH, SESSION_SCHEMA_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIRECTORY = REPOSITORY_ROOT / "artifacts" / "diabetes_demo"
DELIVERY_VERSION = "1.0.0"


class DeliveryAcceptanceError(ValueError):
    """Raised internally when a delivery requirement is not satisfied."""


@dataclass(frozen=True)
class AcceptanceCheck:
    """Outcome of one independently evaluated delivery check."""

    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DeliveryAcceptanceReport:
    """Aggregate readiness result for backend handoff."""

    delivery_version: str
    checks: tuple[AcceptanceCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "delivery_version": self.delivery_version,
            "status": "passed" if self.passed else "failed",
            "educational_only": True,
            "check_count": len(self.checks),
            "checks": [check.as_dict() for check in self.checks],
        }


def run_delivery_acceptance(
    artifact_directory: str | Path = DEFAULT_ARTIFACT_DIRECTORY,
) -> DeliveryAcceptanceReport:
    """Check the committed data, model, contracts, and generated artifacts."""

    artifact_path = Path(artifact_directory)
    checks = (
        _run_check("synthetic_cohort", _check_cohort),
        _run_check(
            "artifact_integrity",
            lambda: _check_manifest(artifact_path),
        ),
        _run_check(
            "model_bundle",
            lambda: _check_model_bundle(artifact_path),
        ),
        _run_check(
            "prediction_contract",
            lambda: _check_prediction_contract(artifact_path),
        ),
        _run_check(
            "training_session_contract",
            lambda: _check_training_session(artifact_path),
        ),
        _run_check(
            "progress_event_contract",
            lambda: _check_progress_events(artifact_path),
        ),
        _run_check(
            "evaluation_consistency",
            lambda: _check_evaluation_consistency(artifact_path),
        ),
        _run_check(
            "local_explanation_contract",
            lambda: _check_local_explanation(artifact_path),
        ),
    )
    return DeliveryAcceptanceReport(
        delivery_version=DELIVERY_VERSION,
        checks=checks,
    )


def _run_check(name: str, check: Callable[[], str]) -> AcceptanceCheck:
    try:
        detail = check()
    except Exception as exc:
        return AcceptanceCheck(name=name, passed=False, detail=str(exc))
    return AcceptanceCheck(name=name, passed=True, detail=detail)


def _check_cohort() -> str:
    records = load_reference_cohort(DEFAULT_COHORT_PATH)
    _require(
        len(records) == EXPECTED_PATIENT_COUNT,
        f"Expected {EXPECTED_PATIENT_COUNT} synthetic patients.",
    )
    return "40 validated synthetic patients"


def _check_manifest(artifact_path: Path) -> str:
    result = verify_artifact_manifest(
        artifact_path / "artifact_manifest.json"
    )
    return f"{len(result.verified_files)} artifact checksums verified"


def _check_model_bundle(artifact_path: Path) -> str:
    bundle = load_model_bundle(
        artifact_path / "diabetes_random_forest.joblib"
    )
    _require(bundle.educational_only, "Model must be educational_only.")
    _require(
        bundle.feature_names == MODEL_FEATURES,
        "Model feature order does not match the contract.",
    )
    _require(bundle.class_labels == (0, 1), "Model classes must be 0 and 1.")
    return (
        f"Random Forest loaded with {bundle.best_tree_count} trees and "
        f"{len(bundle.feature_names)} features"
    )


def _check_prediction_contract(artifact_path: Path) -> str:
    example = _read_json(artifact_path / "backend_prediction_example.json")
    _require(set(example) == {"request", "response"}, "Invalid prediction example.")
    bundle = load_model_bundle(
        artifact_path / "diabetes_random_forest.joblib"
    )
    actual = predict_patient(bundle, example["request"]).as_dict()
    _require(actual == example["response"], "Prediction example is not reproducible.")
    _require(actual["educational_only"] is True, "Missing educational disclaimer.")
    return "JSON request reproduces the committed label and probability"


def _check_training_session(artifact_path: Path) -> str:
    session = _read_json(artifact_path / "training_session_response.json")
    _require(
        session.get("schema_version") == SESSION_SCHEMA_VERSION,
        "Training-session schema version mismatch.",
    )
    _require(session.get("educational_only") is True, "Session disclaimer missing.")
    _require(
        session.get("split_counts")
        == {"training": 20, "validation": 10, "test": 10},
        "Session must contain a 20/10/10 split.",
    )
    _require(
        session["selection_assessment"]["ready"] is True,
        "Demo selection must pass pre-flight validation.",
    )
    _require(
        len(session["training_curve"]["points"]) == 20,
        "Training session must contain 20 checkpoints.",
    )
    _require(
        session["test_evaluation"]["test_sample_count"] == 10,
        "Training session must contain ten test predictions.",
    )
    _require(
        len(session["test_explanations"]["patients"]) == 10,
        "Training session must contain ten local explanations.",
    )
    return f"schema {SESSION_SCHEMA_VERSION} with complete model outputs"


def _check_progress_events(artifact_path: Path) -> str:
    payload = _read_json(artifact_path / "training_progress_events.json")
    events = payload.get("events")
    _require(isinstance(events, list) and len(events) == 20, "Expected 20 events.")
    _require(events[0]["step"] == 1, "First progress step must be 1.")
    _require(events[0]["progress_percent"] == 5.0, "First progress must be 5%.")
    _require(events[-1]["step"] == 20, "Final progress step must be 20.")
    _require(
        events[-1]["progress_percent"] == 100.0,
        "Final progress must be 100%.",
    )
    session = _read_json(artifact_path / "training_session_response.json")
    curve_points = session["training_curve"]["points"]
    stripped_points = [
        {key: value for key, value in point.items() if key != "loss"}
        for point in curve_points
    ]
    _require(events == stripped_points, "Callback events and curve points differ.")
    return "20 ordered checkpoint events from 5% to 100%"


def _check_evaluation_consistency(artifact_path: Path) -> str:
    evaluation = _read_json(artifact_path / "test_evaluation.json")
    session = _read_json(artifact_path / "training_session_response.json")
    _require(
        evaluation == session["test_evaluation"],
        "Standalone and session evaluation results differ.",
    )
    _require(
        evaluation["metrics"]["accuracy"] == 0.9,
        "Committed demo test accuracy changed unexpectedly.",
    )
    return "standalone and session test results match at 90% accuracy"


def _check_local_explanation(artifact_path: Path) -> str:
    explanation = _read_json(
        artifact_path / "patient_explanation_example.json"
    )
    _require(
        explanation.get("method") == LOCAL_EXPLANATION_METHOD,
        "Local explanation method mismatch.",
    )
    _require(
        len(explanation.get("feature_effects", [])) == len(MODEL_FEATURES),
        "Local explanation must contain all six features.",
    )
    _require("actual_label" not in explanation, "Explanation leaks actual label.")
    session = _read_json(artifact_path / "training_session_response.json")
    _require(
        explanation == session["test_explanations"]["patients"][0],
        "Explanation example does not match the session response.",
    )
    return "six-feature local explanation matches the session response"


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeliveryAcceptanceError(f"Could not read JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise DeliveryAcceptanceError(f"Expected JSON object: {path.name}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeliveryAcceptanceError(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run final acceptance checks for the diabetes ML delivery."
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run_delivery_acceptance(args.artifact_directory)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"Diabetes ML delivery {report.delivery_version}: " + (
            "PASSED" if report.passed else "FAILED"
        ))
        for check in report.checks:
            marker = "PASS" if check.passed else "FAIL"
            print(f"[{marker}] {check.name}: {check.detail}")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
