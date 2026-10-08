"""Export and load the selected educational diabetes model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .features import FEATURE_DEFINITIONS, MODEL_FEATURES
from .training import RandomForestTrainingResult


ARTIFACT_VERSION = "1.0"
PREDICTION_THRESHOLD = 0.5
FEATURE_IMPORTANCE_NOTE = (
    "Random Forest split importance; it does not prove causation or clinical "
    "importance."
)
PROBABILITY_NOTE = (
    "Model-estimated educational score; not a calibrated clinical risk."
)


class ModelArtifactError(ValueError):
    """Raised when a model artifact cannot be exported or validated."""


@dataclass(frozen=True)
class FeatureImportanceEntry:
    """One ranked global feature-importance value."""

    rank: int
    feature_name: str
    display_name: str
    importance: float

    def as_dict(self) -> dict[str, int | str | float]:
        return {
            "rank": self.rank,
            "feature_name": self.feature_name,
            "display_name": self.display_name,
            "importance": round(self.importance, 6),
        }


@dataclass(frozen=True)
class ExportedModelPaths:
    """Paths written during model export."""

    model: Path
    metadata: Path


@dataclass(frozen=True)
class LoadedModelBundle:
    """Validated model and contract information loaded from disk."""

    model: RandomForestClassifier
    artifact_version: str
    feature_names: tuple[str, ...]
    class_labels: tuple[int, ...]
    prediction_threshold: float
    best_tree_count: int
    feature_importance: tuple[FeatureImportanceEntry, ...]
    sklearn_version: str
    educational_only: bool


def extract_feature_importance(
    training_result: RandomForestTrainingResult,
) -> tuple[FeatureImportanceEntry, ...]:
    """Return ranked Random Forest split importances for the six inputs."""

    if not isinstance(training_result, RandomForestTrainingResult):
        raise ModelArtifactError(
            "training_result must be a RandomForestTrainingResult."
        )
    try:
        check_is_fitted(training_result.model)
    except NotFittedError as exc:
        raise ModelArtifactError("The selected model has not been fitted.") from exc

    values = training_result.model.feature_importances_
    if len(values) != len(FEATURE_DEFINITIONS):
        raise ModelArtifactError("Model feature importance does not match the schema.")

    ranked = sorted(
        zip(FEATURE_DEFINITIONS, values),
        key=lambda item: (-float(item[1]), item[0].name),
    )
    return tuple(
        FeatureImportanceEntry(
            rank=rank,
            feature_name=definition.name,
            display_name=definition.display_name,
            importance=float(value),
        )
        for rank, (definition, value) in enumerate(ranked, start=1)
    )


def export_model_bundle(
    training_result: RandomForestTrainingResult,
    output_path: str | Path,
) -> ExportedModelPaths:
    """Write the fitted estimator plus a human-readable metadata sidecar."""

    feature_importance = extract_feature_importance(training_result)
    model_path = Path(output_path)
    if model_path.suffix != ".joblib":
        raise ModelArtifactError("The exported model filename must end in .joblib.")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "model_type": type(training_result.model).__name__,
        "feature_names": list(MODEL_FEATURES),
        "class_labels": [int(label) for label in training_result.model.classes_],
        "prediction_threshold": PREDICTION_THRESHOLD,
        "best_tree_count": training_result.best_tree_count,
        "training_config": asdict(training_result.config),
        "feature_importance": [entry.as_dict() for entry in feature_importance],
        "sklearn_version": sklearn.__version__,
        "educational_only": True,
        "probability_note": PROBABILITY_NOTE,
        "feature_importance_note": FEATURE_IMPORTANCE_NOTE,
    }
    payload = {**metadata, "model": training_result.model}
    joblib.dump(payload, model_path)

    metadata_path = model_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ExportedModelPaths(model=model_path, metadata=metadata_path)


def load_model_bundle(path: str | Path) -> LoadedModelBundle:
    """Load a validated bundle. Only load artifacts from a trusted source."""

    model_path = Path(path)
    try:
        payload = joblib.load(model_path)
    except Exception as exc:
        raise ModelArtifactError(f"Could not load model artifact: {model_path}") from exc

    required_keys = {
        "artifact_version",
        "model",
        "feature_names",
        "class_labels",
        "prediction_threshold",
        "best_tree_count",
        "feature_importance",
        "sklearn_version",
        "educational_only",
    }
    if not isinstance(payload, dict) or not required_keys <= set(payload):
        raise ModelArtifactError("Model artifact is missing required fields.")
    if payload["artifact_version"] != ARTIFACT_VERSION:
        raise ModelArtifactError(
            f"Unsupported artifact version: {payload['artifact_version']!r}."
        )
    if tuple(payload["feature_names"]) != MODEL_FEATURES:
        raise ModelArtifactError("Model artifact feature order does not match the schema.")
    if tuple(payload["class_labels"]) != (0, 1):
        raise ModelArtifactError("Model artifact must contain class labels 0 and 1.")
    if payload["educational_only"] is not True:
        raise ModelArtifactError("Model artifact must be marked educational_only.")

    model = payload["model"]
    if not isinstance(model, RandomForestClassifier):
        raise ModelArtifactError("Model artifact does not contain a Random Forest.")
    try:
        check_is_fitted(model)
    except NotFittedError as exc:
        raise ModelArtifactError("Model artifact is not fitted.") from exc

    importance_payload = payload["feature_importance"]
    if not isinstance(importance_payload, list):
        raise ModelArtifactError("Model feature importance must be a list.")
    try:
        feature_importance = tuple(
            FeatureImportanceEntry(
                rank=int(entry["rank"]),
                feature_name=str(entry["feature_name"]),
                display_name=str(entry["display_name"]),
                importance=float(entry["importance"]),
            )
            for entry in importance_payload
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactError("Model feature importance is invalid.") from exc
    if {entry.feature_name for entry in feature_importance} != set(MODEL_FEATURES):
        raise ModelArtifactError("Model feature importance does not match the schema.")

    return LoadedModelBundle(
        model=model,
        artifact_version=str(payload["artifact_version"]),
        feature_names=tuple(payload["feature_names"]),
        class_labels=tuple(int(label) for label in payload["class_labels"]),
        prediction_threshold=float(payload["prediction_threshold"]),
        best_tree_count=int(payload["best_tree_count"]),
        feature_importance=feature_importance,
        sklearn_version=str(payload["sklearn_version"]),
        educational_only=bool(payload["educational_only"]),
    )
