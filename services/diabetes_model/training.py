"""Incremental Random Forest training with validation accuracy history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .features import MODEL_FEATURES, PreparedDataset


DEFAULT_TREE_COUNTS = tuple(range(5, 101, 5))


class ModelTrainingError(ValueError):
    """Raised when model inputs or training configuration are invalid."""


@dataclass(frozen=True)
class RandomForestTrainingConfig:
    """Small, deterministic Random Forest settings for the learning activity."""

    tree_counts: tuple[int, ...] = DEFAULT_TREE_COUNTS
    max_depth: int = 3
    min_samples_leaf: int = 2
    class_weight: str = "balanced"
    random_state: int = 9900

    def validate(self) -> None:
        if not self.tree_counts:
            raise ModelTrainingError("tree_counts must not be empty.")
        if any(type(count) is not int or count <= 0 for count in self.tree_counts):
            raise ModelTrainingError("Every tree count must be a positive integer.")
        if tuple(sorted(set(self.tree_counts))) != self.tree_counts:
            raise ModelTrainingError(
                "tree_counts must be unique and in strictly increasing order."
            )
        if type(self.max_depth) is not int or self.max_depth <= 0:
            raise ModelTrainingError("max_depth must be a positive integer.")
        if type(self.min_samples_leaf) is not int or self.min_samples_leaf <= 0:
            raise ModelTrainingError(
                "min_samples_leaf must be a positive integer."
            )
        if type(self.random_state) is not int:
            raise ModelTrainingError("random_state must be an integer.")
        if self.class_weight != "balanced":
            raise ModelTrainingError(
                "class_weight must remain 'balanced' for this learning activity."
            )


@dataclass(frozen=True)
class TrainingHistoryPoint:
    """Accuracy recorded after a cumulative number of trees has been built."""

    trees_built: int
    training_accuracy: float
    validation_accuracy: float


@dataclass(frozen=True)
class RandomForestTrainingResult:
    """Selected model and graph-ready training history."""

    model: RandomForestClassifier
    history: tuple[TrainingHistoryPoint, ...]
    best_tree_count: int
    best_validation_accuracy: float
    config: RandomForestTrainingConfig

    def history_as_dicts(self) -> list[dict[str, int | float | None]]:
        """Return JSON-ready points for the website's training curve."""

        return [
            {
                "step": step,
                "trees_built": point.trees_built,
                "train_accuracy": round(point.training_accuracy, 4),
                "validation_accuracy": round(point.validation_accuracy, 4),
                "loss": None,
            }
            for step, point in enumerate(self.history, start=1)
        ]

    def summary_as_dict(self) -> dict[str, object]:
        """Return model-selection details without serialising the estimator."""

        return {
            "algorithm": "RandomForestClassifier",
            "feature_names": list(MODEL_FEATURES),
            "best_tree_count": self.best_tree_count,
            "best_validation_accuracy": round(self.best_validation_accuracy, 4),
            "training_history": self.history_as_dicts(),
        }


def train_random_forest(
    training: PreparedDataset,
    validation: PreparedDataset,
    config: RandomForestTrainingConfig | None = None,
) -> RandomForestTrainingResult:
    """Train on 20 records and use validation accuracy to select tree count.

    No test dataset is accepted by this function, which prevents accidental use
    of held-out test patients during model fitting or model selection.
    """

    training_config = config or RandomForestTrainingConfig()
    training_config.validate()
    _validate_model_dataset("training", training)
    _validate_model_dataset("validation", validation)

    if set(training.patient_ids) & set(validation.patient_ids):
        raise ModelTrainingError("Training and validation patients must not overlap.")

    class_weights = _calculate_balanced_class_weights(training.labels)
    progress_model = _build_classifier(
        training_config,
        n_estimators=training_config.tree_counts[0],
        warm_start=True,
        class_weights=class_weights,
    )
    history: list[TrainingHistoryPoint] = []

    for tree_count in training_config.tree_counts:
        progress_model.set_params(n_estimators=tree_count)
        progress_model.fit(training.features, training.labels)

        training_predictions = progress_model.predict(training.features)
        validation_predictions = progress_model.predict(validation.features)
        history.append(
            TrainingHistoryPoint(
                trees_built=tree_count,
                training_accuracy=float(
                    accuracy_score(training.labels, training_predictions)
                ),
                validation_accuracy=float(
                    accuracy_score(validation.labels, validation_predictions)
                ),
            )
        )

    best_point = max(
        history,
        key=lambda point: (point.validation_accuracy, -point.trees_built),
    )
    selected_model = _build_classifier(
        training_config,
        n_estimators=best_point.trees_built,
        warm_start=False,
        class_weights=class_weights,
    )
    selected_model.fit(training.features, training.labels)

    return RandomForestTrainingResult(
        model=selected_model,
        history=tuple(history),
        best_tree_count=best_point.trees_built,
        best_validation_accuracy=best_point.validation_accuracy,
        config=training_config,
    )


def _build_classifier(
    config: RandomForestTrainingConfig,
    *,
    n_estimators: int,
    warm_start: bool,
    class_weights: dict[int, float],
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features="sqrt",
        class_weight=class_weights,
        random_state=config.random_state,
        warm_start=warm_start,
        n_jobs=1,
    )


def _calculate_balanced_class_weights(labels: tuple[int, ...]) -> dict[int, float]:
    """Calculate fixed weights once so warm-start fits reuse identical values."""

    counts = Counter(labels)
    sample_count = len(labels)
    class_count = len(counts)
    return {
        label: sample_count / (class_count * count)
        for label, count in sorted(counts.items())
    }


def _validate_model_dataset(name: str, dataset: PreparedDataset) -> None:
    if not isinstance(dataset, PreparedDataset):
        raise ModelTrainingError(f"{name} must be a PreparedDataset.")
    if dataset.sample_count == 0:
        raise ModelTrainingError(f"{name} must not be empty.")
    if len(dataset.features) != len(dataset.labels):
        raise ModelTrainingError(f"{name} features and labels must have equal rows.")
    if any(len(row) != len(MODEL_FEATURES) for row in dataset.features):
        raise ModelTrainingError(
            f"{name} rows must contain exactly {len(MODEL_FEATURES)} features."
        )
    if set(dataset.labels) != {0, 1}:
        raise ModelTrainingError(f"{name} must contain both class labels.")
