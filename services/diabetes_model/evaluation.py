"""Final held-out test evaluation for the educational diabetes model."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.validation import check_is_fitted

from .features import MODEL_FEATURES, PreparedDataset
from .split import TEST_SIZE
from .training import RandomForestTrainingResult


class ModelEvaluationError(ValueError):
    """Raised when a held-out test evaluation cannot be performed safely."""


@dataclass(frozen=True)
class TestPrediction:
    """Prediction and model-estimated probability for one test patient."""

    patient_id: str
    actual_label: int
    predicted_label: int
    estimated_diabetes_probability: float

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "patient_id": self.patient_id,
            "actual_label": self.actual_label,
            "predicted_label": self.predicted_label,
            "estimated_diabetes_probability": round(
                self.estimated_diabetes_probability, 4
            ),
        }


@dataclass(frozen=True)
class ClassificationMetrics:
    """Metrics calculated once on the locked ten-patient test group."""

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]

    def as_dict(self) -> dict[str, object]:
        return {
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


@dataclass(frozen=True)
class FinalEvaluationResult:
    """Graph/API-ready result from the held-out test group."""

    metrics: ClassificationMetrics
    predictions: tuple[TestPrediction, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluation_scope": "held_out_test",
            "test_sample_count": len(self.predictions),
            "metrics": self.metrics.as_dict(),
            "predictions": [prediction.as_dict() for prediction in self.predictions],
            "probability_note": (
                "Model-estimated educational score; not a calibrated clinical risk."
            ),
        }


def evaluate_test_set(
    training_result: RandomForestTrainingResult,
    test: PreparedDataset,
) -> FinalEvaluationResult:
    """Evaluate the selected model on patients unseen during training/selection."""

    if not isinstance(training_result, RandomForestTrainingResult):
        raise ModelEvaluationError(
            "training_result must be a RandomForestTrainingResult."
        )
    _validate_test_dataset(test)

    seen_patient_ids = set(training_result.training_patient_ids) | set(
        training_result.validation_patient_ids
    )
    overlapping_ids = sorted(seen_patient_ids & set(test.patient_ids))
    if overlapping_ids:
        raise ModelEvaluationError(
            "Test patients must be unseen; overlapping IDs: "
            + ", ".join(overlapping_ids)
        )

    try:
        check_is_fitted(training_result.model)
    except NotFittedError as exc:
        raise ModelEvaluationError("The selected model has not been fitted.") from exc

    model_classes = list(training_result.model.classes_)
    if 1 not in model_classes:
        raise ModelEvaluationError("The fitted model has no positive class label.")
    positive_class_index = model_classes.index(1)

    predicted_labels = training_result.model.predict(test.features)
    predicted_probabilities = training_result.model.predict_proba(test.features)
    positive_probabilities = predicted_probabilities[:, positive_class_index]

    matrix = confusion_matrix(test.labels, predicted_labels, labels=[0, 1])
    metrics = ClassificationMetrics(
        accuracy=float(accuracy_score(test.labels, predicted_labels)),
        balanced_accuracy=float(
            balanced_accuracy_score(test.labels, predicted_labels)
        ),
        precision=float(
            precision_score(test.labels, predicted_labels, zero_division=0)
        ),
        recall=float(recall_score(test.labels, predicted_labels, zero_division=0)),
        f1=float(f1_score(test.labels, predicted_labels, zero_division=0)),
        confusion_matrix=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
    )
    predictions = tuple(
        TestPrediction(
            patient_id=patient_id,
            actual_label=int(actual_label),
            predicted_label=int(predicted_label),
            estimated_diabetes_probability=float(probability),
        )
        for patient_id, actual_label, predicted_label, probability in zip(
            test.patient_ids,
            test.labels,
            predicted_labels,
            positive_probabilities,
        )
    )
    return FinalEvaluationResult(metrics=metrics, predictions=predictions)


def _validate_test_dataset(test: PreparedDataset) -> None:
    if not isinstance(test, PreparedDataset):
        raise ModelEvaluationError("test must be a PreparedDataset.")
    if test.sample_count != TEST_SIZE:
        raise ModelEvaluationError(
            f"The held-out test group must contain {TEST_SIZE} patients."
        )
    if len(test.patient_ids) != len(set(test.patient_ids)):
        raise ModelEvaluationError("Test patient IDs must be unique.")
    if not (
        len(test.patient_ids) == len(test.features) == len(test.labels)
    ):
        raise ModelEvaluationError(
            "Test patient IDs, features, and labels must have equal rows."
        )
    if any(len(row) != len(MODEL_FEATURES) for row in test.features):
        raise ModelEvaluationError(
            f"Test rows must contain exactly {len(MODEL_FEATURES)} features."
        )
    if set(test.labels) != {0, 1}:
        raise ModelEvaluationError("The test group must contain both class labels.")
