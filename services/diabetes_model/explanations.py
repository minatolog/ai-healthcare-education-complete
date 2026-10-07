"""Local one-feature-at-a-time sensitivity explanations."""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .features import FEATURE_DEFINITIONS, MODEL_FEATURES, PreparedDataset
from .training import RandomForestTrainingResult


LOCAL_EXPLANATION_METHOD = "one_feature_at_a_time_training_reference"
LOCAL_EXPLANATION_NOTE = (
    "Model sensitivity relative to training-set reference values; effects are "
    "not additive and do not prove causation or clinical importance."
)


class LocalExplanationError(ValueError):
    """Raised when a local sensitivity explanation cannot be calculated."""


@dataclass(frozen=True)
class LocalFeatureEffect:
    """Probability change associated with one patient's original feature value."""

    rank: int
    feature_name: str
    display_name: str
    patient_value: float
    reference_value: float
    probability_change: float
    direction: str

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "rank": self.rank,
            "feature_name": self.feature_name,
            "display_name": self.display_name,
            "patient_value": round(self.patient_value, 4),
            "reference_value": round(self.reference_value, 4),
            "probability_change": round(self.probability_change, 6),
            "direction": self.direction,
        }


@dataclass(frozen=True)
class PatientLocalExplanation:
    """Local model sensitivity for one fictional patient."""

    patient_id: str
    predicted_label: int
    estimated_diabetes_probability: float
    feature_effects: tuple[LocalFeatureEffect, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "patient_id": self.patient_id,
            "predicted_label": self.predicted_label,
            "estimated_diabetes_probability": round(
                self.estimated_diabetes_probability, 4
            ),
            "method": LOCAL_EXPLANATION_METHOD,
            "feature_effects": [
                effect.as_dict() for effect in self.feature_effects
            ],
            "note": LOCAL_EXPLANATION_NOTE,
        }


def explain_dataset_predictions(
    training_result: RandomForestTrainingResult,
    training: PreparedDataset,
    patients: PreparedDataset,
) -> tuple[PatientLocalExplanation, ...]:
    """Explain predictions using medians/mode calculated from training only."""

    if not isinstance(training_result, RandomForestTrainingResult):
        raise LocalExplanationError(
            "training_result must be a RandomForestTrainingResult."
        )
    if not isinstance(training, PreparedDataset):
        raise LocalExplanationError("training must be a PreparedDataset.")
    if not isinstance(patients, PreparedDataset):
        raise LocalExplanationError("patients must be a PreparedDataset.")
    if training.patient_ids != training_result.training_patient_ids:
        raise LocalExplanationError(
            "Training data must match the records used to fit the model."
        )
    if training.sample_count == 0 or patients.sample_count == 0:
        raise LocalExplanationError("Training and patient datasets must not be empty.")
    if any(len(row) != len(MODEL_FEATURES) for row in patients.features):
        raise LocalExplanationError("Patient feature rows do not match the schema.")

    try:
        check_is_fitted(training_result.model)
    except NotFittedError as exc:
        raise LocalExplanationError("The selected model has not been fitted.") from exc

    class_labels = list(training_result.model.classes_)
    if 1 not in class_labels:
        raise LocalExplanationError("The fitted model has no positive class label.")
    positive_class_index = class_labels.index(1)
    reference_values = _training_reference_values(training)
    predicted_labels = training_result.model.predict(patients.features)
    probabilities = training_result.model.predict_proba(patients.features)

    explanations: list[PatientLocalExplanation] = []
    for patient_id, row, predicted_label, probability_row in zip(
        patients.patient_ids,
        patients.features,
        predicted_labels,
        probabilities,
    ):
        baseline_probability = float(probability_row[positive_class_index])
        raw_effects: list[tuple[int, float]] = []
        for feature_index, reference_value in enumerate(reference_values):
            modified_row = list(row)
            modified_row[feature_index] = reference_value
            reference_probability = float(
                training_result.model.predict_proba([modified_row])[0][
                    positive_class_index
                ]
            )
            raw_effects.append(
                (feature_index, baseline_probability - reference_probability)
            )

        ranked_effects = sorted(
            raw_effects,
            key=lambda item: (-abs(item[1]), item[0]),
        )
        effects = tuple(
            LocalFeatureEffect(
                rank=rank,
                feature_name=FEATURE_DEFINITIONS[feature_index].name,
                display_name=FEATURE_DEFINITIONS[feature_index].display_name,
                patient_value=float(row[feature_index]),
                reference_value=float(reference_values[feature_index]),
                probability_change=float(probability_change),
                direction=_effect_direction(probability_change),
            )
            for rank, (feature_index, probability_change) in enumerate(
                ranked_effects, start=1
            )
        )
        explanations.append(
            PatientLocalExplanation(
                patient_id=patient_id,
                predicted_label=int(predicted_label),
                estimated_diabetes_probability=baseline_probability,
                feature_effects=effects,
            )
        )

    return tuple(explanations)


def _training_reference_values(training: PreparedDataset) -> tuple[float, ...]:
    columns = tuple(zip(*training.features))
    references: list[float] = []
    for feature_name, values in zip(MODEL_FEATURES, columns):
        if feature_name == "family_history":
            counts = Counter(int(value) for value in values)
            reference = min(
                counts,
                key=lambda value: (-counts[value], value),
            )
            references.append(float(reference))
        else:
            references.append(float(statistics.median(values)))
    return tuple(references)


def _effect_direction(probability_change: float) -> str:
    if probability_change > 1e-12:
        return "increased_score"
    if probability_change < -1e-12:
        return "decreased_score"
    return "no_change"
