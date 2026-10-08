"""Backend-ready prediction contract for one fictional patient."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .features import MODEL_FEATURES
from .model_artifact import LoadedModelBundle, PROBABILITY_NOTE
from .schema import NUMERIC_RANGES


PREDICTION_INPUT_FIELDS = ("patient_id", *MODEL_FEATURES)


class PredictionInputError(ValueError):
    """Raised when a prediction request does not match the input contract."""


@dataclass(frozen=True)
class PredictionInput:
    """Validated request payload with no diagnosis label."""

    patient_id: str
    age_years: float
    glucose_mmol_l: float
    hba1c_percent: float
    bmi: float
    systolic_bp_mmhg: float
    family_history: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "PredictionInput":
        if not isinstance(payload, Mapping):
            raise PredictionInputError("Prediction input must be a JSON object.")
        supplied_fields = set(payload)
        required_fields = set(PREDICTION_INPUT_FIELDS)
        missing = sorted(required_fields - supplied_fields)
        extra = sorted(supplied_fields - required_fields)
        if missing or extra:
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unexpected: " + ", ".join(extra))
            message = "Invalid prediction fields (" + "; ".join(details) + ")."
            raise PredictionInputError(message)

        patient_id = payload["patient_id"]
        if (
            type(patient_id) is not str
            or not patient_id.strip()
            or patient_id != patient_id.strip()
        ):
            raise PredictionInputError("patient_id must be a clean, non-empty string.")

        values: dict[str, float | int] = {}
        for feature_name in MODEL_FEATURES:
            value = payload[feature_name]
            if feature_name == "family_history":
                if type(value) is not int or value not in {0, 1}:
                    raise PredictionInputError(
                        "family_history must be integer 0 or 1."
                    )
                values[feature_name] = value
                continue

            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PredictionInputError(f"{feature_name} must be numeric.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise PredictionInputError(f"{feature_name} must be finite.")
            lower, upper = NUMERIC_RANGES[feature_name]
            if not lower <= numeric_value <= upper:
                raise PredictionInputError(
                    f"{feature_name} must be within the educational range "
                    f"[{lower}, {upper}]."
                )
            values[feature_name] = numeric_value

        return cls(patient_id=patient_id, **values)

    @property
    def feature_row(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in MODEL_FEATURES)

    def as_dict(self) -> dict[str, str | int | float]:
        return {field: getattr(self, field) for field in PREDICTION_INPUT_FIELDS}


@dataclass(frozen=True)
class InferencePrediction:
    """JSON-ready educational model response."""

    patient_id: str
    predicted_label: int
    estimated_diabetes_probability: float
    prediction_threshold: float

    def as_dict(self) -> dict[str, str | int | float | bool]:
        return {
            "patient_id": self.patient_id,
            "predicted_label": self.predicted_label,
            "estimated_diabetes_probability": round(
                self.estimated_diabetes_probability, 4
            ),
            "prediction_threshold": self.prediction_threshold,
            "educational_only": True,
            "probability_note": PROBABILITY_NOTE,
        }


def predict_patient(
    bundle: LoadedModelBundle,
    payload: Mapping[str, object],
) -> InferencePrediction:
    """Validate one request and return the label and positive-class probability."""

    if not isinstance(bundle, LoadedModelBundle):
        raise PredictionInputError("bundle must be a LoadedModelBundle.")
    prediction_input = PredictionInput.from_mapping(payload)
    if bundle.feature_names != MODEL_FEATURES:
        raise PredictionInputError("Loaded model feature order does not match the schema.")

    rows = [prediction_input.feature_row]
    predicted_label = int(bundle.model.predict(rows)[0])
    class_labels = list(bundle.model.classes_)
    positive_class_index = class_labels.index(1)
    probability = float(bundle.model.predict_proba(rows)[0][positive_class_index])
    return InferencePrediction(
        patient_id=prediction_input.patient_id,
        predicted_label=predicted_label,
        estimated_diabetes_probability=probability,
        prediction_threshold=bundle.prediction_threshold,
    )
