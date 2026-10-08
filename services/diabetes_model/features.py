"""Convert validated patient records into model-ready feature matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .schema import FEATURE_COLUMNS, PatientRecord
from .split import PatientSplit, validate_patient_split


MODEL_FEATURES = FEATURE_COLUMNS


class FeaturePreparationError(ValueError):
    """Raised when patient records cannot be converted into model inputs."""


@dataclass(frozen=True)
class FeatureDefinition:
    """Human-readable information for one model input feature."""

    name: str
    display_name: str
    unit: str


FEATURE_DEFINITIONS = (
    FeatureDefinition("age_years", "Age", "years"),
    FeatureDefinition("glucose_mmol_l", "Fasting blood glucose", "mmol/L"),
    FeatureDefinition("hba1c_percent", "HbA1c", "%"),
    FeatureDefinition("bmi", "Body mass index", "kg/m²"),
    FeatureDefinition("systolic_bp_mmhg", "Systolic blood pressure", "mmHg"),
    FeatureDefinition("family_history", "Family history of diabetes", "0 or 1"),
)


@dataclass(frozen=True)
class PreparedDataset:
    """Immutable model inputs for one patient group."""

    patient_ids: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return MODEL_FEATURES

    @property
    def sample_count(self) -> int:
        return len(self.patient_ids)

    @property
    def feature_count(self) -> int:
        return len(MODEL_FEATURES)


@dataclass(frozen=True)
class PreparedModelSplit:
    """Prepared training, validation, and test inputs kept as separate groups."""

    training: PreparedDataset
    validation: PreparedDataset
    test: PreparedDataset


def prepare_dataset(records: Iterable[PatientRecord]) -> PreparedDataset:
    """Create an ordered numeric matrix without fitting data transformations.

    Random Forest models do not require standardisation. The synthetic cohort
    also rejects missing values, so preserving the original units keeps later
    feature-importance explanations easier for students to understand.
    """

    patient_records = tuple(records)
    if not patient_records:
        raise FeaturePreparationError("At least one patient record is required.")

    patient_ids: list[str] = []
    feature_rows: list[tuple[float, ...]] = []
    labels: list[int] = []

    for row_number, record in enumerate(patient_records, start=1):
        if not isinstance(record, PatientRecord):
            raise FeaturePreparationError(
                f"Row {row_number} is not a PatientRecord."
            )
        if (
            type(record.patient_id) is not str
            or not record.patient_id.strip()
            or record.patient_id != record.patient_id.strip()
        ):
            raise FeaturePreparationError(
                f"Row {row_number}: patient_id must be a clean, non-empty string."
            )
        if type(record.diabetes_label) is not int or record.diabetes_label not in {
            0,
            1,
        }:
            raise FeaturePreparationError(
                f"{record.patient_id}: diabetes_label must be integer 0 or 1."
            )

        row: list[float] = []
        for feature_name in MODEL_FEATURES:
            value = getattr(record, feature_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise FeaturePreparationError(
                    f"{record.patient_id}: {feature_name} must be numeric."
                )
            if feature_name == "family_history" and (
                type(value) is not int or value not in {0, 1}
            ):
                raise FeaturePreparationError(
                    f"{record.patient_id}: family_history must be integer 0 or 1."
                )
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise FeaturePreparationError(
                    f"{record.patient_id}: {feature_name} must be finite."
                )
            row.append(numeric_value)

        patient_ids.append(record.patient_id)
        feature_rows.append(tuple(row))
        labels.append(record.diabetes_label)

    if len(patient_ids) != len(set(patient_ids)):
        raise FeaturePreparationError("Prepared patient IDs must be unique.")

    return PreparedDataset(
        patient_ids=tuple(patient_ids),
        features=tuple(feature_rows),
        labels=tuple(labels),
    )


def prepare_model_split(patient_split: PatientSplit) -> PreparedModelSplit:
    """Prepare all groups independently while preserving split isolation."""

    cohort = (
        *patient_split.training,
        *patient_split.validation,
        *patient_split.test,
    )
    validate_patient_split(cohort, patient_split)

    return PreparedModelSplit(
        training=prepare_dataset(patient_split.training),
        validation=prepare_dataset(patient_split.validation),
        test=prepare_dataset(patient_split.test),
    )
