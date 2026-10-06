"""Schema and validation rules for the synthetic diabetes cohort.

The ranges in this module are broad educational data guardrails. They are not
diagnostic thresholds and must not be used for clinical decision-making.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


FEATURE_COLUMNS = (
    "age_years",
    "glucose_mmol_l",
    "hba1c_percent",
    "bmi",
    "systolic_bp_mmhg",
    "family_history",
)
TARGET_COLUMN = "diabetes_label"
CSV_COLUMNS = ("patient_id", *FEATURE_COLUMNS, TARGET_COLUMN)

EXPECTED_PATIENT_COUNT = 40
EXPECTED_CLASS_COUNTS = {0: 20, 1: 20}

# These deliberately broad bounds catch impossible or mistyped synthetic values
# without presenting the validator as a medical diagnostic rule.
NUMERIC_RANGES = {
    "age_years": (18, 85),
    "glucose_mmol_l": (3.5, 16.0),
    "hba1c_percent": (4.0, 12.0),
    "bmi": (16.0, 50.0),
    "systolic_bp_mmhg": (85, 190),
}


class DatasetValidationError(ValueError):
    """Raised when the reference cohort violates its data contract."""


@dataclass(frozen=True)
class PatientRecord:
    """One fictional patient used in the diabetes learning activity."""

    patient_id: str
    age_years: int
    glucose_mmol_l: float
    hba1c_percent: float
    bmi: float
    systolic_bp_mmhg: int
    family_history: int
    diabetes_label: int

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "PatientRecord":
        """Parse one CSV row into a typed record."""

        if set(row) != set(CSV_COLUMNS) or any(
            row.get(column) is None for column in CSV_COLUMNS
        ):
            raise DatasetValidationError(
                f"CSV row does not match the required columns: {dict(row)!r}"
            )

        try:
            return cls(
                patient_id=row["patient_id"].strip(),
                age_years=int(row["age_years"]),
                glucose_mmol_l=float(row["glucose_mmol_l"]),
                hba1c_percent=float(row["hba1c_percent"]),
                bmi=float(row["bmi"]),
                systolic_bp_mmhg=int(row["systolic_bp_mmhg"]),
                family_history=int(row["family_history"]),
                diabetes_label=int(row[TARGET_COLUMN]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetValidationError(f"Invalid CSV row: {dict(row)!r}") from exc

    def to_csv_row(self) -> dict[str, str | int | float]:
        """Return values in the stable column order used by the CSV file."""

        return {
            "patient_id": self.patient_id,
            "age_years": self.age_years,
            "glucose_mmol_l": self.glucose_mmol_l,
            "hba1c_percent": self.hba1c_percent,
            "bmi": self.bmi,
            "systolic_bp_mmhg": self.systolic_bp_mmhg,
            "family_history": self.family_history,
            TARGET_COLUMN: self.diabetes_label,
        }


def load_reference_cohort(csv_path: str | Path) -> list[PatientRecord]:
    """Load and validate the committed 40-patient reference cohort."""

    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise DatasetValidationError(
                f"Expected columns {list(CSV_COLUMNS)!r}, got {reader.fieldnames!r}."
            )
        records = [PatientRecord.from_csv_row(row) for row in reader]

    validate_reference_cohort(records)
    return records


def validate_reference_cohort(records: Iterable[PatientRecord]) -> None:
    """Validate the fixed cohort before it is used by later model code."""

    cohort = list(records)
    errors: list[str] = []

    if len(cohort) != EXPECTED_PATIENT_COUNT:
        errors.append(
            f"Expected {EXPECTED_PATIENT_COUNT} patients, found {len(cohort)}."
        )

    patient_ids: list[str] = []
    valid_labels: list[int] = []

    for row_number, record in enumerate(cohort, start=1):
        if not isinstance(record, PatientRecord):
            errors.append(f"Row {row_number} is not a PatientRecord.")
            continue

        if type(record.patient_id) is not str or not record.patient_id.strip():
            errors.append(f"Row {row_number}: patient_id must be a non-empty string.")
        elif record.patient_id != record.patient_id.strip():
            errors.append(
                f"Row {row_number}: patient_id must not have leading or trailing whitespace."
            )
        else:
            patient_ids.append(record.patient_id)

        for field_name in ("age_years", "systolic_bp_mmhg"):
            if type(getattr(record, field_name)) is not int:
                errors.append(
                    f"{record.patient_id}: {field_name} must be an integer."
                )

        for field_name in ("glucose_mmol_l", "hba1c_percent", "bmi"):
            value = getattr(record, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(
                    f"{record.patient_id}: {field_name} must be a finite number."
                )
            elif not math.isfinite(value):
                errors.append(
                    f"{record.patient_id}: {field_name} must be a finite number."
                )

        if type(record.family_history) is not int or record.family_history not in {0, 1}:
            errors.append(
                f"{record.patient_id}: family_history must be integer 0 or 1."
            )

        if type(record.diabetes_label) is not int:
            errors.append(
                f"{record.patient_id}: diabetes_label must be integer 0 or 1."
            )
        else:
            valid_labels.append(record.diabetes_label)

    if len(set(patient_ids)) != len(patient_ids):
        errors.append("patient_id values must be unique.")

    label_counts = Counter(valid_labels)
    if set(label_counts) - {0, 1}:
        errors.append("diabetes_label must contain only 0 or 1.")
    if dict(label_counts) != EXPECTED_CLASS_COUNTS:
        errors.append(
            "The reference cohort must contain 20 label-0 and 20 label-1 patients; "
            f"found {dict(label_counts)}."
        )

    for record in cohort:
        if not isinstance(record, PatientRecord):
            continue
        for field_name, (lower, upper) in NUMERIC_RANGES.items():
            value = getattr(record, field_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                continue
            if not lower <= value <= upper:
                errors.append(
                    f"{record.patient_id}: {field_name}={value} is outside "
                    f"the educational range [{lower}, {upper}]."
                )

    if errors:
        raise DatasetValidationError("\n".join(errors))
