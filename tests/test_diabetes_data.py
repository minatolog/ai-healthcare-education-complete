"""Tests for the fixed synthetic diabetes cohort."""

from __future__ import annotations

import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from services.diabetes_model.generate_synthetic_cohort import (
    DEFAULT_SEED,
    generate_synthetic_cohort,
)
from services.diabetes_model.schema import (
    EXPECTED_PATIENT_COUNT,
    DatasetValidationError,
    load_reference_cohort,
    validate_reference_cohort,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"


class SyntheticDiabetesCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_reference_cohort(DATASET_PATH)

    def test_committed_dataset_has_exactly_40_patients(self) -> None:
        self.assertEqual(len(self.records), EXPECTED_PATIENT_COUNT)

    def test_patient_ids_are_unique(self) -> None:
        patient_ids = [record.patient_id for record in self.records]
        self.assertEqual(len(patient_ids), len(set(patient_ids)))

    def test_labels_are_balanced(self) -> None:
        counts = Counter(record.diabetes_label for record in self.records)
        self.assertEqual(counts, Counter({0: 20, 1: 20}))

    def test_committed_csv_matches_seeded_generator(self) -> None:
        generated = generate_synthetic_cohort(DEFAULT_SEED)
        self.assertEqual(self.records, generated)

    def test_no_single_numeric_feature_almost_perfectly_separates_labels(self) -> None:
        for field_name in (
            "age_years",
            "glucose_mmol_l",
            "hba1c_percent",
            "bmi",
            "systolic_bp_mmhg",
        ):
            values = sorted({getattr(record, field_name) for record in self.records})
            thresholds = [
                (lower + upper) / 2
                for lower, upper in zip(values, values[1:])
            ]
            best_correct = max(
                sum(
                    int((getattr(record, field_name) >= threshold) == direction)
                    == record.diabetes_label
                    for record in self.records
                )
                for threshold in thresholds
                for direction in (False, True)
            )
            with self.subTest(field=field_name):
                self.assertLess(best_correct / len(self.records), 0.90)

    def test_validator_rejects_duplicate_patient_id(self) -> None:
        invalid_records = list(self.records)
        invalid_records[1] = replace(
            invalid_records[1], patient_id=invalid_records[0].patient_id
        )

        with self.assertRaisesRegex(DatasetValidationError, "must be unique"):
            validate_reference_cohort(invalid_records)

    def test_validator_rejects_patient_id_with_surrounding_whitespace(self) -> None:
        invalid_records = list(self.records)
        invalid_records[0] = replace(
            invalid_records[0], patient_id=f" {invalid_records[0].patient_id} "
        )

        with self.assertRaisesRegex(DatasetValidationError, "leading or trailing"):
            validate_reference_cohort(invalid_records)

    def test_validator_rejects_out_of_range_value(self) -> None:
        invalid_records = list(self.records)
        invalid_records[0] = replace(invalid_records[0], glucose_mmol_l=99.0)

        with self.assertRaisesRegex(DatasetValidationError, "educational range"):
            validate_reference_cohort(invalid_records)

    def test_validator_rejects_boolean_as_binary_integer(self) -> None:
        invalid_records = list(self.records)
        invalid_records[0] = replace(invalid_records[0], family_history=True)

        with self.assertRaisesRegex(DatasetValidationError, "integer 0 or 1"):
            validate_reference_cohort(invalid_records)

    def test_loader_rejects_extra_csv_value(self) -> None:
        csv_lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()
        csv_lines[1] += ",unexpected"

        with TemporaryDirectory() as temp_directory:
            malformed_path = Path(temp_directory) / "malformed.csv"
            malformed_path.write_text("\n".join(csv_lines), encoding="utf-8")

            with self.assertRaisesRegex(DatasetValidationError, "required columns"):
                load_reference_cohort(malformed_path)


if __name__ == "__main__":
    unittest.main()
