"""Tests for model-ready diabetes feature preparation."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from services.diabetes_model.features import (
    FEATURE_DEFINITIONS,
    MODEL_FEATURES,
    FeaturePreparationError,
    prepare_dataset,
    prepare_model_split,
)
from services.diabetes_model.schema import FEATURE_COLUMNS, load_reference_cohort
from services.diabetes_model.split import create_student_split


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"


class DiabetesFeaturePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_reference_cohort(DATASET_PATH)
        label_zero_ids = [
            record.patient_id for record in cls.records if record.diabetes_label == 0
        ][:10]
        label_one_ids = [
            record.patient_id for record in cls.records if record.diabetes_label == 1
        ][:10]
        cls.patient_split = create_student_split(
            cls.records, label_zero_ids + label_one_ids
        )
        cls.prepared = prepare_model_split(cls.patient_split)

    def test_feature_order_matches_the_data_contract(self) -> None:
        self.assertEqual(MODEL_FEATURES, FEATURE_COLUMNS)
        self.assertNotIn("diabetes_label", MODEL_FEATURES)
        self.assertEqual(
            tuple(definition.name for definition in FEATURE_DEFINITIONS),
            FEATURE_COLUMNS,
        )

    def test_prepared_groups_have_expected_shapes(self) -> None:
        self.assertEqual(self.prepared.training.sample_count, 20)
        self.assertEqual(self.prepared.validation.sample_count, 10)
        self.assertEqual(self.prepared.test.sample_count, 10)

        for group in (
            self.prepared.training,
            self.prepared.validation,
            self.prepared.test,
        ):
            self.assertEqual(group.feature_count, 6)
            self.assertTrue(all(len(row) == 6 for row in group.features))
            self.assertEqual(len(group.patient_ids), len(group.labels))
            self.assertEqual(len(group.patient_ids), len(group.features))

    def test_feature_values_preserve_original_clinical_units(self) -> None:
        record = self.patient_split.training[0]
        expected_row = tuple(float(getattr(record, name)) for name in MODEL_FEATURES)
        self.assertEqual(self.prepared.training.features[0], expected_row)

    def test_patient_feature_and_label_rows_remain_aligned(self) -> None:
        records_by_id = {record.patient_id: record for record in self.records}

        for group in (
            self.prepared.training,
            self.prepared.validation,
            self.prepared.test,
        ):
            for patient_id, feature_row, label in zip(
                group.patient_ids, group.features, group.labels
            ):
                record = records_by_id[patient_id]
                self.assertEqual(label, record.diabetes_label)
                self.assertEqual(
                    feature_row,
                    tuple(float(getattr(record, name)) for name in MODEL_FEATURES),
                )

    def test_prepared_groups_remain_isolated(self) -> None:
        training_ids = set(self.prepared.training.patient_ids)
        validation_ids = set(self.prepared.validation.patient_ids)
        test_ids = set(self.prepared.test.patient_ids)

        self.assertTrue(training_ids.isdisjoint(validation_ids))
        self.assertTrue(training_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))

    def test_empty_group_is_rejected(self) -> None:
        with self.assertRaisesRegex(FeaturePreparationError, "At least one"):
            prepare_dataset([])

    def test_duplicate_patient_is_rejected(self) -> None:
        duplicate_records = [self.records[0], self.records[0]]
        with self.assertRaisesRegex(FeaturePreparationError, "must be unique"):
            prepare_dataset(duplicate_records)

    def test_non_finite_feature_is_rejected(self) -> None:
        invalid_record = replace(self.records[0], glucose_mmol_l=float("nan"))
        with self.assertRaisesRegex(FeaturePreparationError, "must be finite"):
            prepare_dataset([invalid_record])

    def test_non_integer_label_is_rejected(self) -> None:
        invalid_record = replace(self.records[0], diabetes_label=1.0)
        with self.assertRaisesRegex(FeaturePreparationError, "integer 0 or 1"):
            prepare_dataset([invalid_record])


if __name__ == "__main__":
    unittest.main()
