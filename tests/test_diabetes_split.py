"""Tests for the student-selected 20/10/10 patient split."""

from __future__ import annotations

import unittest
from pathlib import Path

from services.diabetes_model.schema import load_reference_cohort
from services.diabetes_model.split import (
    SplitValidationError,
    assess_training_selection,
    create_student_split,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"


class DiabetesSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_reference_cohort(DATASET_PATH)
        label_zero_ids = [
            record.patient_id for record in cls.records if record.diabetes_label == 0
        ]
        label_one_ids = [
            record.patient_id for record in cls.records if record.diabetes_label == 1
        ]
        cls.training_ids = label_zero_ids[:10] + label_one_ids[:10]

    def test_split_has_required_sizes_and_complete_coverage(self) -> None:
        patient_split = create_student_split(self.records, self.training_ids)
        split_ids = patient_split.as_id_dict()

        self.assertEqual(len(split_ids["training"]), 20)
        self.assertEqual(len(split_ids["validation"]), 10)
        self.assertEqual(len(split_ids["test"]), 10)
        self.assertEqual(
            set().union(*(set(patient_ids) for patient_ids in split_ids.values())),
            {record.patient_id for record in self.records},
        )

    def test_student_selection_is_used_as_training_group(self) -> None:
        patient_split = create_student_split(self.records, reversed(self.training_ids))
        self.assertEqual(
            set(patient_split.as_id_dict()["training"]), set(self.training_ids)
        )

    def test_groups_do_not_overlap(self) -> None:
        split_ids = create_student_split(
            self.records, self.training_ids
        ).as_id_dict()
        training_ids = set(split_ids["training"])
        validation_ids = set(split_ids["validation"])
        test_ids = set(split_ids["test"])

        self.assertTrue(training_ids.isdisjoint(validation_ids))
        self.assertTrue(training_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))

    def test_balanced_selection_produces_balanced_groups(self) -> None:
        class_counts = create_student_split(
            self.records, self.training_ids
        ).class_counts()
        self.assertEqual(class_counts["training"], {0: 10, 1: 10})
        self.assertEqual(class_counts["validation"], {0: 5, 1: 5})
        self.assertEqual(class_counts["test"], {0: 5, 1: 5})

    def test_valid_imbalanced_selection_keeps_both_labels_in_evaluation(self) -> None:
        label_zero_ids = [
            record.patient_id for record in self.records if record.diabetes_label == 0
        ]
        label_one_ids = [
            record.patient_id for record in self.records if record.diabetes_label == 1
        ]
        imbalanced_selection = label_zero_ids[:18] + label_one_ids[:2]

        class_counts = create_student_split(
            self.records, imbalanced_selection
        ).class_counts()
        self.assertEqual(class_counts["training"], {0: 18, 1: 2})
        self.assertEqual(class_counts["validation"], {0: 1, 1: 9})
        self.assertEqual(class_counts["test"], {0: 1, 1: 9})

    def test_same_seed_produces_same_split(self) -> None:
        first = create_student_split(self.records, self.training_ids, seed=123)
        second = create_student_split(self.records, self.training_ids, seed=123)
        self.assertEqual(first, second)

    def test_balanced_selection_assessment_is_ready(self) -> None:
        assessment = assess_training_selection(
            self.records, self.training_ids
        )
        self.assertTrue(assessment.ready)
        self.assertEqual(assessment.selected_count, 20)
        self.assertEqual(assessment.unique_count, 20)
        self.assertEqual(assessment.training_label_0_count, 10)
        self.assertEqual(assessment.training_label_1_count, 10)
        self.assertEqual(assessment.issues, ())

    def test_valid_imbalanced_selection_returns_warning(self) -> None:
        label_zero_ids = [
            record.patient_id for record in self.records if record.diabetes_label == 0
        ]
        label_one_ids = [
            record.patient_id for record in self.records if record.diabetes_label == 1
        ]
        assessment = assess_training_selection(
            self.records,
            label_zero_ids[:18] + label_one_ids[:2],
        )
        self.assertTrue(assessment.ready)
        self.assertEqual(len(assessment.issues), 1)
        self.assertEqual(
            assessment.issues[0].code, "TRAINING_CLASS_IMBALANCE"
        )
        self.assertEqual(assessment.issues[0].severity, "warning")

    def test_invalid_assessment_has_stable_error_code(self) -> None:
        assessment = assess_training_selection(
            self.records, self.training_ids[:-1]
        )
        self.assertFalse(assessment.ready)
        self.assertEqual(assessment.issues[0].code, "INCORRECT_COUNT")
        self.assertEqual(assessment.issues[0].severity, "error")
        self.assertEqual(assessment.as_dict()["training_class_counts"], None)

    def test_wrong_training_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(SplitValidationError, "Exactly 20"):
            create_student_split(self.records, self.training_ids[:-1])

    def test_duplicate_training_id_is_rejected(self) -> None:
        duplicate_ids = self.training_ids[:-1] + [self.training_ids[0]]
        with self.assertRaisesRegex(SplitValidationError, "must be unique"):
            create_student_split(self.records, duplicate_ids)

    def test_unknown_training_id_is_rejected(self) -> None:
        unknown_ids = self.training_ids[:-1] + ["P999"]
        with self.assertRaisesRegex(SplitValidationError, "Unknown"):
            create_student_split(self.records, unknown_ids)

    def test_selection_that_breaks_evaluation_groups_is_rejected(self) -> None:
        label_zero_ids = [
            record.patient_id for record in self.records if record.diabetes_label == 0
        ]
        label_one_ids = [
            record.patient_id for record in self.records if record.diabetes_label == 1
        ]
        invalid_selection = label_zero_ids[:19] + label_one_ids[:1]

        with self.assertRaisesRegex(SplitValidationError, "at least two"):
            create_student_split(self.records, invalid_selection)


if __name__ == "__main__":
    unittest.main()
