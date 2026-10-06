"""Tests for incremental Random Forest training and history output."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from services.diabetes_model.features import prepare_model_split
from services.diabetes_model.schema import load_reference_cohort
from services.diabetes_model.split import create_student_split
from services.diabetes_model.training import (
    DEFAULT_TREE_COUNTS,
    ModelTrainingError,
    RandomForestTrainingConfig,
    train_random_forest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"


class DiabetesRandomForestTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = load_reference_cohort(DATASET_PATH)
        label_zero_ids = [
            record.patient_id for record in records if record.diabetes_label == 0
        ][:10]
        label_one_ids = [
            record.patient_id for record in records if record.diabetes_label == 1
        ][:10]
        patient_split = create_student_split(
            records, label_zero_ids + label_one_ids
        )
        cls.prepared = prepare_model_split(patient_split)
        cls.result = train_random_forest(
            cls.prepared.training, cls.prepared.validation
        )

    def test_history_contains_every_tree_checkpoint(self) -> None:
        self.assertEqual(
            tuple(point.trees_built for point in self.result.history),
            DEFAULT_TREE_COUNTS,
        )

    def test_history_scores_are_valid_accuracies(self) -> None:
        for point in self.result.history:
            self.assertGreaterEqual(point.training_accuracy, 0.0)
            self.assertLessEqual(point.training_accuracy, 1.0)
            self.assertGreaterEqual(point.validation_accuracy, 0.0)
            self.assertLessEqual(point.validation_accuracy, 1.0)

    def test_selected_model_matches_best_validation_checkpoint(self) -> None:
        expected = max(
            self.result.history,
            key=lambda point: (point.validation_accuracy, -point.trees_built),
        )
        self.assertEqual(self.result.best_tree_count, expected.trees_built)
        self.assertEqual(
            self.result.best_validation_accuracy, expected.validation_accuracy
        )
        self.assertEqual(
            self.result.model.n_estimators, self.result.best_tree_count
        )
        self.assertEqual(self.result.model.class_weight, {0: 1.0, 1: 1.0})

    def test_training_is_reproducible_with_fixed_seed(self) -> None:
        repeated = train_random_forest(
            self.prepared.training, self.prepared.validation
        )
        self.assertEqual(repeated.history, self.result.history)
        self.assertEqual(
            repeated.model.predict(self.prepared.validation.features).tolist(),
            self.result.model.predict(self.prepared.validation.features).tolist(),
        )

    def test_history_is_ready_for_json_curve_output(self) -> None:
        history = self.result.history_as_dicts()
        self.assertEqual(len(history), len(DEFAULT_TREE_COUNTS))
        self.assertEqual(history[0]["step"], 1)
        self.assertEqual(history[0]["trees_built"], DEFAULT_TREE_COUNTS[0])
        self.assertIsNone(history[0]["loss"])
        self.assertIn("train_accuracy", history[0])
        self.assertIn("validation_accuracy", history[0])

    def test_test_dataset_is_not_needed_for_training(self) -> None:
        test_ids = set(self.prepared.test.patient_ids)
        used_ids = set(self.prepared.training.patient_ids) | set(
            self.prepared.validation.patient_ids
        )
        self.assertTrue(test_ids.isdisjoint(used_ids))

    def test_overlapping_training_and_validation_is_rejected(self) -> None:
        overlapping_validation = replace(
            self.prepared.validation,
            patient_ids=self.prepared.training.patient_ids[:10],
        )
        with self.assertRaisesRegex(ModelTrainingError, "must not overlap"):
            train_random_forest(self.prepared.training, overlapping_validation)

    def test_invalid_tree_checkpoints_are_rejected(self) -> None:
        invalid_config = RandomForestTrainingConfig(tree_counts=(10, 5, 10))
        with self.assertRaisesRegex(ModelTrainingError, "strictly increasing"):
            train_random_forest(
                self.prepared.training,
                self.prepared.validation,
                invalid_config,
            )


if __name__ == "__main__":
    unittest.main()
