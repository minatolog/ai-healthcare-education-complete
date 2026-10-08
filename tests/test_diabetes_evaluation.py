"""Tests for held-out evaluation and training-curve rendering."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

from services.diabetes_model.evaluation import (
    ModelEvaluationError,
    evaluate_test_set,
)
from services.diabetes_model.features import prepare_model_split
from services.diabetes_model.schema import load_reference_cohort
from services.diabetes_model.split import create_student_split
from services.diabetes_model.training import train_random_forest
from services.diabetes_model.visualization import plot_training_history


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"


class DiabetesEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = load_reference_cohort(DATASET_PATH)
        label_zero_ids = [
            record.patient_id for record in records if record.diabetes_label == 0
        ][:10]
        label_one_ids = [
            record.patient_id for record in records if record.diabetes_label == 1
        ][:10]
        split = create_student_split(records, label_zero_ids + label_one_ids)
        cls.prepared = prepare_model_split(split)
        cls.training_result = train_random_forest(
            cls.prepared.training, cls.prepared.validation
        )
        cls.evaluation = evaluate_test_set(
            cls.training_result, cls.prepared.test
        )

    def test_default_demo_test_metrics(self) -> None:
        metrics = self.evaluation.metrics
        self.assertEqual(metrics.accuracy, 0.9)
        self.assertEqual(metrics.balanced_accuracy, 0.9)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 0.8)
        self.assertAlmostEqual(metrics.f1, 0.8888888888888888)
        self.assertEqual(metrics.confusion_matrix, ((5, 0), (1, 4)))

    def test_every_test_patient_has_a_probability(self) -> None:
        self.assertEqual(len(self.evaluation.predictions), 10)
        self.assertEqual(
            {prediction.patient_id for prediction in self.evaluation.predictions},
            set(self.prepared.test.patient_ids),
        )
        for prediction in self.evaluation.predictions:
            self.assertGreaterEqual(prediction.estimated_diabetes_probability, 0.0)
            self.assertLessEqual(prediction.estimated_diabetes_probability, 1.0)

    def test_evaluation_output_is_json_serialisable(self) -> None:
        payload = self.evaluation.as_dict()
        self.assertEqual(payload["evaluation_scope"], "held_out_test")
        self.assertEqual(payload["test_sample_count"], 10)
        json.dumps(payload)

    def test_seen_patient_is_rejected_from_test_group(self) -> None:
        overlapping_test = replace(
            self.prepared.test,
            patient_ids=(
                self.prepared.training.patient_ids[0],
                *self.prepared.test.patient_ids[1:],
            ),
        )
        with self.assertRaisesRegex(ModelEvaluationError, "must be unseen"):
            evaluate_test_set(self.training_result, overlapping_test)

    def test_unfitted_model_is_rejected(self) -> None:
        unfitted_result = replace(
            self.training_result,
            model=RandomForestClassifier(random_state=9900),
        )
        with self.assertRaisesRegex(ModelEvaluationError, "has not been fitted"):
            evaluate_test_set(unfitted_result, self.prepared.test)

    def test_duplicate_test_patient_is_rejected(self) -> None:
        duplicate_test = replace(
            self.prepared.test,
            patient_ids=(
                self.prepared.test.patient_ids[0],
                self.prepared.test.patient_ids[0],
                *self.prepared.test.patient_ids[2:],
            ),
        )
        with self.assertRaisesRegex(ModelEvaluationError, "must be unique"):
            evaluate_test_set(self.training_result, duplicate_test)

    def test_curve_renderer_writes_a_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "curve.png"
            rendered_path = plot_training_history(
                self.training_result, output_path
            )
            self.assertEqual(rendered_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
