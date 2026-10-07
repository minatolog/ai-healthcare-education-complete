"""Tests for local one-feature-at-a-time model explanations."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from services.diabetes_model.explanations import (
    LOCAL_EXPLANATION_METHOD,
    LocalExplanationError,
    explain_dataset_predictions,
)
from services.diabetes_model.features import MODEL_FEATURES, prepare_model_split
from services.diabetes_model.schema import load_reference_cohort
from services.diabetes_model.split import create_student_split
from services.diabetes_model.training import train_random_forest
from services.diabetes_model.visualization import plot_patient_explanation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"


class DiabetesLocalExplanationTests(unittest.TestCase):
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
        cls.explanations = explain_dataset_predictions(
            cls.training_result,
            cls.prepared.training,
            cls.prepared.test,
        )

    def test_every_test_patient_has_six_ranked_effects(self) -> None:
        self.assertEqual(len(self.explanations), 10)
        self.assertEqual(
            {item.patient_id for item in self.explanations},
            set(self.prepared.test.patient_ids),
        )
        for explanation in self.explanations:
            self.assertEqual(len(explanation.feature_effects), len(MODEL_FEATURES))
            self.assertEqual(
                [effect.rank for effect in explanation.feature_effects],
                list(range(1, len(MODEL_FEATURES) + 1)),
            )
            magnitudes = [
                abs(effect.probability_change)
                for effect in explanation.feature_effects
            ]
            self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_baseline_probability_matches_model_prediction(self) -> None:
        probabilities = self.training_result.model.predict_proba(
            self.prepared.test.features
        )[:, 1]
        for explanation, expected_probability in zip(
            self.explanations, probabilities
        ):
            self.assertAlmostEqual(
                explanation.estimated_diabetes_probability,
                expected_probability,
            )

    def test_effect_direction_matches_probability_change(self) -> None:
        for explanation in self.explanations:
            for effect in explanation.feature_effects:
                if effect.probability_change > 1e-12:
                    self.assertEqual(effect.direction, "increased_score")
                elif effect.probability_change < -1e-12:
                    self.assertEqual(effect.direction, "decreased_score")
                else:
                    self.assertEqual(effect.direction, "no_change")

    def test_explanation_is_json_ready_without_actual_label(self) -> None:
        payload = self.explanations[0].as_dict()
        self.assertEqual(payload["method"], LOCAL_EXPLANATION_METHOD)
        self.assertNotIn("actual_label", payload)
        json.dumps(payload)

    def test_mismatched_training_data_is_rejected(self) -> None:
        mismatched_training = replace(
            self.prepared.training,
            patient_ids=tuple(reversed(self.prepared.training.patient_ids)),
        )
        with self.assertRaisesRegex(LocalExplanationError, "must match"):
            explain_dataset_predictions(
                self.training_result,
                mismatched_training,
                self.prepared.test,
            )

    def test_local_explanation_renderer_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "local_explanation.png"
            rendered_path = plot_patient_explanation(
                self.explanations[0], output_path
            )
            self.assertEqual(rendered_path, output_path)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
