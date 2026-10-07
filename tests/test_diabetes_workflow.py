"""Tests for the end-to-end student-selected training session."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from services.diabetes_model.schema import load_reference_cohort
from services.diabetes_model.split import SplitValidationError
from services.diabetes_model.workflow import (
    DEFAULT_COHORT_PATH,
    SESSION_SCHEMA_VERSION,
    run_training_session,
)


class DiabetesTrainingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = load_reference_cohort(DEFAULT_COHORT_PATH)
        label_zero_ids = [
            record.patient_id for record in records if record.diabetes_label == 0
        ][:10]
        label_one_ids = [
            record.patient_id for record in records if record.diabetes_label == 1
        ][:10]
        cls.selected_ids = label_zero_ids + label_one_ids
        cls.session = run_training_session(cls.selected_ids)

    def test_student_selection_runs_complete_session(self) -> None:
        self.assertEqual(
            set(record.patient_id for record in self.session.patient_split.training),
            set(self.selected_ids),
        )
        self.assertEqual(len(self.session.patient_split.training), 20)
        self.assertEqual(len(self.session.patient_split.validation), 10)
        self.assertEqual(len(self.session.patient_split.test), 10)
        self.assertEqual(len(self.session.training_result.history), 20)
        self.assertEqual(len(self.session.evaluation_result.predictions), 10)
        self.assertEqual(len(self.session.feature_importance), 6)

    def test_session_response_has_backend_ready_sections(self) -> None:
        payload = self.session.as_dict()

        self.assertEqual(payload["schema_version"], SESSION_SCHEMA_VERSION)
        self.assertEqual(payload["session_status"], "completed")
        self.assertTrue(payload["educational_only"])
        self.assertEqual(
            payload["split_counts"],
            {"training": 20, "validation": 10, "test": 10},
        )
        self.assertEqual(payload["model"]["algorithm"], "RandomForestClassifier")
        self.assertEqual(payload["model"]["best_tree_count"], 60)
        self.assertEqual(len(payload["training_curve"]["points"]), 20)
        self.assertEqual(len(payload["feature_importance"]["features"]), 6)
        self.assertEqual(payload["test_evaluation"]["test_sample_count"], 10)
        json.dumps(payload)

    def test_test_predictions_are_for_the_locked_test_group(self) -> None:
        payload = self.session.as_dict()
        test_ids = set(payload["split"]["test"])
        predicted_ids = {
            prediction["patient_id"]
            for prediction in payload["test_evaluation"]["predictions"]
        }
        self.assertEqual(predicted_ids, test_ids)
        self.assertTrue(test_ids.isdisjoint(payload["split"]["training"]))
        self.assertTrue(test_ids.isdisjoint(payload["split"]["validation"]))

    def test_same_selection_produces_same_json_response(self) -> None:
        repeated = run_training_session(reversed(self.selected_ids))
        self.assertEqual(repeated.as_dict(), self.session.as_dict())

    def test_invalid_student_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(SplitValidationError, "Exactly 20"):
            run_training_session(self.selected_ids[:-1])

    def test_default_cohort_path_exists(self) -> None:
        self.assertTrue(Path(DEFAULT_COHORT_PATH).is_file())


if __name__ == "__main__":
    unittest.main()
