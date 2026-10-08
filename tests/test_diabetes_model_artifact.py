"""Tests for model export, feature importance, and prediction contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.diabetes_model.features import MODEL_FEATURES, prepare_model_split
from services.diabetes_model.model_artifact import (
    ARTIFACT_VERSION,
    export_model_bundle,
    extract_feature_importance,
    load_model_bundle,
)
from services.diabetes_model.prediction import (
    PredictionInputError,
    predict_patient,
)
from services.diabetes_model.schema import load_reference_cohort
from services.diabetes_model.split import create_student_split
from services.diabetes_model.training import train_random_forest
from services.diabetes_model.visualization import plot_feature_importance


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"


class DiabetesModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        records = load_reference_cohort(DATASET_PATH)
        label_zero_ids = [
            record.patient_id for record in records if record.diabetes_label == 0
        ][:10]
        label_one_ids = [
            record.patient_id for record in records if record.diabetes_label == 1
        ][:10]
        cls.patient_split = create_student_split(
            records, label_zero_ids + label_one_ids
        )
        cls.prepared = prepare_model_split(cls.patient_split)
        cls.training_result = train_random_forest(
            cls.prepared.training, cls.prepared.validation
        )
        cls.importance = extract_feature_importance(cls.training_result)

    def _example_payload(self) -> dict[str, object]:
        record = self.patient_split.test[0]
        row = record.to_csv_row()
        row.pop("diabetes_label")
        return row

    def test_feature_importance_is_ranked_and_complete(self) -> None:
        self.assertEqual(len(self.importance), len(MODEL_FEATURES))
        self.assertEqual(
            {entry.feature_name for entry in self.importance}, set(MODEL_FEATURES)
        )
        self.assertEqual(
            [entry.rank for entry in self.importance],
            list(range(1, len(MODEL_FEATURES) + 1)),
        )
        self.assertEqual(
            [entry.importance for entry in self.importance],
            sorted(
                (entry.importance for entry in self.importance), reverse=True
            ),
        )
        self.assertAlmostEqual(
            sum(entry.importance for entry in self.importance), 1.0
        )

    def test_exported_model_round_trip_preserves_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            paths = export_model_bundle(
                self.training_result,
                Path(temp_directory) / "model.joblib",
            )
            bundle = load_model_bundle(paths.model)
            payload = self._example_payload()
            exported_prediction = predict_patient(bundle, payload)
            original_prediction = int(
                self.training_result.model.predict(
                    [[float(payload[name]) for name in MODEL_FEATURES]]
                )[0]
            )

            self.assertEqual(exported_prediction.predicted_label, original_prediction)
            self.assertEqual(bundle.artifact_version, ARTIFACT_VERSION)
            self.assertEqual(bundle.feature_names, MODEL_FEATURES)
            self.assertEqual(bundle.best_tree_count, 60)

    def test_metadata_is_json_and_does_not_embed_the_estimator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            paths = export_model_bundle(
                self.training_result,
                Path(temp_directory) / "model.joblib",
            )
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))

            self.assertEqual(metadata["artifact_version"], ARTIFACT_VERSION)
            self.assertTrue(metadata["educational_only"])
            self.assertEqual(metadata["feature_names"], list(MODEL_FEATURES))
            self.assertNotIn("model", metadata)

    def test_prediction_response_is_json_serialisable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            paths = export_model_bundle(
                self.training_result,
                Path(temp_directory) / "model.joblib",
            )
            bundle = load_model_bundle(paths.model)
            response = predict_patient(bundle, self._example_payload()).as_dict()

            self.assertIn(response["predicted_label"], {0, 1})
            self.assertGreaterEqual(response["estimated_diabetes_probability"], 0)
            self.assertLessEqual(response["estimated_diabetes_probability"], 1)
            self.assertTrue(response["educational_only"])
            json.dumps(response)

    def test_prediction_rejects_missing_and_unexpected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            paths = export_model_bundle(
                self.training_result,
                Path(temp_directory) / "model.joblib",
            )
            bundle = load_model_bundle(paths.model)
            missing = self._example_payload()
            missing.pop("bmi")
            with self.assertRaisesRegex(PredictionInputError, "missing: bmi"):
                predict_patient(bundle, missing)

            extra = self._example_payload()
            extra["diabetes_label"] = 0
            with self.assertRaisesRegex(PredictionInputError, "unexpected"):
                predict_patient(bundle, extra)

    def test_prediction_rejects_values_outside_educational_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            paths = export_model_bundle(
                self.training_result,
                Path(temp_directory) / "model.joblib",
            )
            bundle = load_model_bundle(paths.model)
            invalid_glucose = self._example_payload()
            invalid_glucose["glucose_mmol_l"] = 99
            with self.assertRaisesRegex(PredictionInputError, "educational range"):
                predict_patient(bundle, invalid_glucose)

            invalid_history = self._example_payload()
            invalid_history["family_history"] = True
            with self.assertRaisesRegex(PredictionInputError, "integer 0 or 1"):
                predict_patient(bundle, invalid_history)

    def test_feature_importance_renderer_writes_a_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "importance.png"
            rendered_path = plot_feature_importance(self.importance, output_path)
            self.assertEqual(rendered_path, output_path)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
