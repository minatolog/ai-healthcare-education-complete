"""Tests for student-selection sensitivity analysis and visualisation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.diabetes_model.robustness import (
    ROBUSTNESS_NOTE,
    RobustnessAnalysisError,
    analyse_selection_robustness,
)
from services.diabetes_model.visualization import plot_selection_robustness


class DiabetesRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = analyse_selection_robustness(scenario_count=6)

    def test_report_contains_unique_valid_scenarios(self) -> None:
        self.assertEqual(self.report.scenario_count, 6)
        selections = [
            scenario.training_patient_ids for scenario in self.report.scenarios
        ]
        self.assertEqual(len(set(selections)), 6)
        for scenario in self.report.scenarios:
            self.assertEqual(len(scenario.training_patient_ids), 20)
            self.assertGreaterEqual(scenario.training_label_0_count, 2)
            self.assertGreaterEqual(scenario.training_label_1_count, 2)
            self.assertLessEqual(scenario.training_label_0_count, 18)
            self.assertLessEqual(scenario.training_label_1_count, 18)

    def test_report_is_deterministic(self) -> None:
        repeated = analyse_selection_robustness(scenario_count=6)
        self.assertEqual(repeated, self.report)

    def test_summary_is_json_ready_and_bounded(self) -> None:
        payload = self.report.as_dict()
        self.assertEqual(payload["scenario_count"], 6)
        self.assertEqual(payload["note"], ROBUSTNESS_NOTE)
        self.assertEqual(len(payload["mean_feature_importance"]), 6)
        self.assertEqual(len(payload["scenarios"]), 6)
        for metric_summary in payload["summary"].values():
            self.assertGreaterEqual(metric_summary["minimum"], 0.0)
            self.assertLessEqual(metric_summary["maximum"], 1.0)
            self.assertLessEqual(
                metric_summary["minimum"], metric_summary["maximum"]
            )
        json.dumps(payload)

    def test_invalid_scenario_count_is_rejected(self) -> None:
        for invalid_count in (0, -1, 201, True):
            with self.subTest(scenario_count=invalid_count):
                with self.assertRaises(RobustnessAnalysisError):
                    analyse_selection_robustness(
                        scenario_count=invalid_count
                    )

    def test_robustness_renderer_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "robustness.png"
            rendered_path = plot_selection_robustness(
                self.report, output_path
            )
            self.assertEqual(rendered_path, output_path)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
