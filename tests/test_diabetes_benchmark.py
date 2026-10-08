"""Tests for the three-model small-data benchmark."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.diabetes_model.benchmark import (
    MODEL_ORDER,
    benchmark_models,
)
from services.diabetes_model.robustness import analyse_selection_robustness
from services.diabetes_model.visualization import plot_model_benchmark


class DiabetesModelBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robustness = analyse_selection_robustness(scenario_count=6)
        cls.report = benchmark_models(
            scenario_count=6,
            robustness_report=cls.robustness,
        )

    def test_every_scenario_compares_the_same_three_models(self) -> None:
        self.assertEqual(self.report.scenario_count, 6)
        self.assertEqual(len(self.report.scenarios), 6)
        for scenario in self.report.scenarios:
            self.assertEqual(
                tuple(metric.algorithm for metric in scenario.metrics),
                MODEL_ORDER,
            )
            self.assertEqual(len(scenario.training_patient_ids), 20)

    def test_summary_shows_reference_models_and_valid_scores(self) -> None:
        summary = self.report.summary_by_algorithm()
        self.assertEqual(tuple(summary), MODEL_ORDER)
        for algorithm_summary in summary.values():
            for metric_name in ("accuracy", "balanced_accuracy", "f1"):
                metric_summary = algorithm_summary[metric_name]
                self.assertGreaterEqual(metric_summary["minimum"], 0.0)
                self.assertLessEqual(metric_summary["maximum"], 1.0)

        self.assertGreater(
            summary["logistic_regression"]["balanced_accuracy"]["mean"],
            summary["random_forest"]["balanced_accuracy"]["mean"],
        )
        self.assertEqual(
            summary["dummy_most_frequent"]["balanced_accuracy"]["mean"],
            0.5,
        )

    def test_benchmark_output_is_json_ready(self) -> None:
        payload = self.report.as_dict()
        self.assertEqual(payload["analysis_type"], "model_benchmark")
        self.assertEqual(
            payload["highest_mean_balanced_accuracy"],
            "logistic_regression",
        )
        self.assertEqual(len(payload["scenarios"]), 6)
        json.dumps(payload)

    def test_benchmark_is_deterministic(self) -> None:
        repeated = benchmark_models(
            scenario_count=6,
            robustness_report=self.robustness,
        )
        self.assertEqual(repeated, self.report)

    def test_mismatched_robustness_report_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            benchmark_models(
                scenario_count=5,
                robustness_report=self.robustness,
            )

    def test_benchmark_renderer_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "benchmark.png"
            rendered_path = plot_model_benchmark(self.report, output_path)
            self.assertEqual(rendered_path, output_path)
            self.assertEqual(output_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
