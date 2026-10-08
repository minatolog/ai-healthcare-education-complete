"""Compare the educational Random Forest with simple reference models."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import prepare_model_split
from .robustness import (
    DEFAULT_SCENARIO_COUNT,
    DEFAULT_SELECTION_SEED,
    SelectionRobustnessReport,
    analyse_selection_robustness,
)
from .schema import load_reference_cohort
from .split import DEFAULT_SPLIT_SEED, create_student_split
from .workflow import DEFAULT_COHORT_PATH


MODEL_ORDER = (
    "random_forest",
    "logistic_regression",
    "dummy_most_frequent",
)
MODEL_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "dummy_most_frequent": "Dummy baseline",
}
BENCHMARK_NOTE = (
    "Synthetic small-data comparison; results do not represent clinical performance."
)
MODEL_CHOICE_NOTE = (
    "The benchmark measures predictive stability. Random Forest remains the "
    "activity model for its tree-growth visualisation and intuitive split "
    "importance, not because it has the highest benchmark score."
)


@dataclass(frozen=True)
class ScenarioModelMetrics:
    """Three classification metrics for one model in one scenario."""

    algorithm: str
    accuracy: float
    balanced_accuracy: float
    f1: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "algorithm": self.algorithm,
            "display_name": MODEL_DISPLAY_NAMES[self.algorithm],
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(frozen=True)
class ModelBenchmarkScenario:
    """Results from all models on one identical training/test split."""

    scenario_number: int
    training_patient_ids: tuple[str, ...]
    metrics: tuple[ScenarioModelMetrics, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_number": self.scenario_number,
            "training_patient_ids": list(self.training_patient_ids),
            "models": [metric.as_dict() for metric in self.metrics],
        }


@dataclass(frozen=True)
class ModelBenchmarkReport:
    """Aggregate and per-scenario comparison of three classifiers."""

    scenario_count: int
    scenarios: tuple[ModelBenchmarkScenario, ...]

    def summary_by_algorithm(self) -> dict[str, dict[str, object]]:
        summary: dict[str, dict[str, object]] = {}
        for algorithm in MODEL_ORDER:
            model_metrics = [
                metric
                for scenario in self.scenarios
                for metric in scenario.metrics
                if metric.algorithm == algorithm
            ]
            summary[algorithm] = {
                "display_name": MODEL_DISPLAY_NAMES[algorithm],
                "accuracy": _summarise(
                    metric.accuracy for metric in model_metrics
                ),
                "balanced_accuracy": _summarise(
                    metric.balanced_accuracy for metric in model_metrics
                ),
                "f1": _summarise(metric.f1 for metric in model_metrics),
            }
        return summary

    def as_dict(self) -> dict[str, object]:
        summary = self.summary_by_algorithm()
        best_algorithm = max(
            MODEL_ORDER,
            key=lambda algorithm: summary[algorithm]["balanced_accuracy"][
                "mean"
            ],
        )
        return {
            "analysis_type": "model_benchmark",
            "scenario_count": self.scenario_count,
            "models": list(MODEL_ORDER),
            "summary": summary,
            "highest_mean_balanced_accuracy": best_algorithm,
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
            "model_choice_note": MODEL_CHOICE_NOTE,
            "note": BENCHMARK_NOTE,
        }


def benchmark_models(
    *,
    cohort_path: str | Path = DEFAULT_COHORT_PATH,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
    robustness_report: SelectionRobustnessReport | None = None,
) -> ModelBenchmarkReport:
    """Compare models using identical deterministic student selections."""

    robustness = robustness_report or analyse_selection_robustness(
        cohort_path=cohort_path,
        scenario_count=scenario_count,
        selection_seed=selection_seed,
        split_seed=split_seed,
    )
    if robustness.scenario_count != scenario_count:
        raise ValueError(
            "robustness_report scenario count does not match scenario_count."
        )

    records = load_reference_cohort(cohort_path)
    scenarios: list[ModelBenchmarkScenario] = []
    for robustness_scenario in robustness.scenarios:
        patient_split = create_student_split(
            records,
            robustness_scenario.training_patient_ids,
            seed=split_seed,
        )
        prepared = prepare_model_split(patient_split)
        metrics: list[ScenarioModelMetrics] = [
            ScenarioModelMetrics(
                algorithm="random_forest",
                accuracy=robustness_scenario.test_accuracy,
                balanced_accuracy=(
                    robustness_scenario.test_balanced_accuracy
                ),
                f1=robustness_scenario.test_f1,
            )
        ]

        reference_models = {
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    random_state=9900,
                    max_iter=1000,
                ),
            ),
            "dummy_most_frequent": DummyClassifier(
                strategy="most_frequent"
            ),
        }
        for algorithm, model in reference_models.items():
            model.fit(prepared.training.features, prepared.training.labels)
            predictions = model.predict(prepared.test.features)
            metrics.append(
                ScenarioModelMetrics(
                    algorithm=algorithm,
                    accuracy=float(
                        accuracy_score(prepared.test.labels, predictions)
                    ),
                    balanced_accuracy=float(
                        balanced_accuracy_score(
                            prepared.test.labels, predictions
                        )
                    ),
                    f1=float(
                        f1_score(
                            prepared.test.labels,
                            predictions,
                            zero_division=0,
                        )
                    ),
                )
            )

        scenarios.append(
            ModelBenchmarkScenario(
                scenario_number=robustness_scenario.scenario_number,
                training_patient_ids=(
                    robustness_scenario.training_patient_ids
                ),
                metrics=tuple(metrics),
            )
        )

    return ModelBenchmarkReport(
        scenario_count=scenario_count,
        scenarios=tuple(scenarios),
    )


def _summarise(values: Iterable[float]) -> dict[str, float]:
    numeric_values = tuple(float(value) for value in values)
    return {
        "minimum": round(min(numeric_values), 4),
        "median": round(statistics.median(numeric_values), 4),
        "mean": round(statistics.fmean(numeric_values), 4),
        "maximum": round(max(numeric_values), 4),
    }
