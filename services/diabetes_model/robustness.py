"""Sensitivity analysis across different valid student training selections."""

from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .features import FEATURE_DEFINITIONS
from .schema import load_reference_cohort
from .split import DEFAULT_SPLIT_SEED, TRAINING_SIZE
from .workflow import DEFAULT_COHORT_PATH, run_training_session


DEFAULT_SCENARIO_COUNT = 30
DEFAULT_SELECTION_SEED = 9900
ROBUSTNESS_NOTE = (
    "Exploratory sensitivity analysis on synthetic data; this is not independent "
    "clinical validation."
)


class RobustnessAnalysisError(ValueError):
    """Raised when a robustness analysis configuration is invalid."""


@dataclass(frozen=True)
class SelectionScenarioResult:
    """Metrics produced by one valid 20-patient training selection."""

    scenario_number: int
    training_patient_ids: tuple[str, ...]
    training_label_0_count: int
    training_label_1_count: int
    best_tree_count: int
    best_validation_accuracy: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_f1: float
    feature_importance: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_number": self.scenario_number,
            "training_patient_ids": list(self.training_patient_ids),
            "training_class_counts": {
                "label_0": self.training_label_0_count,
                "label_1": self.training_label_1_count,
            },
            "best_tree_count": self.best_tree_count,
            "best_validation_accuracy": round(
                self.best_validation_accuracy, 4
            ),
            "test_accuracy": round(self.test_accuracy, 4),
            "test_balanced_accuracy": round(
                self.test_balanced_accuracy, 4
            ),
            "test_f1": round(self.test_f1, 4),
        }


@dataclass(frozen=True)
class SelectionRobustnessReport:
    """Reproducible results across multiple valid student selections."""

    scenario_count: int
    selection_seed: int
    split_seed: int
    scenarios: tuple[SelectionScenarioResult, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "analysis_type": "student_selection_sensitivity",
            "scenario_count": self.scenario_count,
            "selection_seed": self.selection_seed,
            "split_seed": self.split_seed,
            "summary": {
                "best_validation_accuracy": _summarise(
                    scenario.best_validation_accuracy
                    for scenario in self.scenarios
                ),
                "test_accuracy": _summarise(
                    scenario.test_accuracy for scenario in self.scenarios
                ),
                "test_balanced_accuracy": _summarise(
                    scenario.test_balanced_accuracy
                    for scenario in self.scenarios
                ),
                "test_f1": _summarise(
                    scenario.test_f1 for scenario in self.scenarios
                ),
            },
            "selected_tree_count_frequency": {
                str(tree_count): frequency
                for tree_count, frequency in sorted(
                    Counter(
                        scenario.best_tree_count for scenario in self.scenarios
                    ).items()
                )
            },
            "mean_feature_importance": self._mean_feature_importance(),
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
            "note": ROBUSTNESS_NOTE,
        }

    def _mean_feature_importance(self) -> list[dict[str, str | float]]:
        values_by_feature: dict[str, list[float]] = {
            definition.name: [] for definition in FEATURE_DEFINITIONS
        }
        display_names = {
            definition.name: definition.display_name
            for definition in FEATURE_DEFINITIONS
        }
        for scenario in self.scenarios:
            for feature_name, importance in scenario.feature_importance:
                values_by_feature[feature_name].append(importance)

        ranked = sorted(
            (
                (feature_name, statistics.fmean(values))
                for feature_name, values in values_by_feature.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            {
                "feature_name": feature_name,
                "display_name": display_names[feature_name],
                "mean_importance": round(mean_importance, 6),
            }
            for feature_name, mean_importance in ranked
        ]


def analyse_selection_robustness(
    *,
    cohort_path: str | Path = DEFAULT_COHORT_PATH,
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> SelectionRobustnessReport:
    """Train deterministic scenarios to measure sensitivity to student choice."""

    if type(scenario_count) is not int or scenario_count <= 0:
        raise RobustnessAnalysisError("scenario_count must be a positive integer.")
    if scenario_count > 200:
        raise RobustnessAnalysisError("scenario_count must not exceed 200.")
    if type(selection_seed) is not int or type(split_seed) is not int:
        raise RobustnessAnalysisError("selection_seed and split_seed must be integers.")

    records = load_reference_cohort(cohort_path)
    records_by_id = {record.patient_id: record for record in records}
    patient_ids = sorted(records_by_id)
    rng = random.Random(selection_seed)
    seen_selections: set[tuple[str, ...]] = set()
    scenarios: list[SelectionScenarioResult] = []

    while len(scenarios) < scenario_count:
        selected_ids = tuple(sorted(rng.sample(patient_ids, TRAINING_SIZE)))
        if selected_ids in seen_selections:
            continue
        seen_selections.add(selected_ids)

        class_counts = Counter(
            records_by_id[patient_id].diabetes_label
            for patient_id in selected_ids
        )
        # Both training and remaining groups need at least two of each label so
        # that validation and test can each contain both classes.
        if min(class_counts.get(0, 0), class_counts.get(1, 0)) < 2:
            continue
        if max(class_counts[0], class_counts[1]) > 18:
            continue

        session = run_training_session(
            selected_ids,
            cohort_path=cohort_path,
            split_seed=split_seed,
        )
        metrics = session.evaluation_result.metrics
        scenarios.append(
            SelectionScenarioResult(
                scenario_number=len(scenarios) + 1,
                training_patient_ids=selected_ids,
                training_label_0_count=class_counts[0],
                training_label_1_count=class_counts[1],
                best_tree_count=session.training_result.best_tree_count,
                best_validation_accuracy=(
                    session.training_result.best_validation_accuracy
                ),
                test_accuracy=metrics.accuracy,
                test_balanced_accuracy=metrics.balanced_accuracy,
                test_f1=metrics.f1,
                feature_importance=tuple(
                    (entry.feature_name, entry.importance)
                    for entry in session.feature_importance
                ),
            )
        )

    return SelectionRobustnessReport(
        scenario_count=scenario_count,
        selection_seed=selection_seed,
        split_seed=split_seed,
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
