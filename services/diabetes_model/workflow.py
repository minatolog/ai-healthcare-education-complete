"""End-to-end training session for a student's 20-patient selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .evaluation import FinalEvaluationResult, evaluate_test_set
from .features import MODEL_FEATURES, prepare_model_split
from .explanations import (
    LOCAL_EXPLANATION_METHOD,
    LOCAL_EXPLANATION_NOTE,
    PatientLocalExplanation,
    explain_dataset_predictions,
)
from .model_artifact import (
    FEATURE_IMPORTANCE_NOTE,
    FeatureImportanceEntry,
    extract_feature_importance,
)
from .schema import load_reference_cohort
from .split import (
    DEFAULT_SPLIT_SEED,
    PatientSplit,
    SplitValidationError,
    TrainingSelectionAssessment,
    assess_training_selection,
    create_student_split,
)
from .training import (
    RandomForestTrainingConfig,
    RandomForestTrainingResult,
    train_random_forest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT_PATH = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"
SESSION_SCHEMA_VERSION = "1.2"


class TrainingSelectionError(SplitValidationError):
    """Structured pre-flight failure raised before model training starts."""

    def __init__(self, assessment: TrainingSelectionAssessment) -> None:
        self.assessment = assessment
        super().__init__(assessment.error_message())

    def as_dict(self) -> dict[str, object]:
        return {
            "error": "invalid_training_selection",
            "assessment": self.assessment.as_dict(),
        }


@dataclass(frozen=True)
class TrainingSessionResult:
    """Model objects and a JSON-ready response from one complete session."""

    selection_assessment: TrainingSelectionAssessment
    patient_split: PatientSplit
    training_result: RandomForestTrainingResult
    evaluation_result: FinalEvaluationResult
    feature_importance: tuple[FeatureImportanceEntry, ...]
    test_explanations: tuple[PatientLocalExplanation, ...]

    def as_dict(self) -> dict[str, object]:
        """Return the contract consumed by a future backend endpoint."""

        split_ids = self.patient_split.as_id_dict()
        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_status": "completed",
            "educational_only": True,
            "selection_assessment": self.selection_assessment.as_dict(),
            "split": split_ids,
            "split_counts": {
                group_name: len(patient_ids)
                for group_name, patient_ids in split_ids.items()
            },
            "model": {
                "algorithm": "RandomForestClassifier",
                "feature_names": list(MODEL_FEATURES),
                "best_tree_count": self.training_result.best_tree_count,
                "best_validation_accuracy": round(
                    self.training_result.best_validation_accuracy, 4
                ),
            },
            "training_curve": {
                "x_axis": "trees_built",
                "series": ["train_accuracy", "validation_accuracy"],
                "points": self.training_result.history_as_dicts(),
            },
            "feature_importance": {
                "importance_type": "global_split_importance",
                "features": [
                    entry.as_dict() for entry in self.feature_importance
                ],
                "note": FEATURE_IMPORTANCE_NOTE,
            },
            "test_evaluation": self.evaluation_result.as_dict(),
            "test_explanations": {
                "method": LOCAL_EXPLANATION_METHOD,
                "patients": [
                    explanation.as_dict()
                    for explanation in self.test_explanations
                ],
                "note": LOCAL_EXPLANATION_NOTE,
            },
        }


def run_training_session(
    training_patient_ids: Iterable[str],
    *,
    cohort_path: str | Path = DEFAULT_COHORT_PATH,
    split_seed: int = DEFAULT_SPLIT_SEED,
    training_config: RandomForestTrainingConfig | None = None,
) -> TrainingSessionResult:
    """Run split, preparation, model selection, and held-out evaluation.

    The function is synchronous and framework-independent. A backend can call
    it after receiving the student's 20 patient IDs, while a frontend can
    animate the returned checkpoint history after the response arrives.
    """

    records = load_reference_cohort(cohort_path)
    assessment = assess_training_selection(records, training_patient_ids)
    if not assessment.ready:
        raise TrainingSelectionError(assessment)
    patient_split = create_student_split(
        records,
        assessment.selected_patient_ids,
        seed=split_seed,
    )
    prepared = prepare_model_split(patient_split)
    training_result = train_random_forest(
        prepared.training,
        prepared.validation,
        training_config,
    )
    evaluation_result = evaluate_test_set(training_result, prepared.test)
    feature_importance = extract_feature_importance(training_result)
    test_explanations = explain_dataset_predictions(
        training_result,
        prepared.training,
        prepared.test,
    )

    return TrainingSessionResult(
        selection_assessment=assessment,
        patient_split=patient_split,
        training_result=training_result,
        evaluation_result=evaluation_result,
        feature_importance=feature_importance,
        test_explanations=test_explanations,
    )
