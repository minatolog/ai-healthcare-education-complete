"""Create the 20/10/10 patient split for the student training activity."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .schema import PatientRecord, validate_reference_cohort


TRAINING_SIZE = 20
VALIDATION_SIZE = 10
TEST_SIZE = 10
DEFAULT_SPLIT_SEED = 9900


class SplitValidationError(ValueError):
    """Raised when a student selection cannot produce a valid 20/10/10 split."""


@dataclass(frozen=True)
class SelectionIssue:
    """Machine-readable error or warning about selected training patients."""

    code: str
    severity: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True)
class TrainingSelectionAssessment:
    """Pre-flight result that a backend can return before model training."""

    selected_patient_ids: tuple[str, ...]
    selected_count: int
    unique_count: int
    training_label_0_count: int | None
    training_label_1_count: int | None
    remaining_label_0_count: int | None
    remaining_label_1_count: int | None
    issues: tuple[SelectionIssue, ...]

    @property
    def ready(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def error_message(self) -> str:
        return "\n".join(
            issue.message for issue in self.issues if issue.severity == "error"
        )

    def as_dict(self) -> dict[str, object]:
        training_counts = None
        remaining_counts = None
        if self.training_label_0_count is not None:
            training_counts = {
                "label_0": self.training_label_0_count,
                "label_1": self.training_label_1_count,
            }
            remaining_counts = {
                "label_0": self.remaining_label_0_count,
                "label_1": self.remaining_label_1_count,
            }
        return {
            "ready": self.ready,
            "selected_count": self.selected_count,
            "unique_count": self.unique_count,
            "training_class_counts": training_counts,
            "remaining_class_counts": remaining_counts,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class PatientSplit:
    """Immutable groups used by training, model selection, and final testing."""

    training: tuple[PatientRecord, ...]
    validation: tuple[PatientRecord, ...]
    test: tuple[PatientRecord, ...]

    def as_id_dict(self) -> dict[str, list[str]]:
        """Return a backend-friendly representation containing patient IDs only."""

        return {
            "training": [record.patient_id for record in self.training],
            "validation": [record.patient_id for record in self.validation],
            "test": [record.patient_id for record in self.test],
        }

    def class_counts(self) -> dict[str, dict[int, int]]:
        """Return label counts for checks and educational explanations."""

        return {
            group_name: dict(
                sorted(Counter(record.diabetes_label for record in group).items())
            )
            for group_name, group in (
                ("training", self.training),
                ("validation", self.validation),
                ("test", self.test),
            )
        }


def create_student_split(
    records: Iterable[PatientRecord],
    training_patient_ids: Iterable[str],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
) -> PatientSplit:
    """Use 20 student-selected patients and split the remaining 20 in half.

    The validation and test groups are stratified by the synthetic class label.
    This keeps both labels represented without using the test group for training
    or model selection.
    """

    cohort = list(records)
    assessment = assess_training_selection(cohort, training_patient_ids)
    if not assessment.ready:
        raise SplitValidationError(assessment.error_message())

    selected_ids = list(assessment.selected_patient_ids)
    records_by_id = {record.patient_id: record for record in cohort}

    selected_id_set = set(selected_ids)
    training = sorted(
        (records_by_id[patient_id] for patient_id in selected_id_set),
        key=lambda record: record.patient_id,
    )
    remaining = [
        record for record in cohort if record.patient_id not in selected_id_set
    ]

    validation, test = _split_remaining_patients(remaining, seed)
    patient_split = PatientSplit(
        training=tuple(training),
        validation=tuple(validation),
        test=tuple(test),
    )
    validate_patient_split(cohort, patient_split)
    return patient_split


def assess_training_selection(
    records: Iterable[PatientRecord],
    training_patient_ids: object,
) -> TrainingSelectionAssessment:
    """Validate a proposed selection without starting model training."""

    cohort = list(records)
    validate_reference_cohort(cohort)
    issues: list[SelectionIssue] = []

    if isinstance(training_patient_ids, (str, bytes)):
        issues.append(
            SelectionIssue(
                code="INVALID_COLLECTION",
                severity="error",
                message=(
                    "training_patient_ids must be a collection of patient IDs."
                ),
            )
        )
        return _assessment_without_class_counts((), issues)
    try:
        raw_ids = tuple(training_patient_ids)
    except TypeError:
        issues.append(
            SelectionIssue(
                code="INVALID_COLLECTION",
                severity="error",
                message=(
                    "training_patient_ids must be a collection of patient IDs."
                ),
            )
        )
        return _assessment_without_class_counts((), issues)

    if len(raw_ids) != TRAINING_SIZE:
        issues.append(
            SelectionIssue(
                code="INCORRECT_COUNT",
                severity="error",
                message=(
                    f"Exactly {TRAINING_SIZE} training patients are required; "
                    f"received {len(raw_ids)}."
                ),
            )
        )
        string_ids = tuple(value for value in raw_ids if type(value) is str)
        return _assessment_without_class_counts(string_ids, issues)
    if any(type(patient_id) is not str for patient_id in raw_ids):
        issues.append(
            SelectionIssue(
                code="INVALID_ID_TYPE",
                severity="error",
                message="Every training patient ID must be a string.",
            )
        )
        string_ids = tuple(value for value in raw_ids if type(value) is str)
        return _assessment_without_class_counts(string_ids, issues)

    selected_ids = tuple(raw_ids)
    if len(set(selected_ids)) != len(selected_ids):
        issues.append(
            SelectionIssue(
                code="DUPLICATE_IDS",
                severity="error",
                message="Training patient IDs must be unique.",
            )
        )
        return _assessment_without_class_counts(selected_ids, issues)

    records_by_id = {record.patient_id: record for record in cohort}
    unknown_ids = sorted(set(selected_ids) - set(records_by_id))
    if unknown_ids:
        issues.append(
            SelectionIssue(
                code="UNKNOWN_IDS",
                severity="error",
                message=(
                    f"Unknown training patient IDs: {', '.join(unknown_ids)}."
                ),
            )
        )
        return _assessment_without_class_counts(selected_ids, issues)

    selected_id_set = set(selected_ids)
    training_counts = Counter(
        records_by_id[patient_id].diabetes_label
        for patient_id in selected_ids
    )
    remaining_counts = Counter(
        record.diabetes_label
        for record in cohort
        if record.patient_id not in selected_id_set
    )
    if set(training_counts) != {0, 1}:
        issues.append(
            SelectionIssue(
                code="TRAINING_SINGLE_CLASS",
                severity="error",
                message=(
                    "The training selection must contain both synthetic class "
                    "labels."
                ),
            )
        )
    if any(remaining_counts[label] < 2 for label in (0, 1)):
        issues.append(
            SelectionIssue(
                code="INSUFFICIENT_REMAINING_CLASS",
                severity="error",
                message=(
                    "Choose a different training set: the remaining patients must "
                    "include at least two examples of each label so validation and "
                    "test can both contain both labels."
                ),
            )
        )
    if set(training_counts) == {0, 1} and min(training_counts.values()) < 5:
        issues.append(
            SelectionIssue(
                code="TRAINING_CLASS_IMBALANCE",
                severity="warning",
                message=(
                    "The training selection is strongly imbalanced, so model "
                    "scores may be unstable."
                ),
            )
        )

    return TrainingSelectionAssessment(
        selected_patient_ids=selected_ids,
        selected_count=len(selected_ids),
        unique_count=len(set(selected_ids)),
        training_label_0_count=training_counts[0],
        training_label_1_count=training_counts[1],
        remaining_label_0_count=remaining_counts[0],
        remaining_label_1_count=remaining_counts[1],
        issues=tuple(issues),
    )


def _assessment_without_class_counts(
    selected_ids: tuple[str, ...],
    issues: list[SelectionIssue],
) -> TrainingSelectionAssessment:
    return TrainingSelectionAssessment(
        selected_patient_ids=selected_ids,
        selected_count=len(selected_ids),
        unique_count=len(set(selected_ids)),
        training_label_0_count=None,
        training_label_1_count=None,
        remaining_label_0_count=None,
        remaining_label_1_count=None,
        issues=tuple(issues),
    )


def _split_remaining_patients(
    remaining: list[PatientRecord], seed: int
) -> tuple[list[PatientRecord], list[PatientRecord]]:
    """Split remaining patients into deterministic, approximately stratified halves."""

    rng = random.Random(seed)
    validation_by_label: dict[int, list[PatientRecord]] = {}
    test_by_label: dict[int, list[PatientRecord]] = {}
    odd_labels: list[int] = []

    for label in (0, 1):
        group = sorted(
            (record for record in remaining if record.diabetes_label == label),
            key=lambda record: record.patient_id,
        )
        rng.shuffle(group)
        validation_count = len(group) // 2
        validation_by_label[label] = group[:validation_count]
        test_by_label[label] = group[validation_count:]
        if len(group) % 2:
            odd_labels.append(label)

    validation_count = sum(len(group) for group in validation_by_label.values())
    if validation_count < VALIDATION_SIZE:
        promoted_label = rng.choice(odd_labels)
        validation_by_label[promoted_label].append(
            test_by_label[promoted_label].pop()
        )

    validation = sorted(
        validation_by_label[0] + validation_by_label[1],
        key=lambda record: record.patient_id,
    )
    test = sorted(
        test_by_label[0] + test_by_label[1],
        key=lambda record: record.patient_id,
    )
    return validation, test


def validate_patient_split(
    cohort: Iterable[PatientRecord], patient_split: PatientSplit
) -> None:
    """Check size, coverage, isolation, and label representation for all groups."""

    records = list(cohort)
    validate_reference_cohort(records)
    expected_sizes = {
        "training": TRAINING_SIZE,
        "validation": VALIDATION_SIZE,
        "test": TEST_SIZE,
    }
    groups = {
        "training": patient_split.training,
        "validation": patient_split.validation,
        "test": patient_split.test,
    }

    errors: list[str] = []
    id_sets: dict[str, set[str]] = {}
    for group_name, group in groups.items():
        if len(group) != expected_sizes[group_name]:
            errors.append(
                f"{group_name} must contain {expected_sizes[group_name]} patients; "
                f"found {len(group)}."
            )

        patient_ids = [record.patient_id for record in group]
        if len(patient_ids) != len(set(patient_ids)):
            errors.append(f"{group_name} contains duplicate patient IDs.")
        id_sets[group_name] = set(patient_ids)

        labels = {record.diabetes_label for record in group}
        if labels != {0, 1}:
            errors.append(f"{group_name} must contain both class labels.")

    if id_sets["training"] & id_sets["validation"]:
        errors.append("training and validation groups must not overlap.")
    if id_sets["training"] & id_sets["test"]:
        errors.append("training and test groups must not overlap.")
    if id_sets["validation"] & id_sets["test"]:
        errors.append("validation and test groups must not overlap.")

    split_ids = set().union(*id_sets.values())
    cohort_ids = {record.patient_id for record in records}
    if split_ids != cohort_ids:
        errors.append("The split must include every cohort patient exactly once.")

    if errors:
        raise SplitValidationError("\n".join(errors))
