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
    validate_reference_cohort(cohort)

    if isinstance(training_patient_ids, (str, bytes)):
        raise SplitValidationError(
            "training_patient_ids must be a collection of patient IDs."
        )

    selected_ids = list(training_patient_ids)
    if len(selected_ids) != TRAINING_SIZE:
        raise SplitValidationError(
            f"Exactly {TRAINING_SIZE} training patients are required; "
            f"received {len(selected_ids)}."
        )
    if any(type(patient_id) is not str for patient_id in selected_ids):
        raise SplitValidationError("Every training patient ID must be a string.")
    if len(set(selected_ids)) != len(selected_ids):
        raise SplitValidationError("Training patient IDs must be unique.")

    records_by_id = {record.patient_id: record for record in cohort}
    unknown_ids = sorted(set(selected_ids) - set(records_by_id))
    if unknown_ids:
        raise SplitValidationError(
            f"Unknown training patient IDs: {', '.join(unknown_ids)}."
        )

    selected_id_set = set(selected_ids)
    training = sorted(
        (records_by_id[patient_id] for patient_id in selected_id_set),
        key=lambda record: record.patient_id,
    )
    remaining = [
        record for record in cohort if record.patient_id not in selected_id_set
    ]

    training_counts = Counter(record.diabetes_label for record in training)
    if set(training_counts) != {0, 1}:
        raise SplitValidationError(
            "The training selection must contain both synthetic class labels."
        )

    remaining_counts = Counter(record.diabetes_label for record in remaining)
    if any(remaining_counts[label] < 2 for label in (0, 1)):
        raise SplitValidationError(
            "Choose a different training set: the remaining patients must include "
            "at least two examples of each label so validation and test can both "
            "contain both labels."
        )

    validation, test = _split_remaining_patients(remaining, seed)
    patient_split = PatientSplit(
        training=tuple(training),
        validation=tuple(validation),
        test=tuple(test),
    )
    validate_patient_split(cohort, patient_split)
    return patient_split


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
