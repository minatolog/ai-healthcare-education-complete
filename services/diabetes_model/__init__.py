"""Educational diabetes model package."""

from .features import (
    FEATURE_DEFINITIONS,
    MODEL_FEATURES,
    FeatureDefinition,
    FeaturePreparationError,
    PreparedDataset,
    PreparedModelSplit,
    prepare_dataset,
    prepare_model_split,
)
from .schema import (
    CSV_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DatasetValidationError,
    PatientRecord,
    load_reference_cohort,
    validate_reference_cohort,
)
from .split import (
    DEFAULT_SPLIT_SEED,
    TEST_SIZE,
    TRAINING_SIZE,
    VALIDATION_SIZE,
    PatientSplit,
    SplitValidationError,
    create_student_split,
    validate_patient_split,
)

__all__ = [
    "FEATURE_DEFINITIONS",
    "MODEL_FEATURES",
    "FeatureDefinition",
    "FeaturePreparationError",
    "PreparedDataset",
    "PreparedModelSplit",
    "prepare_dataset",
    "prepare_model_split",
    "CSV_COLUMNS",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "DatasetValidationError",
    "PatientRecord",
    "load_reference_cohort",
    "validate_reference_cohort",
    "DEFAULT_SPLIT_SEED",
    "TEST_SIZE",
    "TRAINING_SIZE",
    "VALIDATION_SIZE",
    "PatientSplit",
    "SplitValidationError",
    "create_student_split",
    "validate_patient_split",
]
