"""Generate reproducible demo curve and final evaluation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_test_set
from .features import prepare_model_split
from .schema import load_reference_cohort
from .split import create_student_split
from .training import train_random_forest
from .visualization import plot_training_history


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPOSITORY_ROOT / "data" / "synthetic" / "diabetes_patients.csv"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "artifacts" / "diabetes_demo"


def generate_demo_artifacts(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, Path]:
    """Run the deterministic balanced demo selection and write its outputs."""

    records = load_reference_cohort(DEFAULT_DATASET)
    label_zero_ids = [
        record.patient_id for record in records if record.diabetes_label == 0
    ][:10]
    label_one_ids = [
        record.patient_id for record in records if record.diabetes_label == 1
    ][:10]
    patient_split = create_student_split(records, label_zero_ids + label_one_ids)
    prepared = prepare_model_split(patient_split)
    training_result = train_random_forest(prepared.training, prepared.validation)
    evaluation_result = evaluate_test_set(training_result, prepared.test)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    curve_path = plot_training_history(
        training_result, output_path / "training_accuracy_curve.png"
    )
    training_path = output_path / "training_summary.json"
    evaluation_path = output_path / "test_evaluation.json"
    split_path = output_path / "patient_split.json"

    _write_json(training_path, training_result.summary_as_dict())
    _write_json(evaluation_path, evaluation_result.as_dict())
    _write_json(split_path, patient_split.as_id_dict())

    return {
        "curve": curve_path,
        "training_summary": training_path,
        "test_evaluation": evaluation_path,
        "patient_split": split_path,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the diabetes model's reproducible demo artifacts."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    args = parser.parse_args()

    outputs = generate_demo_artifacts(args.output_directory)
    for output_name, output_path in outputs.items():
        print(f"{output_name}: {output_path}")


if __name__ == "__main__":
    main()
