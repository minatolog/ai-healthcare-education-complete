"""Generate reproducible demo curve and final evaluation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_manifest import create_artifact_manifest
from .benchmark import benchmark_models
from .model_artifact import (
    FEATURE_IMPORTANCE_NOTE,
    export_model_bundle,
    extract_feature_importance,
    load_model_bundle,
)
from .prediction import PredictionInput, predict_patient
from .robustness import analyse_selection_robustness
from .schema import load_reference_cohort
from .split import assess_training_selection
from .visualization import (
    plot_feature_importance,
    plot_model_benchmark,
    plot_patient_explanation,
    plot_selection_robustness,
    plot_training_history,
)
from .workflow import DEFAULT_COHORT_PATH, run_training_session


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = DEFAULT_COHORT_PATH
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
    session = run_training_session(label_zero_ids + label_one_ids)
    patient_split = session.patient_split
    training_result = session.training_result
    evaluation_result = session.evaluation_result

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    curve_path = plot_training_history(
        training_result, output_path / "training_accuracy_curve.png"
    )
    training_path = output_path / "training_summary.json"
    evaluation_path = output_path / "test_evaluation.json"
    split_path = output_path / "patient_split.json"
    model_paths = export_model_bundle(
        training_result, output_path / "diabetes_random_forest.joblib"
    )
    importance = extract_feature_importance(training_result)
    importance_path = output_path / "feature_importance.json"
    importance_chart_path = plot_feature_importance(
        importance, output_path / "feature_importance.png"
    )

    example_patient = patient_split.test[0]
    example_input = PredictionInput(
        patient_id=example_patient.patient_id,
        age_years=float(example_patient.age_years),
        glucose_mmol_l=float(example_patient.glucose_mmol_l),
        hba1c_percent=float(example_patient.hba1c_percent),
        bmi=float(example_patient.bmi),
        systolic_bp_mmhg=float(example_patient.systolic_bp_mmhg),
        family_history=example_patient.family_history,
    )
    loaded_bundle = load_model_bundle(model_paths.model)
    example_prediction = predict_patient(loaded_bundle, example_input.as_dict())
    prediction_example_path = output_path / "backend_prediction_example.json"
    session_path = output_path / "training_session_response.json"
    robustness = analyse_selection_robustness()
    robustness_path = output_path / "selection_robustness.json"
    robustness_chart_path = plot_selection_robustness(
        robustness, output_path / "selection_robustness.png"
    )
    benchmark = benchmark_models(robustness_report=robustness)
    benchmark_path = output_path / "model_benchmark.json"
    benchmark_chart_path = plot_model_benchmark(
        benchmark, output_path / "model_benchmark.png"
    )
    assessment_examples_path = output_path / "selection_assessment_examples.json"
    explanation = session.test_explanations[0]
    explanation_path = output_path / "patient_explanation_example.json"
    explanation_chart_path = plot_patient_explanation(
        explanation, output_path / "patient_explanation_example.png"
    )

    _write_json(training_path, training_result.summary_as_dict())
    _write_json(evaluation_path, evaluation_result.as_dict())
    _write_json(split_path, patient_split.as_id_dict())
    _write_json(
        importance_path,
        {
            "model_type": "RandomForestClassifier",
            "importance_type": "global_split_importance",
            "features": [entry.as_dict() for entry in importance],
            "note": FEATURE_IMPORTANCE_NOTE,
        },
    )
    _write_json(
        prediction_example_path,
        {
            "request": example_input.as_dict(),
            "response": example_prediction.as_dict(),
        },
    )
    _write_json(session_path, session.as_dict())
    _write_json(robustness_path, robustness.as_dict())
    _write_json(benchmark_path, benchmark.as_dict())
    imbalanced_ids = label_zero_ids + [
        record.patient_id
        for record in records
        if record.diabetes_label == 0
        and record.patient_id not in set(label_zero_ids)
    ][:8] + label_one_ids[:2]
    _write_json(
        assessment_examples_path,
        {
            "valid_balanced": session.selection_assessment.as_dict(),
            "valid_but_imbalanced": assess_training_selection(
                records, imbalanced_ids
            ).as_dict(),
            "invalid_count": assess_training_selection(
                records, label_zero_ids + label_one_ids[:-1]
            ).as_dict(),
            "invalid_duplicate": assess_training_selection(
                records,
                label_zero_ids + label_one_ids[:-1] + [label_zero_ids[0]],
            ).as_dict(),
        },
    )
    _write_json(explanation_path, explanation.as_dict())

    outputs = {
        "curve": curve_path,
        "training_summary": training_path,
        "test_evaluation": evaluation_path,
        "patient_split": split_path,
        "model": model_paths.model,
        "model_metadata": model_paths.metadata,
        "feature_importance": importance_path,
        "feature_importance_chart": importance_chart_path,
        "backend_prediction_example": prediction_example_path,
        "training_session_response": session_path,
        "selection_robustness": robustness_path,
        "selection_robustness_chart": robustness_chart_path,
        "model_benchmark": benchmark_path,
        "model_benchmark_chart": benchmark_chart_path,
        "selection_assessment_examples": assessment_examples_path,
        "patient_explanation_example": explanation_path,
        "patient_explanation_chart": explanation_chart_path,
    }
    manifest_path = create_artifact_manifest(
        outputs, output_path / "artifact_manifest.json"
    )
    outputs["artifact_manifest"] = manifest_path
    return outputs


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
