# Diabetes Model Service

This package contains the educational diabetes model pipeline. It works only
with fictional patient data and is not clinically validated.

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, evaluation results,
limitations, and the client/backend handoff summary.

## Current pipeline

1. Load and validate the fixed 40-patient synthetic cohort.
2. Use 20 student-selected patients for training.
3. Split the remaining patients into 10 validation and 10 test patients.
4. Convert each group independently into model-ready features and labels.
5. Add Random Forest trees incrementally and record training/validation accuracy.
6. Select the smallest tree count that achieves the best validation accuracy.
7. Evaluate the selected model once on the locked test group.

## Student-selected training session

The framework-independent `run_training_session()` function is the single
integration point for the future backend. It accepts exactly 20 patient IDs
selected by a student and performs the full split, preprocessing, training,
validation-based selection, feature-importance calculation, and held-out test
evaluation:

```python
from services.diabetes_model import run_training_session

session = run_training_session(selected_patient_ids)
response_json = session.as_dict()
```

The response contains the 20/10/10 patient IDs, graph-ready training and
validation checkpoints, selected model information, global feature importance,
test metrics, and one prediction per test patient. The fitted estimator remains
available as `session.training_result.model` if the backend needs to retain it.

Training is synchronous and runs in Python on the server side. The browser does
not execute the model file. To create a classroom-friendly progress animation,
the frontend can reveal the returned 5-to-100-tree checkpoint history one point
at a time after the session completes. This keeps the displayed curve faithful
to the values actually measured during model training.

## Selection pre-flight check

The backend can validate the student's IDs before starting any model fitting:

```python
from services.diabetes_model import (
    DEFAULT_COHORT_PATH,
    assess_training_selection,
    load_reference_cohort,
)

records = load_reference_cohort(DEFAULT_COHORT_PATH)
assessment = assess_training_selection(records, selected_patient_ids)
response_json = assessment.as_dict()
```

The response contains `ready`, counts, and stable issue codes. Incorrect count,
duplicate IDs, unknown IDs, a single-class training group, or insufficient
remaining examples are errors. A legal but strongly imbalanced training group
is accepted with `TRAINING_CLASS_IMBALANCE` as a warning. `run_training_session()`
performs the same check automatically and raises `TrainingSelectionError` with
the structured assessment before fitting begins.

## Sensitivity to student selection

Because only 20 fictional patients are used for training, the result can change
substantially when students choose different patient cards. The deterministic
`analyse_selection_robustness()` check trains 30 valid selection scenarios and
summarises validation accuracy, held-out test accuracy, balanced accuracy, F1,
selected tree counts, and mean feature importance.

The generated `selection_robustness.json` and `selection_robustness.png` make
this limitation visible instead of presenting the single demo result as a
guaranteed performance level. This is a sensitivity exercise on the same small
synthetic cohort, not independent validation and not evidence of clinical
performance.

## Reference-model benchmark

`benchmark_models()` compares the activity's Random Forest with a scaled,
class-balanced Logistic Regression and a most-frequent-class Dummy baseline on
the same 30 deterministic selection scenarios. The mean test results are:

| Model | Accuracy | Balanced accuracy | F1 |
| --- | ---: | ---: | ---: |
| Random Forest | 75.7% | 76.8% | 72.5% |
| Logistic Regression | 84.0% | 83.8% | 81.4% |
| Dummy baseline | 43.7% | 50.0% | 18.3% |

Logistic Regression is the strongest predictive reference on this synthetic
cohort. Random Forest remains the classroom activity model because adding trees
provides intuitive progress checkpoints and its split importance supports the
planned feature-importance interaction. The project therefore does not claim
that Random Forest is the most accurate algorithm.

## Model input order

The feature order is fixed because every row supplied to the model must use the
same column positions.

| Position | Feature | Display name | Unit |
| --- | --- | --- | --- |
| 1 | `age_years` | Age | years |
| 2 | `glucose_mmol_l` | Fasting blood glucose | mmol/L |
| 3 | `hba1c_percent` | HbA1c | % |
| 4 | `bmi` | Body mass index | kg/m² |
| 5 | `systolic_bp_mmhg` | Systolic blood pressure | mmHg |
| 6 | `family_history` | Family history of diabetes | 0 or 1 |

`diabetes_label` is stored separately as the learning target and is never part
of the input feature matrix.

## Why the values are not scaled

Random Forest uses decision thresholds and does not require standardised input
features. The cohort validation also rejects missing or non-finite values.
Keeping the original units makes the later prediction and feature-importance
explanations easier to connect to the patient cards shown to students.

The preprocessing step is therefore deliberately stateless: it selects the
six approved fields in a fixed order and converts them to numeric values. It
does not calculate any statistics from validation or test patients.

## Random Forest training

The model is intentionally small and deterministic for a 20-patient classroom
training set:

- Maximum tree depth: `3`
- Minimum samples per leaf: `2`
- Class weighting: `balanced`, calculated once from the training labels
- Random seed: `9900`
- Progress checkpoints: `5, 10, ..., 100` trees

Training uses `warm_start=True` to add trees at each checkpoint. After every
checkpoint, the service records training and validation accuracy. The smallest
tree count tied for the best validation accuracy is selected, then a clean
model with that tree count is fitted using the training patients only.

The horizontal axis is **trees built**, not neural-network epochs. Accuracy can
move up or down slightly as trees are added; the history reports the genuine
result rather than forcing a visually increasing curve. Random Forest does not
provide an epoch loss, so the graph-ready history returns `loss: null`.

The training function accepts only training and validation datasets. It cannot
receive the test dataset, which keeps the test patients out of model selection.

## Final test evaluation and curve

`evaluate_test_set()` checks that all ten test patient IDs are unseen during
training and validation, then reports accuracy, balanced accuracy, precision,
recall, F1, a confusion matrix, and one model-estimated probability per patient.
The probability is an educational model score, not a calibrated clinical risk.

The demo artifact command uses a deterministic balanced 20-patient selection:

```bash
.venv/bin/python -m services.diabetes_model.generate_demo_artifacts
```

It writes an English training curve, training history JSON, test evaluation
JSON, feature-importance JSON/PNG, the fitted `.joblib` model with metadata,
the exact demo split, and a backend prediction example to
`artifacts/diabetes_demo/`. These files are a reproducible example; a student's
different 20-patient selection will produce different validation and test
results.

## Exported model and prediction contract

`diabetes_random_forest.joblib` contains the fitted Random Forest and its fixed
feature-order contract. `diabetes_random_forest.metadata.json` provides the
same contract in a safe, human-readable format without serialising an
estimator. The backend should load the model once when its service starts, not
reload it for every request.

Only load `.joblib` files produced and controlled by this project. Joblib model
files use Python's pickle mechanism and must not be loaded from an untrusted
source.

The backend-facing request contains one fictional patient and no known label:

```json
{
  "patient_id": "P021",
  "age_years": 35,
  "glucose_mmol_l": 6.9,
  "hba1c_percent": 6.0,
  "bmi": 26.4,
  "systolic_bp_mmhg": 119,
  "family_history": 1
}
```

Load the artifact and make a prediction with:

```python
from services.diabetes_model import load_model_bundle, predict_patient

bundle = load_model_bundle("artifacts/diabetes_demo/diabetes_random_forest.joblib")
result = predict_patient(bundle, request_json).as_dict()
```

The response includes `predicted_label` and
`estimated_diabetes_probability`. That probability is an educational model
score, not a calibrated clinical risk. Input fields are validated against the
fixed schema and broad synthetic-data ranges before prediction.

## Feature importance

The exported global feature importance is the Random Forest's mean decrease in
impurity (split importance), normalised across all six input features. It shows
which fields the fitted demo model used most when splitting its tiny synthetic
training sample. It does **not** establish causation, medical importance, or a
diagnostic threshold, and it should not be used as patient-specific reasoning.

## Install the ML dependency

The project pins scikit-learn, Matplotlib, and Joblib for reproducible model
behaviour and artifacts:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r services/diabetes_model/requirements.txt
```

## Run the checks

From the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Verify generated artifacts

Regeneration writes `artifact_manifest.json` with each delivery file's byte
size and SHA-256 checksum. Verify the files with:

```bash
.venv/bin/python -m services.diabetes_model.artifact_manifest \
  artifacts/diabetes_demo/artifact_manifest.json
```

This detects file changes or transfer damage. Because the manifest is stored
beside the artifacts, it is an integrity check rather than a digital signature.
