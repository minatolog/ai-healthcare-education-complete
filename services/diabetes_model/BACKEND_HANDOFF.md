# Diabetes ML Backend Handoff

## Delivery status

The diabetes clinical-data model package is complete for backend integration.
It contains no FastAPI, Node.js, browser, database, or deployment code. All
patient records are fictional, and every model output is for education only.

The current contracts are:

- Delivery version: `1.0.0`
- Training-session schema: `1.3`
- Model artifact schema: `1.0`
- Training/validation/test sizes: `20/10/10`
- Activity model: `RandomForestClassifier`

## Install and verify

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r services/diabetes_model/requirements.txt
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m services.diabetes_model.acceptance
```

The acceptance command checks the dataset, artifact checksums, model loading,
prediction example, training-session response, progress events, evaluation, and
local explanation contract. A non-zero exit code means the delivery should not
be deployed.

## Student-selected online training

The backend receives exactly 20 patient IDs and calls:

```python
from services.diabetes_model import run_training_session

session = run_training_session(selected_patient_ids)
response = session.as_dict()
```

The response includes selection validation, split IDs, model-selection results,
20 curve points, feature importance, held-out test predictions and metrics, and
one local sensitivity explanation per test patient.

### Optional live progress

Pass a synchronous callback to receive one real event per tree checkpoint:

```python
def on_progress(event):
    publish_to_client(event.as_dict())

session = run_training_session(
    selected_patient_ids,
    progress_callback=on_progress,
)
```

The backend may forward these events using SSE or WebSocket. The callback must
return promptly; slow network work should be placed on a queue. Callback errors
are intentionally not hidden and will stop the training request.

## Selection validation and errors

To check a selection before fitting:

```python
from services.diabetes_model import (
    DEFAULT_COHORT_PATH,
    assess_training_selection,
    load_reference_cohort,
)

records = load_reference_cohort(DEFAULT_COHORT_PATH)
assessment = assess_training_selection(records, selected_patient_ids)
payload = assessment.as_dict()
```

`ready: false` prevents training. Stable error codes cover incorrect counts,
duplicates, unknown IDs, single-class training data, and insufficient remaining
class examples. A strongly imbalanced but valid selection returns a warning.
`run_training_session()` performs this pre-flight automatically and raises
`TrainingSelectionError`, whose `as_dict()` method is safe to return as a client
error payload.

## Loading the committed demonstration model

For a single fictional-patient prediction without retraining:

```python
from services.diabetes_model import load_model_bundle, predict_patient

bundle = load_model_bundle(
    "artifacts/diabetes_demo/diabetes_random_forest.joblib"
)
prediction = predict_patient(bundle, request_payload).as_dict()
```

Load the bundle once during backend startup rather than once per request. Never
load an uploaded or untrusted Joblib file. Python model deserialisation can run
unsafe content.

## Important presentation rules

- Say “model-estimated educational score,” not “clinical risk.”
- Do not call global or local importance causal or diagnostic.
- Do not promise that accuracy always rises as trees are added.
- Do not present the default 90% test accuracy as stable performance.
- Explain that different student selections can produce very different results.
- Keep the educational-only disclaimer visible with predictions.

Logistic Regression performed better in the reference benchmark, but Random
Forest remains the activity model for its tree-growth animation and intuitive
split-importance visualisation. This is an educational design decision, not a
claim that Random Forest is the most accurate classifier.

## Main handoff files

- `workflow.py`: complete student-selected training session
- `training.py`: Random Forest fitting and progress callback
- `prediction.py`: validated single-patient inference
- `explanations.py`: local one-feature-at-a-time sensitivity
- `MODEL_CARD.md`: model purpose, metrics, and limitations
- `artifacts/diabetes_demo/`: reproducible examples and fitted model
- `artifact_manifest.json`: delivery file checksums

Any future CT/X-ray or multimodal model is a separate ML workstream and is not
implemented by this diabetes clinical-data package.
