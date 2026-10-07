# Diabetes Random Forest — Model Card

## Model overview

This is a small Random Forest classifier for an interactive healthcare AI
education activity. Students select 20 fictional patients, the service uses 10
remaining patients for validation and 10 for a final test, and the model returns
a binary label plus an estimated diabetes probability.

The model demonstrates a machine-learning workflow. It is not a medical device,
has not been clinically validated, and must not be used to diagnose, screen,
treat, or advise any real person.

## Intended users and use

- Years 8–12 students learning how data selection affects an AI model
- Workshop facilitators explaining training, validation, testing, and features
- Project developers integrating the model outputs into the education platform

Appropriate outputs include a training curve, a held-out test comparison, and a
global feature-importance visualisation. The probability is an uncalibrated
model score for the activity, not an individual's clinical risk.

## Data

The reference cohort contains 40 entirely synthetic patients: 20 with label 0
and 20 with label 1. No real patient data or personally identifiable information
is used. Each record contains:

1. Age in years
2. Fasting blood glucose in mmol/L
3. HbA1c percentage
4. Body mass index in kg/m²
5. Systolic blood pressure in mmHg
6. Family history of diabetes as 0 or 1

The broad input ranges are data-quality guardrails for fictional records, not
diagnostic thresholds.

## Training and model selection

- Algorithm: `RandomForestClassifier`
- Training records: 20 selected by the student
- Validation records: 10, used to select the number of trees
- Test records: 10, excluded from fitting and model selection
- Tree checkpoints: 5 through 100 in steps of 5
- Maximum depth: 3
- Minimum samples per leaf: 2
- Class weighting: balanced from training labels
- Random seed: 9900

The smallest tree count tied for the best validation accuracy is selected. The
default reproducible demonstration selects 60 trees.

## Demonstration results

For the committed balanced demonstration selection:

| Metric | Result |
| --- | ---: |
| Best validation accuracy | 70% |
| Held-out test accuracy | 90% |
| Balanced accuracy | 90% |
| Precision | 100% |
| Recall | 80% |
| F1 | 88.9% |

These results describe only 10 synthetic test records. They are not a reliable
estimate of real-world performance.

Across 30 deterministic valid student-selection scenarios, test accuracy ranges
from 20% to 100%, with a mean of 75.7% and median of 80%. This wide variation is
expected with such a small cohort and is an important part of the educational
message: data selection can strongly affect model behaviour.

## Reference-model comparison

The same 30 selection scenarios were also evaluated with a scaled,
class-balanced Logistic Regression and a most-frequent-class Dummy baseline:

| Model | Mean accuracy | Mean balanced accuracy | Mean F1 |
| --- | ---: | ---: | ---: |
| Random Forest | 75.7% | 76.8% | 72.5% |
| Logistic Regression | 84.0% | 83.8% | 81.4% |
| Dummy baseline | 43.7% | 50.0% | 18.3% |

Logistic Regression is the strongest predictive reference in this experiment.
Random Forest remains the activity model because incremental tree checkpoints
support the planned training animation and split importance is straightforward
to visualise for students. This is an educational design decision, not a claim
that Random Forest is the best-performing classifier.

## Explainability

The project exports global Random Forest mean-decrease-in-impurity importance.
For the default demonstration, glucose and HbA1c receive the largest importance
values. These values show how this fitted model used features for tree splits.
They do not establish causation, clinical importance, or patient-specific
reasoning, and impurity-based importance can favour features with more possible
split points.

The project also provides a local one-feature-at-a-time sensitivity view for
each test patient. It replaces one value with a training-set median (or mode for
family history) and records how the estimated probability changes. This makes
individual model behaviour easier to discuss, but the effects are not additive,
are affected by correlated features, and must not be interpreted as medical or
causal explanations.

## Limitations and risks

- Forty synthetic records are far too few for clinical model development.
- Student-selected samples can be imbalanced and produce unstable results.
- Test groups contain only 10 records, so one record changes accuracy by about
  ten percentage points.
- The probability is not calibrated against real disease prevalence.
- The synthetic cohort cannot support subgroup fairness or bias claims.
- The model covers clinical tabular data only; it does not process CT or X-ray
  images and is not a multimodal model.
- Results should always be presented with the educational-only disclaimer.

## Reproducibility and handoff

Install the pinned dependencies and regenerate outputs from the repository root:

```bash
.venv/bin/python -m services.diabetes_model.generate_demo_artifacts
```

Verify all generated files against the manifest:

```bash
.venv/bin/python -m services.diabetes_model.artifact_manifest \
  artifacts/diabetes_demo/artifact_manifest.json
```

SHA-256 can detect an accidental or unexpected file change, but the manifest is
not a cryptographic signature. Only load the Joblib model from a trusted project
source because Python model deserialisation can execute unsafe content.
