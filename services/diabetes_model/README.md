# Diabetes Model Service

This package contains the educational diabetes model pipeline. It works only
with fictional patient data and is not clinically validated.

## Current pipeline

1. Load and validate the fixed 40-patient synthetic cohort.
2. Use 20 student-selected patients for training.
3. Split the remaining patients into 10 validation and 10 test patients.
4. Convert each group independently into model-ready features and labels.
5. Add Random Forest trees incrementally and record training/validation accuracy.
6. Select the smallest tree count that achieves the best validation accuracy.
7. Evaluate the selected model on the locked test group in a later stage.

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

## Install the ML dependency

The project currently pins scikit-learn for reproducible model behaviour:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r services/diabetes_model/requirements.txt
```

## Run the checks

From the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```
