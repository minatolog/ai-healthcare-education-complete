# Diabetes Model Service

This package contains the educational diabetes model pipeline. It works only
with fictional patient data and is not clinically validated.

## Current pipeline

1. Load and validate the fixed 40-patient synthetic cohort.
2. Use 20 student-selected patients for training.
3. Split the remaining patients into 10 validation and 10 test patients.
4. Convert each group independently into model-ready features and labels.
5. Train and evaluate the Random Forest model in later stages.

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

## Run the checks

From the repository root:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
