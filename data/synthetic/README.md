# Synthetic Diabetes Cohort

`diabetes_patients.csv` is a reproducible, fictional cohort created only for
the COMP9900 healthcare AI learning activity. It contains 40 patients: 20 with
label `0` and 20 with label `1`.

No row represents a real person. The data and future predictions must not be
used for diagnosis, treatment, or any other clinical decision.

The deliberate 50/50 class balance supports a small classroom activity. It
does not represent diabetes prevalence. The class labels, feature averages,
feature spreads, and family-history probabilities are all synthetic design
choices; they are not estimates of real individual risk or clinically
calibrated relationships.

## Data dictionary

| Column | Type | Meaning |
| --- | --- | --- |
| `patient_id` | string | Anonymous fictional ID |
| `age_years` | integer | Age in years |
| `glucose_mmol_l` | float | Simplified fasting blood glucose in mmol/L |
| `hba1c_percent` | float | HbA1c percentage, representing longer-term glucose level |
| `bmi` | float | Body mass index in kg/m² |
| `systolic_bp_mmhg` | integer | Systolic blood pressure in mmHg |
| `family_history` | binary | `1` if a family history of diabetes is assigned, otherwise `0` |
| `diabetes_label` | binary | Synthetic learning target: `1` or `0` |

The generator intentionally creates overlapping class distributions so the
future model learns patterns rather than a single hard-coded rule. The broad
validation ranges only catch data-entry mistakes; they are not diagnostic
criteria.

For context, Diabetes Australia describes HbA1c of 6.5% or more and fasting
blood glucose of 7.0 mmol/L or more as values at which diabetes is likely. A
real diagnosis requires appropriate clinical assessment and may require repeat
testing:

- https://www.diabetesaustralia.com.au/about-diabetes/what-is-diabetes/
- https://www.cdc.gov/diabetes/diabetes-testing/index.html

## Reproduce and validate

From the repository root:

```bash
python3 -m services.diabetes_model.generate_synthetic_cohort
python3 -m unittest discover -s tests -p 'test_*.py'
```

The default random seed is `9900`. The committed CSV is the reference file used
by the project and is checked against the generator by the automated tests.

The split into 20 training, 10 validation, and 10 test patients will be added
in a separate change. This CSV is currently the complete selection pool.
