# Patient JSON store

The backend reads these files at startup. No database server, credentials or
SQL import is required. Restart `python backend/server.py` after changing the
store. The API is read-only; this release does not save training outcomes.

- `patients.json`: 40 fictional English identities and unchanged synthetic
  diabetes values, with unspecified gender (`null`) and source provenance.
- `patient_images.json`: 3,757 CT records and 3,757 reference-mask records.
- `patient_mapping.csv`: the stable identity-to-source mapping.
- `training_runs.json`: an empty list, because no training runs existed in the seed.
- `import_summary.json`: source hashes and cohort counts.
- `migration_report.json`: parity checks and hashes recorded before SQL removal.

The migration compared every patient/image field with the original seed JSON
and regenerated the complete legacy SQL seed for an exact text comparison.
This preserves the repository seed; it does not export later edits from an
external PostgreSQL instance. The SQL scripts are recoverable from Git history.

| Kidney pool | Patients | CT slices | Reference masks |
| --- | ---: | ---: | ---: |
| train | 20 | 2,071 | 2,071 |
| validation | 10 | 1,156 | 1,156 |
| test | 10 | 530 | 530 |

`data/kidney/project40_manifest.csv` is the CT cohort authority. Cases are
sorted and paired with sorted diabetes IDs P001–P040. Names use random seed
9900. Each patient's CT slices are sorted numerically by original `z`; missing
z positions are expected because the manifest contains a selection of slices.
Image paths are relative to the repository root. Original PNGs stay under
`data/kidney/data2d/`, without duplicate image folders or extra windowing.

Diabetes and CT are independent sources paired for teaching. Names and
diabetes ages do not identify the original CT subjects. Kidney views expose
only CT cases and slices, without diabetes measurements or ages. Reference
masks remain available to the existing model code but are not served by the
patient API. Mask annotations are not patient diagnoses.

The diabetes activity reserves P031–P040 for prediction and allows choosing
20 of P001–P030 for training. Its remaining 10 development records are used for
evaluation. Kidney keeps the manifest's fixed train/validation/test pools.

Rebuild generated data with Python 3.10+ from the repository root:

```powershell
python backend/build_patient_data.py
python -m unittest discover -s backend -p "test_patient*.py"
```

Regeneration replaces the generated patient/image/mapping/summary files only;
it preserves the migration audit and training history. `--check` validates
sources without writing files. See [the startup guide](../../docs/JSON_PATIENT_SETUP.md)
and the original [diabetes](../../data/synthetic/README.md) and
[CT documentation](../../data/kidney/README.md), including its data license.
