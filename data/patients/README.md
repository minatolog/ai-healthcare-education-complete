# Forty teaching patients

This dataset contains 40 fictional English names, the unchanged synthetic diabetes
values, and all 3,757 CT slices plus 3,757 reference masks selected by
`../kidney/project40_manifest.csv`. Gender is unspecified (JSON null, SQL NULL,
and an empty CSV cell).

The manifest is the authority for the kidney cohort: group by `case`, retain
every selected slice, and preserve `pool`. Patients are sorted by case ID and
paired with the sorted diabetes IDs `P001` through `P040`. Names are randomly
assigned with seed 9900. CSV row reordering does not change the mapping.

| Kidney pool | Patients | CT slices | Reference masks |
| --- | ---: | ---: | ---: |
| train | 20 | 2,071 | 2,071 |
| validation | 10 | 1,156 | 1,156 |
| test | 10 | 530 | 530 |

## Files

- `patient_mapping.csv`: readable 40-row identity and source mapping.
- `patients.json`: patient identities, diabetes values, labels and provenance.
- `patient_images.json`: all 7,514 image records, paths and source metadata.
- `import_summary.json`: counts and hashes of the two source CSV files.
- `validation_report.json`: generated only after an actual PostgreSQL verification succeeds.

All image paths are relative to the repository root, not this directory.
Original PNG files remain under `data/kidney/data2d/`; no duplicate images or
patient folders are needed. `kidney_ct` records are model inputs and
`kidney_reference_mask` records are reference targets. Source voxel spacing is
not the spacing of resized 256-by-256 PNG pixels. No raw 3D scans are supplied.

Diabetes and CT are independent sources. These are educational composite
identities, not matched real people. The name and diabetes age do not describe
the original CT subject. `label` means only the synthetic diabetes target;
mask `has_tumor` values must not be interpreted as patient diagnoses. Kidney
pools do not override the diabetes activity's existing student-selected splits.
This dataset does not establish a paired multimodal training cohort.

Diabetes source documentation: `../synthetic/README.md`.
CT provenance, preprocessing and licensing: `../kidney/README.md` and
`../kidney/LICENSE_DATA.md`.

Regenerate from the repository root with Python 3.10 or newer:

```powershell
python database/build_patient_seed.py
```

This overwrites only generated files in this directory and
`database/seed_patients.sql`; it does not alter either source dataset or connect
to a database. Run the validation again after regeneration to refresh its report.
For the import procedure, see `../../database/README.md`.
