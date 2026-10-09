# Kidney Tumour CT Data Handoff

This folder contains the **actual processed CT images and reference masks** for the project's fixed 40-patient Kidney module. These are real KiTS19 data, separate from the Diabetes synthetic clinical dataset. The files are copied without changing their names or pixels.

| Pool | Patients | Selected CT/mask pairs | Use |
| --- | ---: | ---: | --- |
| Train | 20 | 2,071 | Fitting and development |
| Validation | 10 | 1,156 | Model selection before freezing |
| Test | 10 | 530 | Completed frozen evaluation; demo patient picker |
| Total | 40 | 3,757 | 7,514 PNG files; 87,913,810 PNG bytes |

All slices of a patient have the same pool. `case_00000` was used during development and remains in training. For the current demo, select one of the ten test patients for CT-only inference; no extra five-patient pool is required. The reference mask is a training target or evaluation reference, never an input channel or a patient diagnosis.

## Files and responsibilities

```text
data/kidney/
  README.md
  LICENSE_DATA.md
  dataset_info.json             Package counts, preprocessing and provenance
  patient_splits.csv            Forty patient IDs and their fixed roles
  project40_manifest.csv        Only the 3,757 packaged slice pairs, plus pool
  checksums.sha256              File integrity catalogue
  requirements-data.txt
  validate_handoff.py
  test_validate_handoff.py
  data2d/
    images/case_XXXXX_zNNN.png  One grayscale CT slice per file
    masks/case_XXXXX_zNNN.png   Matching reference class-ID mask
    manifest.csv               Unchanged 210-case source manifest
    splits.json                Unchanged legacy 147/32/31 split; NOT project split
    dataset_stats.json         Unchanged original 210-case statistics
  provenance/
    project40_splits.json      Original immutable cohort lock (includes legacy5)
    png_data_audit.json        Original complete 210-case machine audit
    source_processing/
      preprocess.py            Historical exporter snapshot, not a new command
      config.py                Historical exporter configuration snapshot
```

`project40_manifest.csv` is the processing teammate's working index. The `image` and `mask` paths are relative to **`data/kidney/data2d/`**, not the repository root. It has the original CSV columns plus `pool`.

The local `.gitattributes` preserves exact file bytes across operating systems, including the historical source snapshots' CRLF line endings. Keep it with the checksum catalogue so Git does not silently change the bound metadata on checkout.

`data2d/manifest.csv` deliberately retains all 20,686 records/210 cases. The current model binds its original bytes to the saved audit and checkpoint. **Only 40 patients' PNGs are packaged.** Do not loop over the entire original CSV expecting every other patient's PNG to exist. Do not replace this CSV with the filtered table or overwrite the historical lock.

The original lock records five earlier prediction patients. It is preserved byte for byte as model provenance; their PNGs are omitted from this handoff. The historical Step 08 command targeting those five patients therefore needs the separate original local data. The current test-patient demo uses the ten packaged test patients. The old `splits.json` is provenance and must not determine the project's 20/10/10 assignments.

## Check after pulling the branch

Run from the repository root in an environment with Python 3.11 or later:

```sh
python3 -m pip install -r data/kidney/requirements-data.txt
python3 data/kidney/validate_handoff.py --metadata-only
python3 data/kidney/validate_handoff.py
python3 -m unittest discover -s data/kidney -p 'test_validate_handoff.py' -v
```

The successful full check reports 40 patients, 3,757 CT/mask pairs and pools 20/10/10. It checks file hashes, exact PNG inventories, dimensions/mode, class IDs, mask pixel counts and the patient/source bindings. It runs no model or training and writes no output files. Metadata-only mode does not open PNG bytes. Machine checks establish file integrity, not anatomical annotation quality.

The CI `kidney_data_handoff_check` job runs the same tests and full check. No raw scans, pretrained weights or GPU are required.

## Process or display one patient

Use the filtered table and retain the explicit pool:

```python
from pathlib import Path
import csv
from PIL import Image

base = Path('data/kidney')
with (base / 'project40_manifest.csv').open(newline='') as handle:
    rows = list(csv.DictReader(handle))
training_rows = [row for row in rows if row['pool'] == 'train']
row = training_rows[0]
ct = Image.open(base / 'data2d' / row['image'])
reference = Image.open(base / 'data2d' / row['mask'])
print(row['case'], row['z'], ct.mode, ct.size, reference.mode)
```

Both PNGs must be `L` mode and 256x256. A mask may look almost black because its stored values are only 0, 1 and 2. Create a separate display overlay if needed; preserve the class-ID reference mask unchanged. Model-generated masks/overlays belong in a separate output directory and must not replace `data2d/masks/`.

For a CT-only demo patient picker, filter `patient_splits.csv` by `pool == 'test'` (or `demo_prediction_eligible == 'true'`). Inference reads the chosen patient's CT paths only. Do not use `has_tumor` or reference-mask pixels as model inputs or silently rank patients by their test score.

## Existing Kidney model compatibility

The existing package can still read the original full manifest and filter explicit patient IDs. In the model environment, this read-only preflight demonstrates the saved bindings without constructing a model or reading PNG pixels:

```sh
PYTHONPATH=backend/kidney python3 - <<'PY'
from kidney_segmentation.png_training import load_training_context, AuditedPNGDataset
root = 'data/kidney'
context = load_training_context(
    root, root + '/provenance/project40_splits.json',
    root + '/provenance/png_data_audit.json')
print('train', len(AuditedPNGDataset(context, 'train')))
print('validation', len(AuditedPNGDataset(context, 'validation')))
PY
```

Expected counts: `train 2071`, `validation 1156`. Actual training additionally needs the existing reviewed training-reference record and pinned offline encoder cache; they are not supplied by this dataset handoff. The trained checkpoint and frozen evaluation remain separate artifacts. Do not rerun the full 210-case cohort audit against this subset as if it were a complete cache, or regenerate the existing lock from the filtered CSV.

## Fields, preprocessing and scope

| Field | Meaning |
| --- | --- |
| `case` | Original case ID; patient-level split key |
| `z` | Original axial slice index; gaps are expected after selection |
| `image`, `mask` | Relative paths under `data2d/`; same case/z filename |
| `has_kidney`, `has_tumor` | Reference mask presence flags; not patient diagnoses |
| `kidney_px`, `tumor_px` | Class-1/class-2 pixel counts in the processed mask |
| `n_slices` | Original scan's slice count; not the selected PNG count |
| `sp_z`, `sp_y`, `sp_x` | Original NIfTI voxel spacing; not resized PNG spacing |
| `pool` | Fixed project role; added only to the project40 working index |

Label IDs are **0 background, 1 kidney, 2 tumour**. The CSV retains the original American-spelling field `has_tumor`; project display text uses `tumour`.

The historical exporter clips CT HU to [-79, 304], converts to uint8, uses axial slices, keeps all foreground slices plus a random 15% of background slices, and resizes CT/masks to 256x256 with bilinear/nearest interpolation respectively. The included source snapshots document that pipeline; this handoff does not rerun it or establish an exact raw-to-PNG replay. Their historical path configuration is not the handoff's runtime configuration.

For the current pure 2D model, the PNG already contains the processed intensity. Use one channel and normalize `(PNG / 255.0 - 0.456) / 0.224`. Do not apply HU windowing again or add neighbouring slices. Reference masks are separate integer targets.

These exported slices were selected with reference labels. Describe results as **selected exported-slice segmentation**, not blind complete-scan detection or normal-versus-tumour patient classification. The ten-patient final test is already completed for the frozen run05 model; use training/validation for development and preserve its existing test report.

## Changing the data

Keep this handoff as the reproducible baseline. If new processing, resizing or augmentation is needed, use a new version/output folder, record the change and preserve original case IDs/pools. Changes to PNG pixels, source CSV, audit or lock invalidate the current hashes and cannot silently reuse the frozen checkpoint's original evidence. Never randomly split slices across patients' pools.

Raw NIfTI scans, the other 170 cases, clinical records, model weights, environments, training logs and generated predictions are not included. Source attribution and redistribution terms are in [LICENSE_DATA.md](LICENSE_DATA.md).
