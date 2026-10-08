# Step 8: CT-Only Inference on the Independent Prediction Pool

Run05 epoch 10 was frozen before final test. Step 7B completed evaluation of all 530 exported slices from the ten reserved test patients. Patient-mean kidney Dice was 0.876164; tumour Dice 0.368032, IoU 0.272769, precision 0.524337 and recall 0.378970. Tumour precision has eight contributors because two patients predicted no tumour pixels. Four test patients have zero tumour Dice. These results describe the already selected model; they do not select a new checkpoint, training configuration or threshold.

Step 8 uses the same frozen checkpoint to produce segmentation outputs on the five separate prediction patients (676 exported CT PNGs). This pool is separate from training, validation and final test, but comes from the same KiTS19 dataset and is not external clinical validation. Its images were originally exported using reference-based slice selection; CT-only inference does not undo that prior selection policy.

## What runs

The CT-only dataset reads patient/slice identity and image paths from the bound manifest. It validates membership/counts against the prediction pool and image identities/byte/pixel hashes against the saved audit. Input remains one processed grayscale 256x256 PNG with `(PNG/255 - 0.456)/0.224` normalization. No HU windowing, neighbouring slices, augmentation or resizing is added.

Reference-mask PNGs are not checked for existence, opened or decoded in this step. Reference label fields do not select slices or preview examples. All prediction CT rows run once; the model receives CT only. The bound legacy manifest/audit still contain reference metadata for provenance, so the claim is no reference-mask pixel access or label-guided inference selection, not removal of reference metadata from the project.

The frozen model is restored in evaluation/inference mode with no optimizer. Hard argmax assigns 0 background, 1 kidney or 2 tumour at each pixel. The full saved parameter/buffer fingerprint and all protected bindings must remain unchanged. The completed final-test receipt is checked by its hashes and completion state; test scores never influence inference.

## Run from the original Starter

Keep the original run05, frozen record and completed final-test evidence in place. No additional dependency installation is needed in the existing environment. The output directory must be new.

```sh
cd /Users/realnin/9900/Kidney_Segmentation_Demo_Starter
source .venv/bin/activate

python learning/08_predict_new_patients.py --root ../kits_project --check-only && \
python learning/08_predict_new_patients.py --root ../kits_project
```

The preflight checks metadata, saved tensors, patient/image paths and completed final-test receipts, without initializing a model, decoding CT PNGs or writing outputs. The actual run uses batch size 2 and two CPU threads (338 inference batches). It prints periodic progress and finishes with:

```text
[FROZEN CT-ONLY PREDICTION COMPLETED]
Patients=5; CT slices=676; reference-mask PNGs read=0
Prediction assets: .../learning_outputs/step8/run05_prediction
Report: .../learning_outputs/step8/run05_prediction/prediction_report.json
```

No Dice, accuracy, reference comparison or patient disease label is produced. A model-generated red region indicates a predicted tumour-class region, with the misses/false positives documented in validation/test results. It is a segmentation output rather than Grad-CAM or a calibrated clinical confidence score.

## Outputs and code responsibilities

For every CT slice, the run saves an unchanged-class-ID predicted label PNG and a model overlay (green kidney, red tumour). Every patient also has a preview containing CT and model overlays for fixed first/middle exported slices. Preview selection depends only on patient/slice order, and does not choose the largest reference or predicted tumour region. It can legitimately show no predicted tumour on a chosen slice.

`prediction_report.json` records every patient/slice, source image hash, predicted class pixel counts, output paths/hashes, selected preview sample identities, model/freeze fingerprints and no-update/no-mask-read scope. Counts represent model outputs, not measured anatomical tumour volume or patient-level disease probability. `prediction_summary.json` records execution state and completed report hash; `progress.jsonl` preserves progress if the terminal scrolls or execution is interrupted. Paths for image assets are supplied by the report rather than manually inferred filenames.

| File | Responsibility |
| --- | --- |
| `kidney_segmentation/png_prediction.py` | CT-only audited loading, frozen inference, masks/overlays/previews and provenance report |
| `learning/08_predict_new_patients.py` | Portable command options, completed-test receipt checks, progress and failure status |
| `learning/STEP8.md` | Reproducible workflow and interpretation |

If execution fails, preserve partial output and its failed/interrupted status, and inspect the error. No missing slice is silently skipped. Source data, weights, freeze and final-test records are never overwritten.

From the team module `backend/kidney`, supply absolute external `--root`, `--cohort`, `--audit`, `--run-dir`, `--freeze` and `--test-dir` paths. The sibling defaults above belong to the original Starter. Outputs, environments and weights stay outside Git.

## Following work

Review the saved CT/model previews as model outputs, preserve this frozen result, then prepare the next source MR and the shared backend integration. The artefacts can later be served by the API for frontend rendering. This step is a backend inference component; it does not add a website, upload workflow, shared HTTP contract or Diabetes adapter, and it does not close their Jira acceptance criteria.
