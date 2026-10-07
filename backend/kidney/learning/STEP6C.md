# Step 6C: Inspect Training and Validation Predictions

This entry loads a completed Step 6B PNG checkpoint and produces diagnostic CT/reference/prediction comparisons. It makes no optimizer updates. The complete Step 6B validation report remains the source of validation metrics.

Each of the 20 training and 10 validation patients contributes two fixed examples: maximum reference tumour area (or maximum kidney area when no tumour is present), then a distinct exported slice near the median original axial index with reference kidney pixels. Ties use the lower axial index. Missing distinct kidney candidates fall back to other exported slices.

**Reference labels guide selection. These are diagnostic examples, not blind inference sampling or representative performance results.** The script never constructs a test or independent-prediction pixel dataset.

## Existing Starter environment and evidence

When learning locally, use the unchanged original Starter environment and evidence. From that Starter directory:

```sh
source .venv/bin/activate
python learning/06c_review_predictions.py --root ../kits_project --check-only
python learning/06c_review_predictions.py --root ../kits_project
```

The first command verifies current cohort/audit/manifest/split bindings and the completed checkpoint metadata and tensor fingerprints. It neither initializes a model nor reads PNG pixels or writes output. The second command restores the full saved model without downloading pretrained weights and generates 60 selected-pair predictions.

## Repository layout

When using the team module, run from `backend/kidney` with an environment containing the module dependencies. Provide absolute data/evidence/output paths; the repository does not carry the original Starter outputs or environment:

```sh
python learning/06c_review_predictions.py \
  --root /absolute/path/to/kits_project \
  --cohort /absolute/path/to/starter/learning_outputs/step5/project40_splits.json \
  --audit /absolute/path/to/starter/learning_outputs/step5/png_data_audit.json \
  --run-dir /absolute/path/to/starter/learning_outputs/step6b/run01 \
  --output-dir /absolute/path/to/starter/learning_outputs/step6c/run01
```

The default run directory is `learning_outputs/step6b/run01`; the default output is `learning_outputs/step6c/run01`. Existing output directories are rejected. Keep prior evidence and choose a new review directory for another review. Outputs cannot be placed inside the original data or training run directories.

## Outputs and interpretation

Expected completion:

```text
[PREDICTION REVIEW COMPLETED]
Decoded pairs: train=40; validation=20; test=0; prediction=0
```

Every `train/<patient>` or `validation/<patient>` directory contains:

- `prediction_comparison.png`: two rows, with CT, reference overlay and model-predicted overlay as separate columns.
- `zNNN_predicted_mask.png`: uncoloured model argmax labels, 0 background / 1 kidney / 2 tumour. These low-valued PNGs may look black in an ordinary image viewer.

Green indicates kidney and red indicates tumour. The overlay is segmentation output; it is not Grad-CAM, physical tumour volume, a confidence score or a patient disease diagnosis. Already processed PNG pixels are not HU-windowed or resized again.

`predictions_review.json` records checkpoint/source bindings, fixed selection reasons, decoded-pair counts, per-slice reference/prediction pixel counts, output hashes and completion status. It intentionally reports no aggregate performance score for these selected examples. Model weights/state are checked for no change after inference. An interrupted or failed directory must not be described as completed.

Open the patient images at full size. Inspect missed red regions, misplaced regions, kidney/tumour confusion and CT/reference alignment. Read both successful and failed examples. Any human observation remains separate from the automatic `manual_review_status=pending` field; do not convert it into clinical certification.

The complete run01 validation report found tumour true positives only in `case_00108`; `case_00010` is among patients with no tumour-pixel predictions. These are examples for examining a known result, not evidence selected to claim overall performance. Keep final-test and new-patient pools untouched while investigating and tuning.
