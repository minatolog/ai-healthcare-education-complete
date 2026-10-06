# Step 6A: Review Training CT and Reference-Mask Alignment

> Feature-branch layout: run commands from `backend/kidney` (use `cd backend/kidney` from the repository root). Replace `/absolute/path/to/kits_project` with your own absolute data path. Prepare the branch environment using [the module README](../README.md); environments, data, caches and generated results are not included in Git. The original local starter is unchanged.

Step 5 passed the machine audit and locked the patient cohort. Before running the formal PNG training entry, this step creates reference previews for **the 20 locked training patients only**. It helps you check that a CT slice and its existing annotation appear in the same position and carry the correct patient/slice identity.

This is a visual data-quality exercise. It does not run a model, update weights or measure segmentation performance. It does not open validation, final-test or independent-prediction image/mask pixels. The existing cohort lock and original data remain unchanged.

## Run in the feature-branch checkout

Prepare the module environment using its README. Select `backend/kidney/.venv/bin/python` in VS Code. The script is included in this module; run from the repository root:

```sh
cd backend/kidney
source .venv/bin/activate
python learning/06a_review_training_pairs.py --root /absolute/path/to/kits_project --cohort learning_outputs/step5/project40_splits.json --audit learning_outputs/step5/png_data_audit.json
```

The default output directory is `learning_outputs/step6a`. Keep these generated previews outside `kits_project` and outside GitLab, alongside the other local learning outputs.

The command consumes the existing Step 5 audit and cohort lock. It must use their reviewed bindings, rather than generating another split or changing patient assignments. If it reports a source, fingerprint or identity mismatch, stop and resolve that specific error before training. Do not delete the lock merely to make a check pass.

## Which slices are shown?

The default review shows **two representative slices for each of the 20 training patients: 40 slices in total**. The first has the largest reference tumour area; ties use the lower original slice index. If no tumour is annotated in that patient's exported slices, it uses the largest kidney area and records that fallback. The second is a different foreground slice near the opposite axial quartile, using actual original slice indices. If no second foreground slice exists, it records a distinct background-context fallback. Fewer than two exported slices causes an explicit error.

This selection uses existing ground-truth/reference information **for training-data QA**. It is not an evaluation or inference slice-selection rule. An unseen patient's reference tumour mask would not be available to select slices at prediction time. Do not use this preview procedure to claim blind full-scan results or to choose favourable validation/test examples.

All reviewed slices stay within the already locked training pool. No patient's other slices may move into another pool because this inspection was useful.

## Understand the colours and reference displays

The previews show the processed grayscale CT and reference regions aligned to that same image. The reference overlay places the existing mask over the CT:

| Display value | Meaning |
| --- | --- |
| Grayscale | Processed CT brightness |
| Green | Reference class 1: kidney |
| Red | Reference class 2: tumour |
| Background/class 0 | No coloured foreground annotation |

These colours come from the reference annotation, not a prediction. A red region does not represent a new diagnosis or a model's confidence. The source PNGs are already processed 2D slices; their brightness is not an original HU measurement.

## Open each patient's preview

The command creates:

| File under `learning_outputs/step6a` | Purpose |
| --- | --- |
| `patient_previews/{case}_reference_preview.png` | The two selected reference slice views for one training patient, with patient/slice identities |
| `training_review_overview.png` | A small overview of the training review |
| `training_reference_review.json` | Machine provenance and the pending manual-review record |

An unchanged repeat reuses identical outputs. Existing edited or different evidence is preserved; use a deliberately versioned output directory when a new review is needed. The script does not overwrite your edited review report.

For example, open `patient_previews/case_00000_reference_preview.png` in VS Code. Review **all 20 patient files**. The overview is useful for navigation, but its thumbnails may be too small to assess an edge or a small tumour region. Use the individual patient previews at a readable size.

Check these points for each selected pair:

1. **Identity:** the patient and slice labels match the intended training case. The CT and reference view belong to the same case and slice.
2. **Position:** foreground outlines lie on the corresponding visible structures, rather than being displaced across the image.
3. **Transpose:** the mask does not look as if its row and column axes were swapped relative to the CT.
4. **Mirror or rotation:** image and annotation have the same relative orientation, without one being flipped or rotated independently.
5. **Shift or scale:** boundaries do not show a consistent offset, stretching or resize mismatch.
6. **Class numbering:** green and red follow the stated 1/2 mapping, and background is not displayed as a foreground class.

Record the case ID, slice index and suspected problem if something looks wrong. Describe the visible mismatch instead of correcting original PNGs or changing the split. A second slice can help identify an obvious pairing problem, but two slices do not establish correctness across that patient's entire selected series.

PNG appearance alone does not establish the original scan's left/right orientation, physical spacing or anatomical annotation quality. This exercise checks visible CT/reference correspondence; it does not certify a clinician's labels or replace medical review.

## Manual review stays pending

The JSON report contains:

```json
{
  "manual_review_status": "pending"
}
```

Creating or opening previews does not automatically turn this into an approval. Machine decoding/hash checks and human visual review are different evidence. The script must not infer that the review passed merely because no program error occurred.

After inspecting the individual files, report what you checked and any unresolved cases. Keep the generated record pending until an explicit review decision is recorded through the agreed workflow. Do not present preview generation as completed anatomical validation.

There are no Dice/IoU scores, accuracy, calibrated confidence, trained checkpoint or new-patient result from Step 6A. The remaining source-alias and selected-slice/full-scan limitations from Step 5 still apply.

## Next: the formal training entry

The Step 6B PNG training entry is implemented and fixture tested; real training and validation are pending. After reviewing the training previews and resolving visible pair problems, its first epoch uses the 20 locked training patients. Validation has a separate role in model selection; final test and the five independent prediction patients remain reserved for their designated stages.

Run that entry as a separate, explicit step. Step 4's one-batch update and these reference previews do not count as formal locked-cohort training.
