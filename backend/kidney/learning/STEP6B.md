# Step 6B: One Real Training Epoch on the Locked PNG Cohort

> Feature-branch layout: run commands from `backend/kidney` (use `cd backend/kidney` from the repository root). Replace `/absolute/path/to/kits_project` with your own absolute data path. Prepare the branch environment using [the module README](../README.md); environments, data, caches and generated results are not included in Git. The original local starter is unchanged.

Step 6A generated training reference previews. **Open the individual training-patient PNG previews and check CT/reference alignment.** The JSON report lists provenance and status; it does not display the CT/mask correspondence. The overview alone may be too small. Describe any pairing, transpose, mirror, shift or class-number problem before using those targets to train.

The Step 6B training entry is implemented separately from the original volume workflow. Preparing/testing its code does not mean real training has run. This guide describes the first run after reviewing the training previews. It does not automatically change Step 6A's pending manual-review record or certify clinical annotation quality.

This remains three-class **2D segmentation**: background, kidney and tumour. It does not create a healthy-patient/Tumour-No Tumour classifier or establish clinical validity.

## Run from the feature-branch checkout

Prepare the module environment and pinned encoder cache using its README. Select `backend/kidney/.venv/bin/python` in VS Code. The branch does not include the original starter's environment or its generated audit/cohort/review evidence.

To check the current metadata and pinned offline encoder files without training, run:

```sh
cd backend/kidney
source .venv/bin/activate
python learning/06b_train_png.py --root /absolute/path/to/kits_project --check-only
```

This check decodes no CT/mask PNGs, initializes no model and creates no run directory. It does not replace per-sample source verification during training.

The command defaults to local `learning_outputs/step5` and `step6a` evidence. If those files remain in the original starter, pass their absolute paths through `--cohort`, `--audit` and `--reference-review`; no copy or change to the original evidence is required. The module README shows those explicit arguments.

When you are ready to start the real first epoch, run:

```sh
source .venv/bin/activate
python learning/06b_train_png.py --root /absolute/path/to/kits_project --epochs 1 --batch-size 2 --device cpu
```

The entry uses the Step 5 cohort lock and its bound audit. It must check the selected source identities, hashes and preprocessing configuration before consuming the PNG data. Keep the original manifest, `splits.json`, CTs and masks unchanged.

The default output is `learning_outputs/step6b/run01`. Existing run artifacts must not be overwritten. If you deliberately choose to run a new experiment, use an explicit new directory, for example:

```sh
python learning/06b_train_png.py --root /absolute/path/to/kits_project --epochs 1 --batch-size 2 --device cpu --output-dir learning_outputs/step6b/run02
```

The current version starts fresh from the verified pretrained encoder and a newly initialized decoder/head. It does **not** resume `run01`, Step 4's one-update checkpoint or another saved run. A new output directory represents a new run, not additional epochs on an earlier checkpoint.

You may record what you inspected using an optional `--visual-review-note` text. The note is recorded verbatim; it is not a clinical approval and does not alter the Step 6A evidence. There is no need to replace every `pending` field in its JSON.

## One epoch uses every locked training slice

For the currently verified cohort, the run uses:

| Pool | Patients | Exported slices | Role in this entry |
| --- | ---: | ---: | --- |
| Training | 20 | 2,071 | Learn decoder/head parameters |
| Validation | 10 | 1,156 | Compute separate validation results and select the checkpoint |
| Final test | 10 | 530 | Image/mask pixels are not loaded |
| Independent prediction | 5 | 676 | Image/mask pixels are not loaded |

These counts come from the existing lock. They describe reference-selected exported PNG slices, not complete original CT volumes.

Training shuffles the complete training-slice list and visits it once per epoch. With batch size 2, 2,071 slices require **1,036 training batches**. The final batch contains one slice and must still be processed; it is not discarded. All validation slices are processed separately after the epoch, without training on their targets.

An epoch is a pass through the selected training dataset. It is much larger than Step 4's single-batch, single-update exercise. One epoch verifies a real training path on this cohort; it does not guarantee a useful segmentation model or a good validation score.

## Follow the tensor shapes

Let `B` be the current batch size, normally 2 and 1 for the last training batch:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| CT input | `[B, 1, 256, 256]` | One normalized grayscale slice per sample |
| Reference target | `[B, 256, 256]` | Integer pixel class IDs 0/1/2 |
| Model logits | `[B, 3, 256, 256]` | Three class scores at every pixel |
| Predicted mask | `[B, 256, 256]` | Highest-scoring class ID at every pixel |

The reference mask is the learning target and never an input channel. The three output channels are background/kidney/tumour scores; they are not three CT slices. The input is a single 2D slice, without the old adjacent-slice 2.5D representation.

The model is a U-Net with a strictly verified, cached ImageNet ResNet-18 encoder. The encoder parameters remain frozen, and its BatchNorm statistics stay unchanged. Only the decoder and segmentation head learn. Missing or unverified pretrained weights cause a clear failure; there is no silent random-encoder fallback.

For each training batch, the entry clears gradients, performs a forward pass, computes loss, calls `backward()` and takes one optimizer step. Validation uses evaluation mode with no gradient updates.

## Loss and segmentation metrics answer different questions

The training objective remains:

```text
total loss = cross-entropy + foreground soft-Dice loss
```

Cross-entropy compares class scores with reference pixel IDs. Foreground soft-Dice uses soft class weights for kidney/tumour overlap so gradients can flow. This combined loss supplies the learning signal; a lower loss is not itself a reported hard Dice score.

Validation metrics use the **hard predicted class mask** from argmax and compare it with the reference mask. Dice, IoU, precision, recall and F1 describe overlap or pixel errors. Read the report's class definitions and empty-class handling when interpreting them; do not treat soft-Dice loss and hard Dice as interchangeable values.

The validation summary is **patient balanced**. Results are accumulated for each of the 10 validation patients, then the patient results are averaged so a patient with many exported slices does not dominate merely by contributing more images. Inspect the individual patient/class results as well as the mean.

Validation may be used to select a checkpoint. Final test is not used to choose an epoch, learning rate, model or favourable result. With one epoch, there is only one completed epoch candidate; the word “best” in its filename does not imply a search has established strong performance.

These are selected-slice segmentation results. They do not establish full-volume coverage, physical tumour volume, patient-level cancer diagnosis or calibrated medical confidence.

## Monitor progress and interruption

The entry prints progress every 10 batches, at the final batch, or after 30 seconds since the last progress record. It writes progress to `progress.jsonl`. The first run uses the Mac's CPU. Training speed depends on the machine, other activity and the installed runtime; no completion-time estimate is guaranteed.

CPU is the default. Apple MPS acceleration may be explored later as an explicit option after compatibility is checked; it is not the default route for this first run.

If you press **Ctrl+C**, the entry records an interrupted run summary. An interrupted run must not be described as completed training. An intermediate file or a printed loss does not prove that the epoch and full validation both finished. Check the final run status and completed-epoch information.

Resolve a readable failure message before retrying. Preserve the earlier run evidence and use an explicit new run directory for a fresh attempt. The current version does not resume partial progress.

## Read the outputs

The entry uses these files under the chosen run directory:

| File | Purpose |
| --- | --- |
| `best_model.pt` | Checkpoint selected using validation, with the new PNG schema and provenance |
| `training_history.json` | Actual completed-epoch training/validation history |
| `validation_report.json` | Per-patient/class validation results and patient-balanced summary |
| `run_summary.json` | Configuration, provenance, run status and completed work |
| `progress.jsonl` | Sequential progress records written during the run |

Some final outputs may be absent if the run fails or is interrupted before their stage. Use `run_summary.json` and the actual error/status rather than inventing missing results. Keep generated checkpoints and reports outside GitLab by default.

The new checkpoint schema is **`png_segmentation_v1`**. It binds the model to the single-channel PNG configuration, reviewed cohort/data provenance and pretrained encoder proof. It is consumed through the new compatible PNG loading path. The legacy volume checkpoint format and Step 4's `learning_step4_v1` checkpoint are not compatible substitutes; do not rename one to bypass the loader.

Before saying the run completed, confirm that all 2,071 training slices and all 1,156 validation slices were processed, the loss and defined metrics are finite, the run status records completion, and source/cohort/preprocessing bindings remain intact. Rates with zero denominators are JSON `null`; patient means exclude those undefined rates and retain contributing-patient counts. Copy actual runtime observations into the learning record; no loss decrease or segmentation score is promised here.

## Continue one step at a time

Your immediate task is still to inspect the Step 6A training previews. The next task is to run the prepared Step 6B entry and understand its training progress, validation report and checkpoint. Independent final testing and the five new-patient predictions come later through their own stages; this first epoch does not consume them.
