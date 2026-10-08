# Step 4: Understand One Forward Pass and One Parameter Update

> Feature-branch layout: run commands from `backend/kidney` (use `cd backend/kidney` from the repository root). Replace `/absolute/path/to/kits_project` with your own absolute data path. Prepare the branch environment using [the module README](../README.md); environments, data, caches and generated results are not included in Git. The original local starter is unchanged.

You have checked that Step 3 produces a one-channel input tensor and a separate reference-mask target. This exercise now performs **exactly one optimizer update** on one development batch from `case_00000`, normally slices `z294` and `z4`. It is not a formal training run, held-out validation/test evaluation, or independent new-patient prediction.

The model uses a strictly verified, cached ImageNet ResNet-18 encoder and a U-Net decoder. The encoder remains frozen; only the decoder and segmentation head learn from this batch. There is no silent fallback to a random encoder. Missing or unverified cached weights stop the exercise with an error.

## Run in your existing VS Code environment

Open the feature-branch repository in VS Code. Prepare the module environment and pinned encoder cache using its README. Select `backend/kidney/.venv/bin/python`, then run from the repository root:

```sh
cd backend/kidney
source .venv/bin/activate
python learning/04_one_training_step.py --root /absolute/path/to/kits_project --case case_00000 --batch-size 2 --output-dir learning_outputs/step4
```

The original starter and its environment remain separate. Do not copy either into Git. Use the module's tested dependency snapshot for this checkout.

The script imports the local `kidney_segmentation` source package and reuses Step 3's explicit patient and sample-selection helpers. Keep those existing files in place. Keep the actual PNGs and masks in the absolute `--root` path; do not copy them into the repository. The first sample is `z294` when available, followed by another existing row of the same patient ordered by `z`. Sample selection does not use tumour labels.

Defaults are seed 42, CPU with two threads, Adam learning rate `0.001`, one batch, and one update. `--learning-rate` accepts only a positive finite number. This is a learning control, not permission to tune against test patients. Any patient used here is exposed to development and must later belong to training or remain outside the formal cohort.

Each run starts again from the verified ImageNet encoder and a fresh seed-controlled decoder/head. It does not resume `learning_step.pt`. Repeating the command produces separate one-update runs and replaces the files in the selected output folder; it does not accumulate training steps. Use another output directory if you want to retain multiple runs.

## One input channel becomes three score channels

| Tensor | Default shape | Meaning |
| --- | --- | --- |
| Input | `[2, 1, 256, 256]` | Two grayscale 2D images, each with one channel |
| Reference target | `[2, 256, 256]` | Integer class IDs 0/1/2 at corresponding pixels |
| Model logits | `[2, 3, 256, 256]` | Three class scores at every pixel of every image |
| Predicted mask | `[2, 256, 256]` | The highest-scoring class ID at each pixel |

The input's one channel is image brightness; the output's three channels are scores for background, kidney and tumour. They do not represent three adjacent input slices. The model's one-channel input layer adapts the original RGB ImageNet weights by summing their three input-channel weights. The offline verification checks this adaptation against the official cached encoder tensors.

`softmax` converts the three scores at a pixel into normalized class weights. `argmax` selects the class with the highest score and creates a predicted mask. This script uses argmax for region displays. It does not report those weights as calibrated medical confidence or a disease probability.

## How the loss teaches the model

The reference mask is the target. It is never an input image channel. The combined loss is:

```text
total loss = cross-entropy + foreground soft-Dice loss
```

- **Cross-entropy (CE)** penalizes class scores that disagree with the reference pixel IDs.
- **Foreground soft-Dice loss** encourages overlap with kidney and tumour reference regions, using soft class weights so gradients can flow. It excludes background from that overlap term.
- **Loss** is a scalar error signal for this development batch, not a final performance score.

`backward()` computes how each trainable parameter contributes to that error. These derivatives are its gradients. `optimizer.step()` uses the gradients to update decoder/head parameters once. The script checks that the gradients are finite and at least some are nonzero.

The pretrained encoder's parameters have `requires_grad=False`. Its BatchNorm layers also remain in eval mode, so their stored statistics do not change. The script compares every encoder parameter and buffer before/after, and confirms that decoder/head parameters changed.

## Follow the code in order

1. `load_runtime` reuses the Step 3 loader and imports the model/loss plus strict offline pretrained helper.
2. `checked_output_dir` keeps all new artifacts outside the original project and rejects symlinked destinations.
3. `PNGPatientDataset` and `DataLoader` produce one deterministic, one-channel development batch, preserving patient/slice identity and source hashes.
4. `build_verified_model(input_channels=1)` loads and verifies the official cached encoder weights. The decoder starts from the seed-controlled initialization.
5. A forward pass in `model.eval()` computes the **before** logits and CE/Dice loss. Here eval is a module setting, not evaluation on an unseen dataset.
6. `model.train()` enables decoder training while the encoder stays frozen. The script clears gradients, runs a separate training forward pass, calls `backward()` once and `optimizer.step()` once.
7. Another forward pass in `model.eval()` computes the **after** loss and predicted regions on exactly the same development batch.
8. The script saves labelled reference/before/after displays, a learning-only checkpoint and a JSON record.

The before and after losses use the same images, targets and eval mode. The loss used for backward comes from train mode and is reported separately. Decoder BatchNorm statistics can update during the training pass; that also affects the after result. Loss may decrease or increase after one update. Neither outcome demonstrates generalization.

## Read the outputs

The terminal prints actual input/logit shapes, sample identities, CE/Dice/total loss before and after, the loss used for backward, and update checks. Numeric losses are computed at runtime; do not expect a promised decrease or copy a predetermined metric.

The engineering verification on 2026-10-06 observed total eval-mode loss **1.855392 before** and **1.614814 after**, with train-mode backward loss **2.364693**. The entire encoder state was unchanged, and 32 decoder/head parameter tensors changed. These numbers describe this same development batch with the installed versions and seed; they are not accuracy or held-out scores. The predicted overlays still show substantial disagreement with the reference despite lower total loss. One update chiefly verifies that the computation and learning path work.

Open `learning_outputs/step4/comparison_preview.png` in VS Code. Its four panels are clearly labelled:

| Panel/file | Meaning |
| --- | --- |
| CT input | The first development sample's processed image |
| `reference_overlay.png` | Existing ground-truth/reference annotation |
| `prediction_before_overlay.png` | Model-predicted regions before the update |
| `prediction_after_overlay.png` | Model-predicted regions after one update |
| `comparison_preview.png` | CT and all three overlays shown together with case/slice identity and learning-only labels |

Green marks kidney and red marks tumour. Reference and predictions remain distinct. These predictions use the same development sample that supplied the learning target; they are not unseen-patient results, a diagnosis or a physical tumour volume.

`training_step.json` records actual loss components, batch identities, normalized input properties, PNG/manifest/config hashes, pretrained proof, finite-gradient and update checks, display identity, and the saved checkpoint hash. It reports no accuracy, calibration/confidence or full-cohort performance.

`learning_step.pt` has scope `training_patient_only_not_validated_model` and schema `learning_step4_v1`. It records one-channel PNG configuration, development sources and pretrained provenance alongside model weights. It is a learning artifact, not a validated model checkpoint. The formal model CLI does not accept this format. Keep it outside GitLab and do not use it to bypass the formal data/cohort audit.

## Before formal training

First confirm that the encoder state is unchanged, decoder/head parameters changed, losses/gradients are finite, and the overlay labels and source identities match. If an error occurs, keep the complete message and resolve it. No CT, mask, manifest or split file is modified by this script.

The next step is to lock a separately reviewed patient cohort and preprocessing configuration: 20 training / 10 validation / 10 final test patients, plus independent prediction patients. Keep `case_00000` in training or outside that formal cohort because it has been inspected and used in this learning update. All its slices follow the same patient assignment. Formal training needs versioned splits, leakage/duplicate-content checks, patient-level aggregation of segmentation metrics, and independent validation/test stages. The legacy mask-selected PNG cache must still be described as selected-slice data rather than a blind full-scan evaluation.
