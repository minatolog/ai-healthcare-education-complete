# Step 3: Turn One CT PNG and Its Mask into a Tensor Batch

> Feature-branch layout: run commands from `backend/kidney` (use `cd backend/kidney` from the repository root). Replace `/absolute/path/to/kits_project` with your own absolute data path. Prepare the branch environment using [the module README](../README.md); environments, data, caches and generated results are not included in Git. The original local starter is unchanged.

Run this step yourself in VS Code, then inspect the terminal and JSON report. It uses at most two existing PNG/mask pairs from the explicit development patient `case_00000`. It does not run a model, train, select a formal cohort, evaluate validation/test patients, or create predictions. All source data stays in `kits_project`.

## Use the flat module layout

Open the feature-branch repository in VS Code. This step uses:

- `learning/03_check_2d_loader.py`: the small command-line inspection script.
- `learning/requirements-step3.txt`: dependencies for this step.
- `kidney_segmentation/png_dataset.py`: the reusable pure 2D adapter.
- `learning/STEP3.md`: this explanation.

Keep the module files in place. The adapter is the PNG entry point; the earlier full-volume/2.5D pipeline remains separate. This script finds `kidney_segmentation` beside `learning`, without requiring a global package installation.

## Run one command at a time

Prepare the module environment using its README. From the repository root, run:

```sh
cd backend/kidney
source .venv/bin/activate
python learning/03_check_2d_loader.py --root /absolute/path/to/kits_project --case case_00000 --batch-size 2 --output-dir learning_outputs/step3
```

Select the module's `.venv/bin/python` using **Python: Select Interpreter** in VS Code. Activation selects the same interpreter in the terminal. Script and output paths are relative to `backend/kidney`; the external data path is absolute and does not depend on that working directory. The original starter's environment is separate from this checkout.

The default inspection selects up to two existing pairs, preferring the `z294` pair inspected in Step 2, then another available row from the same patient ordered by `z`. It does not select examples using tumour labels. If only one pair exists, the actual batch contains one sample. `--max-samples 1` restricts inspection to one pair; `--batch-size 1` stacks one sample per batch. No old `splits.json` is loaded.

## Understand the shapes

| Value | Shape | Meaning |
| --- | --- | --- |
| One input | `[1, 256, 256]` | One grayscale channel, 256 image rows, 256 image columns |
| One target | `[256, 256]` | One class ID at every matching pixel |
| Input batch | `[B, 1, 256, 256]` | `B` images stacked together; normally 2 in this inspection |
| Target batch | `[B, 256, 256]` | The `B` corresponding reference masks |

`B` means batch size, `C` means channels, `H` means image height, and `W` means image width. A batch groups examples for computation; it does not turn two slices into two patients. Both samples in this step belong to `case_00000`, and each retains its original slice index.

This is **pure 2D**: each input reads one PNG with `C=1`. It does not load neighbouring slices. Repeating one PNG three times would still be repeated 2D input, not real 2.5D context; this adapter requires one channel.

## Normalize the CT, preserve the labels

The existing CT PNG already contains windowed, resized intensities from 0 to 255. Those values are not raw HU. The adapter performs:

```text
input = (PNG / 255.0 - 0.456) / 0.224
```

Dividing by 255 maps brightness to 0–1. Subtracting the mean and dividing by the standard deviation puts the image on the documented scale expected by the model. The resulting input uses `torch.float32`. Negative normalized values are expected; they do not indicate an invalid image. For PNG values spanning 0–255, the theoretical normalized range is approximately −2.035714 to 2.428571. The script prints the actual observed range of each batch; it may be narrower.

The mask is different: its values are class IDs, not brightness. Keep 0=background, 1=kidney and 2=tumour unchanged as `torch.int64`. Do not divide labels by 255, normalize them, colour them, merge kidney/tumour, or append them to the input. The mask is only the supervised target, not a model input channel. Actual kidney/tumour pixel counts and flags must match the manifest metadata.

## Follow the functions

1. `load_adapter` imports PyTorch and the reusable `PNGPatientDataset`/`PNGConfig` from the local source package.
2. `checked_output_dir` protects the original project and rejects symlinked output locations. Reports belong in the module's `learning_outputs/step3` directory.
3. `PNGPatientDataset(..., patient_ids=[case])` retains the selected patient's manifest rows and identity checks. `select_samples` identifies at most two existing pairs; this small development inspection does not claim the remaining rows were validated.
4. The dataset's `__getitem__` reads one grayscale CT and one mask, checks dimensions/labels/metadata, and produces separate input and target tensors with source paths and hashes.
5. `Subset` limits the inspected indices. `DataLoader(..., shuffle=False, num_workers=0)` stacks the samples in deterministic order in the current process. It does not change the patient grouping.
6. `inspect` checks batch shapes, dtypes, finite inputs and label IDs, then prints the observed values and saves metadata only.

The reusable adapter belongs in the model package so a later training script can use the same loading rules. The learning script only demonstrates those rules; it contains no neural network or optimizer.

## Expected output

The local check on 2026-10-06 found 322 manifest records and 322 existing pairs for `case_00000`. It inspected only two of those pairs. Running the default command in the module's environment produced:

```text
input shape=(2, 1, 256, 256); dtype=torch.float32; normalized range=-2.035714..2.428571
target shape=(2, 256, 256); dtype=torch.int64; unique class IDs=[0, 1, 2]
patient_id=case_00000; original slice_index=294
patient_id=case_00000; original slice_index=4
```

Observed class IDs must be a subset of `[0, 1, 2]`; not every slice needs to contain all three classes. The known `z294` reference mask from Step 2 contains all three. Other patients or settings can produce different ranges, class sets or batch sizes. The terminal reports actual observations, not model accuracy or an invented performance metric. The existence of 322 pairs is a file check, not validation of all their decoded pixels or coverage of the original full scan.

Open `learning_outputs/step3/loader_report.json` in VS Code. It records normalization/configuration and its hash, the manifest hash, actual batch shapes/ranges, patient/slice identities, source paths/hashes and development exposure. It contains no tensor arrays, copied CT images or model predictions. A changed configuration or source file should produce a different corresponding hash.

## Stop here before Step 4

Check that inputs have one channel, targets keep integer IDs, and both sample identities belong to `case_00000`. If an error occurs, keep its complete message and resolve it before moving on. Inspection of two pairs does not establish formal dataset readiness or full-scan coverage. The legacy preprocessing selected slices using reference masks, so later results need to describe evaluation on selected PNG slices.

`case_00000` has been inspected during development. Later place the whole patient in training or outside the formal cohort. Do not present this patient or another of its slices as an unseen final-test case or independent new-patient example.

Step 4 introduces the implemented one-channel model forward pass, loss and backward update. The ResNet input and compatible decoder accept this tensor directly. No model adaptation or training is performed by Step 3 itself.
