# Step 2: Understand One Real 2D CT and Its Reference Mask

> Feature-branch layout: run commands from `backend/kidney` (use `cd backend/kidney` from the repository root). Replace `/absolute/path/to/kits_project` with your own absolute data path. Prepare the branch environment using [the module README](../README.md); environments, data, caches and generated results are not included in Git. The original local starter is unchanged.

This step reads one real exported CT PNG and its existing reference annotation. It does not train a model or change the original CT, mask or manifest. All comments, terminal messages and output fields are in English.

## Before running

The restored local dataset contains 20,686 CT PNGs and 20,686 matching masks for 210 cases. This describes coverage of the existing manifest, not every slice in each original CT scan. Step 5 has subsequently locked the formal cohort; `case_00000` belongs to training because it was inspected during development.

The inspection script and its source helpers are already included in this module. Keep the actual data in your external `kits_project` directory. Do not copy images into the repository.

## Run in the VS Code terminal

Open the feature-branch repository in VS Code. Prepare the module environment using its README, then run from the repository root:

```sh
cd backend/kidney
source .venv/bin/activate
python learning/02_inspect_2d_pair.py --root /absolute/path/to/kits_project --case case_00000 --slice-index 294
```

| Command | Purpose |
| --- | --- |
| `cd backend/kidney` | Use the module directory as the working directory |
| `source .venv/bin/activate` | Use that environment in this terminal |
| `python learning/02_inspect_2d_pair.py ...` | Read the selected CT/mask pair, verify its labels and CSV counts, and save reference displays |

If using the VS Code Run button later, select the module's `.venv/bin/python` through **Python: Select Interpreter**. These terminal commands use the activated environment directly.

## Understand the inputs

The selected files are `data2d/images/case_00000_z294.png` and `data2d/masks/case_00000_z294.png`. The CSV row identifies patient `case_00000` and original slice index `294`. It is not the 294th row of the CSV or necessarily the 294th saved PNG.

The CT is a grayscale array with shape `(256, 256)`: 256 rows and 256 columns. `uint8` means an unsigned integer from 0 to 255. These PNG intensities have already been windowed and resized; they are not raw Hounsfield units. Do not apply HU windowing again.

The mask has the same shape. Each pixel stores a class ID, not a display brightness:

| Original mask value | Class | Display colour |
| --- | --- | --- |
| 0 | Background | Black |
| 1 | Kidney | Green |
| 2 | Tumour | Red |

The original mask looks almost black in an ordinary image viewer because its values are only 0, 1 and 2. The script creates separate coloured display files while leaving those training labels unchanged.

## What the code does

Read the numbered comments in `02_inspect_2d_pair.py` in order:

1. `find_record` locates exactly one patient/slice row. `checked_asset` checks the corresponding CT and mask paths and their filename identities.
2. `read_greyscale_png` decodes each PNG into a two-dimensional pixel array and records its file hash.
3. `verify_mask` accepts only class IDs 0/1/2 and compares actual kidney/tumour pixel counts and presence flags with the CSV.
4. `reference_images` gives the classes display colours and blends the foreground annotation onto the CT. `labelled_preview` assembles the three panels.
5. `inspect` saves the display files and `inspection.json` outside the original data project.

Matching dimensions, filenames and metadata are useful checks. They do not prove anatomical correctness for all images; this step inspects one pair. No neural network runs here.

## Expected result for this pair

The actual verification on 2026-10-06 produced:

```text
[DECODED CT PNG]
shape=(256, 256); dtype=uint8; range=0..255

[DECODED REFERENCE MASK]
shape=(256, 256); dtype=uint8; range=0..2
label=0 (background); pixels=64640
label=1 (kidney); pixels=839
label=2 (tumour); pixels=57
CSV kidney_px/tumor_px/has_kidney/has_tumor match the decoded mask.
```

These counts sum to 65,536 pixels, which is 256 × 256. They describe this slice only; they are not a physical tumour volume or model performance score.

The default output folder is `backend/kidney/learning_outputs/step2`. Open `reference_preview.png` in VS Code to see the three panels:

| Output | Meaning |
| --- | --- |
| `ct_display.png` | CT copied into an RGB display image |
| `reference_mask_coloured.png` | Existing reference labels shown in black, green and red |
| `reference_overlay.png` | Existing reference labels blended onto the matching CT |
| `reference_preview.png` | Labelled CT, reference mask and overlay shown together |
| `inspection.json` | Source paths and hashes, array properties, label counts and CSV comparisons |

The coloured region comes from the existing reference mask. It is the learning target, not a model prediction. A future trained model will produce its own predicted mask, which can be compared with this reference.

## Before the next step

Check the terminal output and open the preview. If there is an error, retain its complete message and resolve it before proceeding. This script is not evidence of model training or final performance.

Patient `case_00000` has been inspected during development and is assigned to training by the current lock. Do not present it as an unseen final-test patient or an independent new-patient example. Other slices from this patient follow the same assignment.

Continue with the implemented pure 2D loader in Step 3 to understand how a CT becomes an input tensor and its mask becomes a target. Formal training uses the Step 5 lock; keep the original `splits.json` unchanged.
