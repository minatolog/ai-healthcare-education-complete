# Kidney 2D Learning Roadmap

Run the feature-branch learning commands from `backend/kidney`. The original local starter and dataset are unchanged. This checkout contains source and tests, not their environments, patient PNGs, pretrained weights or generated evidence.

Prepare the environment and pinned encoder cache using [the module README](../README.md). From the repository root:

```sh
cd backend/kidney
source .venv/bin/activate
```

Replace `/absolute/path/to/kits_project` in every command with your external data directory. The old sibling `../kits_project` default is unsuitable for this layout. Outputs are local ignored `learning_outputs/` files; keep them outside Git. Existing reviewed audit/cohort/review evidence may remain in the original starter and be referenced by absolute paths.

## Current project route and status

The user chose **pure 2D PNG segmentation**: one grayscale CT slice per input, with separate reference class IDs `0=background`, `1=kidney`, `2=tumour`. The model uses a verified offline ImageNet ResNet-18 encoder with frozen parameters/BatchNorm and a trainable U-Net decoder/head. The legacy full-volume/2.5D entry is a separate reference route, not the default learning command.

The dataset's original preprocessing retained foreground slices and a subset of background slices using reference masks. Results must describe segmentation on selected exported slices, not blind full-scan performance. PNG intensities have already been processed; do not apply HU windowing again or treat uint8 brightness as HU.

The local Step 5 audit decoded 20,686 CT/mask pairs from 210 cases, with no machine issues or informative exact cross-patient CT duplicates under its documented policy. It preserves the original 147/32/31 split and locks:

| Pool | Patients | Exported slices |
| --- | ---: | ---: |
| Training | 20 | 2,071 |
| Validation | 10 | 1,156 |
| Final test | 10 | 530 |
| Independent prediction | 5 | 676 |

The first three pools form the 40-patient modelling cohort; prediction uses five additional patients. `case_00000` is training because it was inspected and updated during development. Patient/source/content overlap cannot cross pools under the reviewed controls. Machine checks do not establish anatomical correctness, external source aliases or complete original-scan coverage.

Step 6A generated 40 training reference slice views in 20 patient files and an overview; manual review remains explicit/pending. Step 6B is implemented and engineering tested. Its local metadata/offline-cache preflight passed without model initialization or PNG decoding. **Real locked-cohort training and validation are still pending.** The next user action is to inspect the training previews, then explicitly start the first epoch.

## Follow the steps

| Step | Command/guide | What it teaches |
| --- | --- | --- |
| 1 | `01_check_2d_data.py` | Read-only metadata inventory, file pairing and selected-slice coverage |
| 2 | [STEP2.md](STEP2.md) | Decode one `case_00000`, z294 CT/reference pair and create labelled reference displays |
| 3 | [STEP3.md](STEP3.md) | Load pure 2D tensors with patient/slice identity, normalized CT and unchanged integer labels |
| 4 | [STEP4.md](STEP4.md) | One forward/loss/backward/optimizer update on a development batch; not formal performance |
| 5 | [STEP5.md](STEP5.md) | Full machine PNG audit and deterministic audit-bound patient-cohort lock |
| 6A | [STEP6A.md](STEP6A.md) | Training-only visual pairing/alignment review; GT-based QA slice choice is not an inference rule |
| 6B | [STEP6B.md](STEP6B.md) | All training slices per epoch, separate patient-balanced validation and PNG checkpoint |
| Later | [Feature handoff](../../../docs/kidney/FEATURE_BRANCH_HANDOFF.md) | Frozen-model PNG final test, independent prediction, backend/API/frontend and Diabetes integration |

To repeat Step 1 from the module directory:

```sh
python learning/01_check_2d_data.py --root /absolute/path/to/kits_project --output learning_outputs/step1/step1_data_audit.json
```

It uses the standard-library metadata auditor in `scripts/`. Its 16 fixture tests cover inventory behavior. File existence is not pixel/anatomical validation; Step 5 supplies broader machine checks. Do not use the original data project's old training command: it uses a different split and task configuration.

## Preserve historical observations without claiming performance

The original local Step 3 check used case_00000 z294 and z4. Its input was `[2,1,256,256]` float32 with range approximately −2.035714 to 2.428571; the target was `[2,256,256]` int64 with IDs 0/1/2. There were 322 selected records for that patient, but only two pairs were decoded in that exercise.

Step 4's one Adam update changed 32 decoder/head parameter tensors while preserving the entire encoder state. Same-batch eval-mode loss changed from 1.855392 to 1.614814; train-mode backward loss was 2.364693. This was a computation/update check, not held-out performance, and its predictions remained inaccurate. Step 4's learning-only checkpoint is not accepted as a formal PNG checkpoint.

The flat-layout engineering suite passed **188 tests**, including 16 metadata-audit tests. The 13 core package Python files were SHA-identical to the existing source during relocation. GitLab server CI has not run. No real Step 6B epoch or validation result follows from these fixture checks.

## Next manual run

Inspect the training-patient reference PNG previews, then use Step 6B's check-only command before a deliberate training run. Pass absolute `--cohort`, `--audit` and `--reference-review` paths if the reviewed evidence remains in the original starter. Do not edit a pending reference-review field merely to make a check appear approved.

A completed Step 6B run will use all 2,071 training and 1,156 validation slices, leave final-test/prediction pixels unused, and create `png_segmentation_v1` checkpoint/history/progress evidence in a new run directory. It starts fresh, refuses overwrites and records interrupted/failed status accurately. The dedicated PNG final-test and prediction entry points, shared service contract, frontend assets and separate Diabetes adapter are follow-up work; no API compatibility or story completion is claimed here.
