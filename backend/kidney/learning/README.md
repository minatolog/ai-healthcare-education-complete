# Kidney 2D Learning Roadmap

Run the feature-branch learning commands from `backend/kidney`. The original local Starter remains a separate learning copy; patient data stays in the external data project. This checkout contains source and tests, not their environments, patient PNGs, pretrained weights or generated evidence.

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

Step 6A generated 40 training reference slice views in 20 patient files and an overview; its saved manual review remains pending. The student completed local Step 6B run01 (one epoch, 1,036 updates) and run02 (five epochs, 5,180 updates), each using all 2,071 training and 1,156 validation slices per epoch. Checkpoint and source fingerprints passed read-only verification. Run01 patient-mean validation Dice was 0.806567 for kidney and 0.040669 for tumour. Run02 retained epoch 4 under minimum validation loss: kidney Dice 0.862837, tumour Dice 0.108358. Epoch 5 had tumour Dice 0.135298 but its weights were not saved; that score does not describe the retained checkpoint. Both runs have completed Step 6C reference-guided train/validation previews. These images diagnose errors rather than supply aggregate performance evidence. Artifacts remain local; those initial runs preceded the frozen test/prediction stages described below. Step 6D run03 has now completed five epochs, retaining epoch 5 with tumour Dice 0.135298, tumour recall 0.113804 and kidney Dice 0.842071. All five optimization histories exactly reproduce run02; the selection change did not alter learning. Four of ten validation patients have tumour true positives; tumour precision uses only six contributing patients. The run03 checkpoint, three training artifacts and 90 diagnostic PNG hashes passed read-only checks. Step 6E run04 completed five epochs with relative CE class weights [1,1,2], retaining epoch 4: tumour Dice 0.217023, tumour recall 0.210563 and kidney Dice 0.865631. Five of ten validation patients have tumour true positives; total tumour false-positive pixels increased to 118,261. The checkpoint, reports, 90 generated PNGs and 120 selected source-file byte hashes passed read-only checks. Step 6F run05 completed its predefined ten epochs with weight 2 and all other settings fixed. The first five epochs exactly reproduced run04. Tumour-Dice selection retained epoch 10: tumour Dice 0.253372, tumour recall 0.261521 and kidney Dice 0.867657. Total tumour false-positive pixels increased to 156,734. Three validation patients have zero tumour Dice; two additional patients have very few true-positive pixels. The retained checkpoint, three training artifacts, 90 diagnostic PNG hashes and 120 selected source-file hashes passed read-only checks. Step 7 froze run05 epoch 10 and completed final testing on ten patients/530 selected slices. Patient-mean kidney Dice was 0.876164; tumour Dice 0.368032, IoU 0.272769, precision 0.524337 (eight contributors) and recall 0.378970. Four test patients have zero tumour Dice. Completed evidence hashes, all test row identities and patient metrics from confusion matrices passed read-only checks; model state remained unchanged with zero optimizer steps. Step 8 completed independent CT-only inference for five additional patients/676 CT images, writing 676 predicted label PNGs, 676 overlays and five fixed previews. All 1,357 output PNG hashes/formats/counts passed read-only checks; the frozen model and completed-test records stayed unchanged. No reference-mask pixels or reference metrics were used in prediction.

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
| 6C | [STEP6C.md](STEP6C.md) | Fixed reference-guided train/validation CT/reference/prediction comparisons, without retraining |
| 6D | [STEP6D.md](STEP6D.md) | Optional patient-mean validation tumour-Dice checkpoint retention, with the same training protocol |
| 6E | [STEP6E.md](STEP6E.md) | A controlled tumour CE class-weight experiment; default loss stays unchanged |
| 6F | [STEP6F.md](STEP6F.md) | A fixed ten-epoch budget with the same weight-2 training configuration |
| 7 | [STEP7.md](STEP7.md) | Freeze the selected run05 checkpoint and evaluate all reserved test rows once, without updating model state |
| 8 | [STEP8.md](STEP8.md) | Frozen CT-only inference, no reference-mask pixel access, labels/overlays/fixed previews and JSON |
| Later | [Feature handoff](../../../docs/kidney/FEATURE_BRANCH_HANDOFF.md) | Shared backend/API/frontend and Diabetes integration |

To repeat Step 1 from the module directory:

```sh
python learning/01_check_2d_data.py --root /absolute/path/to/kits_project --output learning_outputs/step1/step1_data_audit.json
```

It uses the standard-library metadata auditor in `scripts/`. Its 16 fixture tests cover inventory behavior. File existence is not pixel/anatomical validation; Step 5 supplies broader machine checks. Do not use the original data project's old training command: it uses a different split and task configuration.

## Preserve historical observations without claiming performance

The original local Step 3 check used case_00000 z294 and z4. Its input was `[2,1,256,256]` float32 with range approximately −2.035714 to 2.428571; the target was `[2,256,256]` int64 with IDs 0/1/2. There were 322 selected records for that patient, but only two pairs were decoded in that exercise.

Step 4's one Adam update changed 32 decoder/head parameter tensors while preserving the entire encoder state. Same-batch eval-mode loss changed from 1.855392 to 1.614814; train-mode backward loss was 2.364693. This was a computation/update check, not held-out performance, and its predictions remained inaccurate. Step 4's learning-only checkpoint is not accepted as a formal PNG checkpoint.

The initial flat-layout suite passed 188 tests, including 16 metadata-audit tests; MR !3 server CI passed before merging to develop. The 13 core package Python files retained their source hashes during relocation. Step 6C adds 36 fixture checks for binding, diagnostic-example selection, inference-only rendering and output preservation. Step 6D adds 57 core checkpoint-selection checks and five CLI checks, including unchanged optimization trajectories and legacy checkpoint compatibility. Step 6E added 35 core loss-weight checks and 12 CLI checks; the complete local module suite passed 333 tests. Step 6F only adds guidance, with no model-code or dependency change. Step 7 adds 28 core freeze/test fixture checks and 44 CLI orchestration checks. The deployed feature module passed 405 local tests, and real run05 freeze/test preflights passed with model initialization and PNG decoding explicitly forbidden. No real test was opened during preparation. Fixture checks remain separate from the real Step 6B execution reported above.

## Next source merge and integration

The student completed Steps 7 and 8. The current review-repaired suite passed 481 local tests; genuine run01–run05 metadata/state remained unchanged in no-PNG/no-model compatibility checks. Preserve all existing run01–run05, freeze, final-test and prediction artifacts. The completed prediction run need not be repeated. The current training command does not resume; future training experiments should predeclare their complete budget and preserve this frozen baseline/test evidence.

Prepare the next source MR using [its English description](../../../docs/kidney/NEXT_MR_DESCRIPTION.md), and follow the verified validation/test/prediction observations in [the results summary](../../../docs/kidney/RUN05_MODEL_RESULTS.md). Source, fixture tests and small documentation belong in the MR; data, weights and generated local assets remain excluded. GitLab server CI must pass for the new submitted commits.

Then agree the shared service contract with the frontend/backend and Diabetes contributor. HTTP routes, asset serving, job persistence and real application integration remain pending. No API compatibility, calibrated confidence, patient diagnosis or Jira completion is claimed here. Do not change a pending reference-review field merely to make a record appear approved.
