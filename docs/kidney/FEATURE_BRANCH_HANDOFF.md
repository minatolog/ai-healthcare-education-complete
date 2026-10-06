# Kidney Feature-Branch Handoff

Feature branch: `feat/yining-kidney-segmentation`

Base branch: `develop`

Source module: `backend/kidney`

This contribution packages the existing Kidney Tumour learning/model source in the team's backend tree. It preserves the local starter, dataset, reviewed patient assignments, pinned pretrained weight hashes and data bindings. It does not include a real trained PNG model or claim frontend/API compatibility.

## Current scope and evidence

The implemented project route is one-channel, 256×256 PNG segmentation with `KidneyUNet(input_channels=1)`: verified offline ImageNet ResNet-18 encoder, frozen encoder parameters/BatchNorm, trainable U-Net decoder and three-class head. The earlier complete-volume/2.5D modules remain a separate reference route.

The source includes metadata inventory, PNG/pixel audit, explicit-patient loader, development one-update exercise, cohort creation/validation, train-only reference previews, audit-bound PNG training/validation and PNG checkpoint save/load. The flat-layout local suite passed **188 engineering tests**, including 16 metadata-audit fixture tests. All 13 core model-package Python files were SHA-identical to the existing source during migration. GitLab server CI has not run; these checks are separate from real-data performance.

The reviewed local data audit covered 20,686 paired selected slices from 210 cases. Its fixed cohort contains 20 training patients/2,071 slices, 10 validation/1,156, 10 final test/530 and five independent prediction/676. `case_00000` is training because it was exposed to development. The 40 modelling patients and five prediction patients remain separate. The original 147/32/31 split is unchanged.

Real PNG training and validation have not been completed. Training reference preview generation does not establish completed manual anatomical review. No final PNG test, independent PNG prediction, clinical performance evidence, calibrated confidence or patient-level healthy/tumour diagnosis is claimed.

## Actual Python boundary

The following are source interfaces, not HTTP endpoints:

| Interface | Current role |
| --- | --- |
| `PNGConfig`, `PNGPatientDataset` in `png_dataset.py` | Explicit patient PNG loading; one input channel; separate integer target; identity/source hashes |
| `audit_png_dataset` in `png_audit.py` | Programmatic file, pixel, label/CSV, content and original-split checks |
| `make_cohort`, `validate_cohort` in `cohort.py` | Deterministic audit-bound 20/10/10 + five prediction assignments and leakage controls |
| `load_training_context` in `png_training.py` | Validate cohort/audit/manifest/original-split/preprocessing bindings without decoding PNGs |
| `train_png` in `png_training.py` | Fresh train/validation run; returns `model`, `history`, `metadata`, `proof`, `best_validation_report`; caller persists artifacts |
| `evaluate_png_validation` in `png_training.py` | Only an audited validation dataset from the fitted cohort; no optimizer work |
| `save_png_checkpoint`, `load_png_checkpoint` | Create/restore `png_segmentation_v1`; loader returns `(model, config, metadata)` |
| `learning/06b_train_png.py` | CPU orchestration, preflight, progress, failure/interruption records and non-overwriting output directory |

Inputs are `[B,1,256,256]` float32 CTs with `(PNG/255−0.456)/0.224`; targets are `[B,256,256]` int64 IDs 0/1/2; logits are `[B,3,256,256]`. Patient and original slice identities must accompany every eventual output asset. PNG intensities are already processed uint8, not HU.

## Fields available for a future backend adapter

These are the actual local structures that an adapter can map. There is currently no committed shared JSON envelope, request route or public asset endpoint in this module.

| Local result | Actual fields useful for integration |
| --- | --- |
| Training progress callback | `event`, `timestamp_utc`, `phase`, `epoch`, `epochs`, `batch_index`, `total_batches`, `slices_seen`, `total_slices`, `loss`, `progress`; event-specific fields vary |
| Epoch history | `epoch`, `epochs`, `train_loss`, `validation_loss`, `train_slices_seen`, `validation_slices_seen`, `validation_patient_mean`, `optimizer_steps_completed`, `metric_scope`, timestamps |
| Validation report | `unit`, `patient_count`, `patient_mean`, `contributing_patient_counts`, `patients`, `absent_class_policy`, `split`, `slice_count`, `loss`, `loss_scope`, `metric_scope`, `model_selection_only` |
| Each patient report | `patient_id`, `exported_slice_count`, `per_class`, `confusion_matrix`, `confusion_matrix_convention="rows_actual_columns_predicted"` |
| Each class result | `dice`, `iou`, `precision`, `recall`, `f1`, `support_pixels`, `predicted_pixels`, `true_positive`, `false_positive`, `false_negative` |
| Checkpoint metadata | `schema_version`, `architecture`, `input_mode`, `input_channels`, `classes`, `preprocessing`, `preprocessing_sha256`, `audit`, `cohort`, `pretrained_proof`, `best_epoch`, `best_validation_loss`, state/data fingerprints and limitations |
| CLI run summary | `status`, `configuration`, `patient_counts`, `selected_slice_counts`, audit/cohort fingerprints, `reference_review`, `formal_training_completed`; successful runs add checkpoint/artifact hashes and validation results |

Undefined metric denominators produce `null`, and means exclude those entries while reporting contributor counts. Loss is a slice-weighted minibatch CE+soft-Dice mean; reported overlap metrics use hard argmax predictions and patient-balanced means. Best checkpoint selection uses validation loss, not final-test scores.

Do not expose raw checkpoint metadata as a public API response: it contains local paths and extensive audit/source records. A shared service must map approved fields into its own response schema and asset registry.

## Capabilities to advertise accurately

The stable project identity is **Kidney Tumour**, task **segmentation**, three classes background/kidney/tumour. Those are domain facts for a future capability response, not implemented `model_id`/`task` JSON keys here. A team adapter must agree the final identifiers with the Diabetes contributor rather than inventing an available Diabetes implementation.

| Capability | Dedicated PNG path status |
| --- | --- |
| Audit and patient-cohort checks | Implemented |
| Training-only reference previews | Implemented; manual review remains explicit/pending |
| Frozen-encoder training + validation | Implemented and fixture tested; real run pending |
| PNG checkpoint save/load | Implemented; generated weights remain local |
| Final-test PNG evaluation | Follow-up; test pixels remain unused by Step 6B |
| Independent new-patient PNG prediction | Follow-up; five prediction cases remain reserved |
| Predicted segmentation overlays/explanation | Follow-up for the PNG backend path; reference overlays are not predictions |
| HTTP API, job persistence, public asset serving, frontend wiring | Follow-up |
| Diabetes adapter/shared envelope | Follow-up; separate model/task, no multimodal fusion |

There is no region “confidence” or patient-level malignant/healthy label to map. The legacy/synthetic demo code elsewhere is not evidence that this PNG contribution integrates with the actual team frontend.

## Portability and Git boundaries

Run commands from `backend/kidney`, with an explicit absolute `--root /absolute/path/to/kits_project`; see [the module README](../../backend/kidney/README.md). Do not use the old sibling-data default. Bind local audit/cohort/review files through explicit arguments when they remain in the original starter. No code migration requires editing or copying the actual dataset.

Only source, tests, configuration and English documentation belong in this branch. Keep CTs/masks, `.venv`, Hugging Face caches, encoder/model weights, checkpoints, previews, audit/run outputs, logs and `learning_outputs/` outside Git. Small reviewed metadata copies can be considered later through a deliberate team decision; this source contribution does not include them.

## Next work, in order

1. Run the migrated engineering suites and CI in the personal feature-branch checkout; inspect the source-only diff against `develop`.
2. Inspect the Step 6A training previews and record any visible pairing/alignment problem without claiming medical label certification.
3. Run the user-started Step 6B baseline on the unchanged locked train/validation pools; retain actual history, progress, provenance and interruption/completion status locally.
4. Add final PNG testing for the frozen validation-selected model, then independent new-patient prediction with identity-bound predicted overlays. Keep the final test separate from model selection.
5. Agree backend contracts and capability/error/asset fields with the frontend and Diabetes contributor. Implement the shared service adapter, persistence and input/output validation, then exercise the real application flow.
6. Reconcile Proposal, Jira acceptance criteria and the actual Figma stages against that implemented flow before updating completion claims or requesting merge review.

This handoff records a source contribution and the remaining work. It does not mark a Jira story done or declare the branch ready to merge.
