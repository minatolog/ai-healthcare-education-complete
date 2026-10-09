# Kidney Backend Handoff

Source module: `backend/kidney`. Target integration branch: `develop`. The next suggested feature branch is `feat/yining-kidney-evaluation-inference`; creating it does not change the reviewed data or model.

The implemented route is one grayscale 256x256 CT PNG per input, with a verified pretrained frozen ResNet-18 encoder and trainable U-Net decoder/head, producing background/kidney/tumour pixel labels. The full-volume/2.5D reference route is separate. Kidney and Diabetes remain independent tasks without multimodal fusion.

## Completed local evidence

The reviewed patient lock is 20 training/2,071 selected slices, 10 validation/1,156, 10 final test/530 and five additional prediction patients/676. All slices for a patient share one pool; the original 210-case legacy split remains unchanged.

Local run05 trained for the predefined ten epochs with relative CE tumour weight 2, fixed seed 42, batch size 2 and frozen encoder. Epoch 10 was selected by maximum patient-mean validation tumour Dice before freezing. Final test completed once on all 530 locked test slices, with kidney Dice 0.876164 and tumour Dice 0.368032. Four test patients have zero tumour Dice. CT-only prediction completed all 676 prediction CTs, producing predicted masks, model overlays and five fixed previews; original model state and test evidence stayed unchanged.

See [the reviewed results summary](RUN05_MODEL_RESULTS.md) for fingerprints, metric contributors and limitations. Raw data, weights and generated run evidence remain local. The original MR !3 server CI passed; the next published MR requires its own pipeline result. Current fixture-test totals are recorded in [the next MR description](NEXT_MR_DESCRIPTION.md).

## Actual Python interfaces

These are callable local interfaces, not committed HTTP endpoints.

| Module/interface | Role |
| --- | --- |
| `png_dataset.PNGConfig`, `PNGPatientDataset` | Single-channel normalization; explicit patient loading with separate reference target |
| `png_audit.audit_png_dataset`, `cohort.make_cohort`, `validate_cohort` | Machine file/pixel/CSV audit and deterministic audit-bound patient pools |
| `png_training.load_training_context` | Current cohort/audit/manifest/legacy-split/preprocessing bindings; no PNG decode |
| `png_training.train_png` | Fresh train/validation run; optional selection metric/CE tumour weight; returns model, history, metadata, proof and selected validation report |
| `png_training.evaluate_png_validation` | Fixed validation scoring without optimizer updates |
| `png_training.save_png_checkpoint`, `load_png_checkpoint` | Save/restore `png_segmentation_v1`; verify complete state/provenance and metadata evidence |
| `png_final_test.inspect_completed_run`, `make_freeze_record`, `validate_freeze_record` | Metadata/tensor inspection and exact model/configuration/pool freeze |
| `png_final_test.inspect_frozen_model`, `evaluate_frozen_test` | Check persisted freeze; evaluate every fixed test row once with unchanged model state |
| `png_prediction.inspect_prediction_inputs`, `build_prediction_dataset` | Frozen CT-only prediction identities/paths; no PNG decode in preflight |
| `png_prediction.predict_frozen_pool` | CT-only prediction, labels/overlays/fixed previews and persisted JSON report |

Learning CLI commands wrap these interfaces with portable external paths, preflight, progress, non-overwriting outputs and failure/interruption records. The final-test CLI also records a persistent access ledger. Original checkpoints and completed test/prediction records must not be edited to bypass workflow controls.

Inputs are `[B,1,256,256]` float32 CT tensors normalized as `(PNG/255-0.456)/0.224`. Supervised targets are `[B,256,256]` int64 labels 0/1/2. Logits are `[B,3,256,256]`; frozen inference uses hard argmax. PNG intensities are already processed uint8, not raw HU.

## Local fields available to the shared adapter

| Result | Actual useful fields |
| --- | --- |
| Progress | `event`, `timestamp_utc`, `phase`, epoch/batch/slice counts, `loss`, `progress`, optimizer counts as applicable |
| Training history | Epoch losses, `validation_patient_mean`, `validation_contributing_patient_counts`, `selection_metric`, `selection_score`, `checkpoint_selected`, `tumour_ce_weight`, update counts |
| Validation/final-test metrics | `patient_count`, `patient_mean`, `contributing_patient_counts`, `patients`, slice count, scope and hashes |
| Patient/class metric | `patient_id`, `exported_slice_count`, `per_class`, confusion matrix; Dice/IoU/precision/recall/F1, supports, TP/FP/FN |
| Frozen record | Checkpoint/state/configuration/data fingerprints, selected epoch/policy, fixed pools and preprocessing |
| CT-only prediction report | `patient_ids`, `patients`, `slice_count`, `patient_slice_counts`, `classes`, `overlay`, model/freeze hashes and scope flags |
| Patient prediction | `patient_id`, `exported_slice_count`, `slices`, `preview` |
| Slice prediction | `sample_id`, `slice_index`, source image hashes, `predicted_class_counts`, `predicted_mask`, `model_overlay` |
| Prediction asset | `path`, `sha256`; preview also contains `sample_ids` and `selection_rule` |

Undefined metric denominators are `null`; means exclude them and report contributor counts. Metrics are computed from hard labels per patient then averaged across patients. Default checkpoint selection uses validation loss; the explicit tumour-Dice route records its ranking/tie policy. Test and prediction results never select checkpoint/threshold settings.

Do not expose the full checkpoint metadata or local prediction paths directly as a public response. They include machine paths and extensive audit records. The shared service needs an approved envelope, model/task identifiers, validated requests, job states and an asset registry mapping local generated files to stable public URLs. Those are future adapter contracts, not existing response fields.

## Capability status

| Capability | Status |
| --- | --- |
| Audit, patient lock and training reference previews | Implemented; manual anatomical review remains pending |
| Frozen-encoder training/validation and compatible PNG checkpoint | Implemented and executed locally |
| Frozen PNG final test | Implemented and executed locally |
| CT-only independent prediction and segmentation overlays | Implemented and executed locally; no reference accuracy or diagnosis claim |
| Calibrated confidence, patient-level healthy/tumour diagnosis | Not implemented or supported by this cohort |
| Shared HTTP routes, jobs, uploads and public asset serving | Pending |
| Actual frontend wiring and Diabetes adapter/shared envelope | Pending |

The model overlay is segmentation, not Grad-CAM. Reference-based PNG export means all results describe selected 2D slices, not complete-scan or external clinical performance. Hashes/counts do not certify anatomical correctness or external source aliases.

## Next integration work

1. Publish the reviewed source MR into `develop`, inspect its Changes and wait for the new server CI.
2. Read the actual team backend/frontend and Diabetes contributor's code before defining shared capability, job, metric, prediction, asset and error contracts.
3. Implement the backend adapter/asset serving and exercise train progress, validation/test evaluation and CT-only prediction through the real frontend flow.
4. Reconcile Proposal/Jira/Figma wording with segmentation and the fixed 20/10/10 plus separate prediction pool before marking user-facing acceptance criteria complete.

Run team commands from `backend/kidney` with explicit absolute data/evidence paths; see [the module setup](../../backend/kidney/README.md). The later [40-patient data handoff](../../data/kidney/README.md) now versions the reviewed processed CT/mask subset and its original metadata/audit/lock under `data/kidney/`. Its current demo selection uses the ten test patients; the historical five-patient prediction pool remains in the unchanged lock for provenance. Raw scans, environments, caches, checkpoints, logs, previews and generated `learning_outputs/` remain external. Existing first-MR documents and completed-run results remain historical records.
