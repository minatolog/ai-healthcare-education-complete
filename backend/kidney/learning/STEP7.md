# Step 7: Freeze Run05 and Evaluate the Reserved Test Pool

Run05 completed the predefined ten-epoch budget. Its first five epochs exactly reproduced run04's training and validation measurements. Under the existing patient-mean validation tumour-Dice criterion, epoch 10 is the selected candidate. No additional training factor or threshold is selected in this step.

| Validation measurement | Run04, epoch 4 | Run05, epoch 10 |
| --- | ---: | ---: |
| Kidney patient-mean Dice | 0.865631 | 0.867657 |
| Tumour patient-mean Dice | 0.217023 | 0.253372 |
| Tumour patient-mean recall | 0.210563 | 0.261521 |
| Tumour false-positive pixels, all validation patients | 118,261 | 156,734 |

The higher mean Dice comes with more false positives. Three validation patients have zero tumour Dice, and two other patients have very few correctly labelled tumour pixels. The reference-guided diagnostic previews show misses and predictions outside the reference regions. This is a limited educational baseline, not a reliable patient-diagnosis model. Patient means and pooled pixel counts describe different aggregations.

## What the two commands do

Step 7A validates the completed run, selected checkpoint tensors, reports, current manifest/audit/cohort bindings and configuration. It writes `frozen_model.json`, recording the exact checkpoint, selected epoch, preprocessing, loss, selection rule and fixed patient pools. It references the original `best_model.pt`; it does not copy or modify weights, initialize a model, decode PNGs or open the test pool. Run05's checkpoint SHA256 is `24dbf2c5048b87976586b9d5b5f7a8c183312d1ef1cda51238630e88b3ab7914`.

Step 7B restores only that frozen checkpoint in evaluation mode and processes all 530 exported pairs from the 10 locked test patients once, without shuffling or optimizer updates. CT is the model input; reference masks are used for scoring. Hard argmax remains the inference rule. It reports per-patient confusion matrices and patient-mean Dice, IoU, precision, recall and F1, including contributor counts for undefined metrics. The fixed weighted loss is descriptive only. All model parameters/buffers and original evidence must retain their fingerprints.

The 20 training patients, 10 validation patients and five independent-prediction patients are not decoded during Step 7B. Test scores cannot select another epoch, threshold, loss or training run. Preserve run04 as comparison evidence; it is not evaluated on test to choose between candidates.

## Run from the original local Starter

The prepared local environment already supplies the model dependencies. Keep the data and all run01–run05 evidence in place. The output directories below must be new.

```sh
cd /Users/realnin/9900/Kidney_Segmentation_Demo_Starter
source .venv/bin/activate

python learning/07a_freeze_model.py --root ../kits_project --check-only && \
python learning/07a_freeze_model.py --root ../kits_project && \
python learning/07b_evaluate_test.py --root ../kits_project --check-only && \
python learning/07b_evaluate_test.py --root ../kits_project
```

Both `--check-only` commands inspect metadata and saved tensors without model initialization, PNG decoding, test opening or output writing. Successful Step 7A prints `[MODEL FROZEN]`. Successful Step 7B prints `[FROZEN PNG FINAL TEST COMPLETED]`, `Patients=10; selected slices=530` and the actual test patient means. Batch size 2 produces 265 inference batches; progress is printed periodically. No test metric value is predicted in advance.

Output evidence:

| File under local `learning_outputs/` | Meaning |
| --- | --- |
| `step7a/run05/frozen_model.json` | Exact selected model/configuration/pool snapshot |
| `step7a/run05/test_access.json` | Persistent access state for this freeze: reserved/running/completed/failed/interrupted |
| `step7b/run05_test/progress.jsonl` | Recorded final-test progress, including the last available event |
| `step7b/run05_test/evaluation_summary.json` | Execution state and completed report hash |
| `step7b/run05_test/test_report.json` | Complete patient metrics, counts, fixed input/model hashes and zero-update evidence |

The access record is created exclusively before model/pixel access, preventing accidental concurrent or repeated evaluation of the same freeze even with another output directory. If execution fails or is interrupted, preserve the access record and partial output and bring the error for software debugging. Do not delete evidence or rerun training to improve a revealed test score. A recovery, if necessary, must be documented explicitly.

If Step 7A has already succeeded but Step 7B has not opened the test pool, continue with the two Step 7B commands rather than recreating the freeze. Once Step 7B completes, inspect the saved report; rerunning is unnecessary.

## Running from the team checkout

From `backend/kidney`, supply absolute external paths for `--root`, `--cohort`, `--audit` and `--run-dir`, and choose local ignored output directories. Step 7B's `--freeze` must reference the snapshot created by Step 7A. The defaults above are designed for the original Starter layout. This source contribution contains no CT/mask files, pretrained/model weights or real generated results.

## Interpretation and following work

These are segmentation metrics on reference-selected exported 2D slices, not complete-volume performance, external clinical validation or healthy-versus-tumour patient classification. Machine hashes/counts do not certify anatomical correctness or resolve the pending manual reference-review status. Test results are reported with these limits regardless of their value.

After this single frozen-model evaluation, preserve the selected model and result. Prepare CT-only inference on the separate prediction pool, then the shared service contract and Kidney/Diabetes/frontend adapters. This step does not implement those adapters or close a Jira user-facing acceptance criterion.
