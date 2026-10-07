# Step 6F: A Fixed Ten-Epoch Training Budget

Run04 completed five epochs using relative CE class weights [1,1,2] and tumour-Dice checkpoint selection. The retained epoch is 4, not the final epoch. Validation patient-mean tumour Dice improved from run03's 0.135298 to 0.217023, recall from 0.113804 to 0.210563, and kidney Dice from 0.842071 to 0.865631. Correct tumour pixels appear in five rather than four validation patients. False-positive tumour pixels also increased from 91,083 to 118,261. These pooled error counts must not be confused with patient-mean scores.

The fixed run04 history is:

| Epoch | Training loss | Validation loss (weight 2) | Tumour Dice |
| --- | ---: | ---: | ---: |
| 1 | 0.489318 | 0.331513 | 0.000033 |
| 2 | 0.338995 | 0.439047 | 0.060692 |
| 3 | 0.245505 | 0.330543 | 0.180697 |
| 4 | 0.203797 | 0.287314 | 0.217023 |
| 5 | 0.183477 | 0.275922 | 0.151216 |

Decreasing loss does not ensure improving hard tumour Dice. A single decline does not establish overfitting either. Run05 tests whether a ten-epoch budget improves this fixed model and objective. It does not promise that more epochs will help.

## Fixed protocol before execution

Change only epochs from 5 to 10. Keep CE tumour weight 2, equal foreground soft-Dice, seed 42, batch size 2, learning rate 0.001, CPU two threads, frozen encoder, all training slices once per epoch and patient-mean tumour-Dice selection unchanged. Preserve the audited cohort, preprocessing and reference-guided diagnostic-example policy. Final-test and independent-prediction PNGs remain unused.

Run05 starts fresh. The current checkpoints do not contain the optimizer state needed to continue the prior training process; the learning entry does not implement resume. Do not edit run04 metadata or move its checkpoint into a new directory to claim another ten-epoch run.

```sh
source .venv/bin/activate
python learning/06b_train_png.py --root ../kits_project --epochs 10 --batch-size 2 --learning-rate 0.001 --device cpu --cpu-threads 2 --seed 42 --selection-metric tumour_dice --tumour-ce-weight 2 --output-dir learning_outputs/step6b/run05 --check-only
python learning/06b_train_png.py --root ../kits_project --epochs 10 --batch-size 2 --learning-rate 0.001 --device cpu --cpu-threads 2 --seed 42 --selection-metric tumour_dice --tumour-ce-weight 2 --output-dir learning_outputs/step6b/run05
python learning/06c_review_predictions.py --root ../kits_project --run-dir learning_outputs/step6b/run05 --output-dir learning_outputs/step6c/run05
```

Run from the original Starter after activating its environment. Continue only after the preceding command succeeds. Team-layout users must supply absolute external data, cohort, audit, reference-review and output paths as described in Step 6B. No model code or dependencies change in Step 6F.

## Expected evidence and decision

Ten epochs process 2,071 training and 1,156 validation slices each time, with 10,360 optimizer updates. The saved epoch is the best tumour-Dice candidate from epochs 1–10; it may remain epoch 4. Step 6C still creates 40 training and 20 validation diagnostic pairs, with zero final-test/prediction pairs. Output directories must be new.

Under the same runtime/data/seed, the first five epochs are expected to reproduce run04's optimization measurements. Verify those observations rather than forcing equality. If they differ, investigate configuration and provenance before interpreting the epoch-budget comparison. The training loop uses a fixed learning rate, so the longer declared budget does not introduce a schedule change.

Compare retained run04/run05 validation reports and the complete history. Their loss definitions are the same, but checkpoint selection remains tumour Dice. Track all ten patients, precision contributor counts, kidney Dice, and true/false positive and false-negative tumour counts. In run04, `case_00160` accounts for 97,636 tumour false-positive pixels; `case_00059` has 9,603 tumour predictions with zero true positives. Five patients still have zero tumour Dice.

Complete the predeclared ten epochs without extending the budget based on an intermediate result. Then consolidate the current candidates: if later epochs do not improve the selected model, retain the earlier candidate. Mean Dice improvement must be considered alongside false positives and kidney performance. Do not mechanically add epochs or several new factors. Document the selected model and its limitations, freeze its weights/configuration, then prepare the final-test entry for the ten reserved patients. This step itself does not run a final test or close any API/frontend/Jira work.
