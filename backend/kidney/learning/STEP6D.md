# Step 6D: Select the Checkpoint for the Tumour Task

The initial PNG entry selects the epoch with the lowest validation loss. In the completed local run02, epoch 4 had the lowest loss (0.283585), while epoch 5 had higher patient-mean tumour Dice (0.135298 versus 0.108358). The saved run02 checkpoint therefore remains epoch 4. The epoch 5 weights were not saved; changing a JSON field cannot recover them.

The optional `--selection-metric tumour_dice` now selects the highest **patient-mean validation tumour Dice**. Ties use lower validation loss, then the earlier epoch. Zero Dice remains a valid candidate; undefined, non-finite or unsupported metrics fail explicitly rather than silently changing selection policy. The default remains `validation_loss` for earlier commands and checkpoints.

This changes checkpoint retention, not the loss, gradient updates, data sampling or encoder training. It does not independently improve the learning algorithm. Use the same five-epoch budget and seed for this controlled comparison. Inspect tumour precision/recall, individual patients and kidney results alongside Dice, since larger red regions can include false positives.

## Run from the original Starter

```sh
source .venv/bin/activate
python learning/06b_train_png.py --root ../kits_project --selection-metric tumour_dice --check-only
python learning/06b_train_png.py --root ../kits_project --epochs 5 --batch-size 2 --learning-rate 0.001 --device cpu --cpu-threads 2 --seed 42 --selection-metric tumour_dice --output-dir learning_outputs/step6b/run03
python learning/06c_review_predictions.py --root ../kits_project --run-dir learning_outputs/step6b/run03 --output-dir learning_outputs/step6c/run03
```

Run only after the previous command succeeds. From `backend/kidney`, use the module environment and absolute external `--root`, `--cohort`, `--audit`, `--reference-review`, `--run-dir` and `--output-dir` paths, as described in Step 6B/6C. The original Starter environment and evidence are not part of GitLab.

The first command is metadata/cache checking only. The second starts a **fresh** decoder/head with the same verified pretrained encoder; it does not resume run02. The final command uses the newly selected checkpoint to create reference-guided train/validation diagnostic images.

Expected completion remains five epochs and 5,180 optimizer steps. The footer adds `Checkpoint selection: tumour_dice; selected epoch: <actual epoch>`. If the trajectory reproduces run02 exactly, epoch 5 is the highest tumour-Dice candidate among those five epochs; use actual recorded values rather than promising equality or improvement.

## Evidence and compatibility

`training_history.json` records every epoch. `run_summary.json` and checkpoint metadata record the selection policy, score and selected epoch. `validation_report.json` belongs to the **saved** epoch. The existing `best_validation_loss` field describes that selected checkpoint's loss; the separate lowest-loss observation records the minimum seen across the run.

New metadata is validated against the selected report and full history, including the tie rule. Existing `png_segmentation_v1` checkpoints without selection-policy fields remain readable without rewriting their fingerprints or original evidence. Step 6C continues to load `best_model.pt`; it never treats another epoch's history entry as that checkpoint's performance.

Preserve run01 and run02. Output directories must be new. Final-test and independent-prediction images remain untouched. Training artifacts stay outside Git.

The next learning experiment, if tumour results remain weak, should change one training factor such as a carefully defined loss weight while keeping selection policy and other settings fixed. Its protocol must be specified before running; no such loss or sampling change is introduced in this selection step.
