# Step 6E: A Controlled Tumour CE Weight Experiment

Run03 successfully retained epoch 5 using patient-mean validation tumour Dice. Its five-epoch optimization measurements exactly reproduce run02; only checkpoint selection changed. The retained model has kidney Dice 0.842071, tumour Dice 0.135298 and tumour recall 0.113804 across ten validation patients. Only four patients have true-positive tumour pixels. Four patients have no tumour prediction, so tumour precision is undefined for those patients; its mean uses six contributors. These are selected exported-slice results, not full-scan or diagnostic performance.

Run04 changes one training factor: the relative cross-entropy class weights become `[1, 1, 2]` for background, kidney and tumour. The equal foreground soft-Dice term remains unchanged. Five epochs, seed 42, batch size 2, learning rate 0.001, CPU two threads, all selected slices once per epoch, frozen encoder and tumour-Dice selection remain fixed. This tests whether emphasizing reference tumour pixels helps this model; it does not establish that class imbalance caused its errors. Training tumour and kidney pixel fractions are similar (1.322% and 1.379% respectively), while background is 97.299%.

The optional flag is `--tumour-ce-weight 2`. Its default `1` follows the original unweighted loss path exactly. Positive finite values are supported; invalid values fail before training. No extra dependencies are needed.

## Run from the original Starter

```sh
source .venv/bin/activate
python learning/06b_train_png.py --root ../kits_project --selection-metric tumour_dice --tumour-ce-weight 2 --check-only
python learning/06b_train_png.py --root ../kits_project --epochs 5 --batch-size 2 --learning-rate 0.001 --device cpu --cpu-threads 2 --seed 42 --selection-metric tumour_dice --tumour-ce-weight 2 --output-dir learning_outputs/step6b/run04
python learning/06c_review_predictions.py --root ../kits_project --run-dir learning_outputs/step6b/run04 --output-dir learning_outputs/step6c/run04
```

Continue only after the preceding command succeeds. Run04 starts fresh and does not resume run03. Preserve all prior run directories. From the team module, use absolute external data/evidence paths as described in Step 6B. This guide does not start a training run.

## What the weight changes

For integer class targets, PyTorch's mean cross entropy is the sum of weighted pixel losses divided by the sum of the target-class weights in that minibatch. `[1,1,2]` changes the tumour class's relative contribution; it does not multiply the whole loss by two or guarantee that every tumour gradient doubles. The soft-Dice equation, foreground classes and smoothing stay fixed. See the [PyTorch 2.8 CrossEntropyLoss definition](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html).

Train and validation use the same configured loss. Validation loss is still a slice-weighted mean of minibatch objective values, not a single globally weighted pixel loss. Because the objective changes between run03 and run04, their raw validation-loss values are not a common measure of model quality. Compare unchanged hard-argmax patient segmentation metrics and individual error counts instead.

## Expected evidence

- The start identifies relative CE weights `[1.0, 1.0, 2.0]` and unchanged foreground Dice.
- Five epochs still perform 5,180 optimizer updates; no final-test or prediction PNG is loaded.
- The selected epoch is the highest validation patient-mean tumour-Dice candidate, with the same loss/earlier-epoch tie rule as Step 6D. Its actual epoch and score cannot be predicted.
- Run configuration, per-epoch history, selected validation report and checkpoint metadata record the loss settings. Older v1 checkpoints remain readable without rewriting their original evidence.
- Step 6C produces 40 training and 20 validation diagnostic pairs, with zero test or prediction pairs.

## Compare before deciding on another change

Use the retained run03 and run04 validation reports, each of which belongs to its own saved checkpoint. Compare tumour Dice/recall for all ten patients, precision with its contributor counts, patients with no tumour prediction, true positives, false positives and false negatives. Track kidney metrics too. In run03, `case_00160` contributes 87,820 of 91,083 tumour false-positive pixels; 77,125 were reference kidney pixels. Increasing tumour weight may worsen this confusion even if mean Dice improves.

If run04 fails to improve the tradeoff, preserve it as an experiment and retain the better earlier candidate. Do not tune against final-test patients. Any next change to epochs, sampling, augmentation or encoder training should be a separately documented experiment. Final test, independent prediction and the shared Kidney/Diabetes HTTP/front-end integration remain later stages.
