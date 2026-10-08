# Frozen Run05: Local Kidney Segmentation Results

The model is a pure single-slice three-class segmentation baseline: frozen verified ImageNet ResNet-18 encoder and trainable U-Net decoder/head. It uses existing 256x256 grayscale CT PNGs with fixed normalization, not patient-level healthy-versus-tumour classification. KiTS19's tumour cohort and reference-guided PNG export define the data scope.

## Fixed model and cohort

Training used 20 patients/2,071 selected slices per epoch, validation 10 patients/1,156 selected slices, and final test 10 patients/530 selected slices. Five additional patients/676 selected slices form a separate prediction pool. Patient/source/content overlap controls follow the saved audit and cohort lock. The original 210-case split is preserved.

Run05 used seed 42, ten epochs, batch size 2, CPU two threads, Adam learning rate 0.001, all training rows shuffled once per epoch, CE relative weights [1,1,2], and equally averaged kidney/tumour soft-Dice. Its first five optimization histories reproduced run04. The retained checkpoint is epoch 10 under the predeclared maximum patient-mean validation tumour-Dice rule, with lower validation loss then earlier epoch as tie-breaks.

Checkpoint SHA256: `24dbf2c5048b87976586b9d5b5f7a8c183312d1ef1cda51238630e88b3ab7914`.

Saved model-state SHA256: `d3e4dcff5b0b42c31903b68b65d62414215880ab874ada80ffdfd62a38f1bea9`.

## Patient-mean metrics

| Split / class | Dice / F1 | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Validation kidney | 0.867657 | 0.786217 | 0.927260 | 0.842048 |
| Validation tumour | 0.253372 | 0.180801 | 0.297211 | 0.261521 |
| Final test kidney | 0.876164 | 0.786589 | 0.899531 | 0.860621 |
| Final test tumour | 0.368032 | 0.272769 | 0.524337 | 0.378970 |

Metrics are computed from all exported pixels per patient, then averaged over patients with a defined value. Validation tumour precision has nine contributors; final-test tumour precision has eight because two patients predict zero tumour pixels, giving undefined precision (`null`). Final-test tumour Dice, IoU, recall and F1 have all ten contributors. F1 equals Dice under these pixel definitions. Validation and test contain different patients; higher test scores do not represent learning during test.

Final-test tumour pooled errors are TP=66,185, FP=42,623 and FN=47,561 (support=113,746 pixels). These totals are distinct from patient-mean metrics. Four test patients have zero tumour Dice: case_00046, case_00170, case_00019 and case_00077. The strongest test patient has tumour Dice 0.841788; substantial misses and false positives remain. Do not report background dominance as evidence of successful tumour segmentation.

## Verification and interpretation

The frozen test evaluated exactly 530 samples once, covering 34,734,080 pixels. The saved checkpoint/freeze/data/evidence hashes agree; model-state fingerprints before and after inference match the original. The recorded run contains zero optimizer steps and no training, selection or independent-prediction access. A read-only audit independently recomputed patient metrics from confusion matrices, checked patient/sample identities against the bound metadata, and verified all report/artifact hashes without rerunning inference or rereading test PNG pixels. Source byte/pixel verification was recorded by the evaluation itself.

This is a limited educational segmentation baseline on reference-selected exported 2D slices. It does not establish blind complete-volume or clinical performance, patient-level diagnosis, calibration or performance on healthy patients. Machine integrity checks do not certify anatomical annotation correctness; the prior manual review status remains pending. Preserve this frozen candidate and report its test results without using them for further model/threshold selection.

Raw CT/mask data, model/pretrained weights and generated local reports/assets stay outside Git. This small summary documents completed local evidence; it does not claim shared HTTP/frontend/Diabetes integration or close their Jira acceptance criteria. Independent CT-only prediction is also complete as described below.

## Independent CT-only prediction

The same frozen checkpoint processed all 676 CT images from the five separate prediction patients once, with unchanged grayscale preprocessing and hard argmax. Reference-mask PNGs were not opened; no ground-truth metrics, test rerun, training or threshold tuning occurred. It saved 676 class-ID label PNGs (L mode, 256x256), 676 model overlays (RGB, 256x256), five first/middle-identity previews (RGB, 536x614), a JSON report and progress/completion records.

A read-only audit verified the exact 1,357 output PNG path/hash set, class IDs/counts, overlay colors, fixed preview identities and all original model/data/test bindings. The model-state fingerprint before and after remains the original hash above. Preview review checked rendering and CT/model presentation without reference labels; it does not establish segmentation accuracy. Some preview frames show no predicted tumour while other slices for that patient do.

Prediction report SHA256: `407de6a6b0a8984461e4c05b32b7f60ef67a03df76a7e37933cfefdd3cdc1cda`. These patients are separate from fitting/selection/test but share the KiTS19 source and reference-selected PNG export; this is not external clinical validation. Outputs are region predictions, not a calibrated risk score or a healthy/tumour patient classification.
