Title: feat(kidney): add frozen evaluation and CT-only prediction

Suggested source branch: `feat/yining-kidney-evaluation-inference`

Target branch: `develop`

Primary story: H18JDONUT-7

Supporting stories: H18JDONUT-5, H18JDONUT-20, H18JDONUT-15, H18JDONUT-21

The Kidney backend baseline could train and validate, but lacked the dedicated PNG route for freezing a selected model, evaluating its reserved test patients and producing CT-only prediction assets. This MR completes those local model stages while preserving the audited patient split and separate Diabetes scope.

The change adds reference-guided train/validation diagnostic previews, optional patient-mean tumour-Dice checkpoint selection and relative tumour CE weighting with unchanged defaults, strict checkpoint evidence validation, metadata-only model freezing, a recorded one-time fixed-test execution, and an image-only prediction loader. Prediction saves class-ID masks, coloured segmentation overlays, identity-selected patient previews and a hashed JSON manifest. The final-test/prediction paths verify original model state and input bindings; progress and failure/interruption records preserve local execution evidence.

Validation:

- 481 local engineering/CLI tests passed after the review repair, including five new partial-selection/legacy compatibility regressions.
- Genuine legacy checkpoint metadata remains compatible and unchanged. Incomplete new selection evidence is rejected before model construction, including after a metadata fingerprint is recomputed.
- Local run05 completed ten epochs, retaining epoch 10 by validation tumour Dice. Patient-mean validation Dice: kidney 0.867657; tumour 0.253372.
- The frozen test completed 10 patients/530 exported slices once, with unchanged model state and zero optimizer updates. Patient-mean test Dice: kidney 0.876164; tumour 0.368032. Four test patients have zero tumour Dice.
- CT-only prediction completed 5 separate patients/676 CTs with no reference-mask pixel reads, training or test rerun. All 1,357 generated output PNG hashes/formats/counts were audited; outputs comprise 676 label masks, 676 overlays and 5 fixed previews.
- Completed checkpoint/data/test bindings and recomputed patient metrics passed read-only checks. The same frozen checkpoint is used throughout. Details and limitations are in `docs/kidney/RUN05_MODEL_RESULTS.md`.
- Existing GitLab CI discovers the new synthetic checks without real data or model weights. The new branch's server pipeline must be checked after publication.

Scope and remaining work:

- This MR contains source, synthetic tests and English documentation, including a small reviewed results summary. CTs/masks, weights, environments, caches and generated local reports/assets remain excluded.
- Metrics describe reference-selected exported 2D slices. The tumour cohort does not support healthy-versus-tumour patient classification. The overlay is a model segmentation result, not Grad-CAM, calibrated confidence or a clinical diagnosis.
- Shared HTTP endpoints, request/job persistence, public asset URLs, actual frontend integration and the separate Diabetes adapter remain follow-up work. Local file paths are not public API URLs.
- No Jira user-facing acceptance criterion is closed by this source merge alone. Current manual anatomical review status is preserved.
