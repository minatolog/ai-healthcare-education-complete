# Kidney Data Handoff

This MR supplies the actual processed KiTS19 CT images and reference masks needed by the data-processing teammate. `data/kidney/` contains the same fixed 40-patient cohort used by the existing Kidney model: 20 training patients/2,071 pairs, 10 validation/1,156 and 10 test/530. The 3,757 pairs are 7,514 unchanged 256x256 grayscale PNGs (87,913,810 bytes).

The handoff adds a filtered working manifest with explicit pool roles, a patient table, English processing instructions, data attribution/license, source-processing snapshots and a SHA-256 catalogue. It preserves the original full manifest, legacy split, audit and immutable cohort lock byte for byte so the existing model bindings remain usable. The original lock's five legacy prediction cases are documented but their PNGs are omitted; the current demo selects from the ten test patients.

Validation:

- All packaged PNG bytes match the original audit; dimensions, mask class IDs and CSV pixel counts are checked.
- The validator checks patient/slice identities, exact inventories, roles and metadata bindings. Negative tests cover altered/missing files and unsafe paths.
- A dedicated CPU data-handoff CI job runs the validator and its tests. Existing Kidney model CI jobs remain enabled.
- Read-only existing-model preflight confirms the train/validation datasets contain 2,071/1,156 samples without initializing or fitting a model.

Source data uses KiTS19's CC BY-NC-SA 4.0 terms with attribution and an adaptation notice. These are real processed CT slices; results retain the selected exported-slice segmentation scope. Raw NIfTI scans, other cases, clinical records, weights, environments and generated model outputs are excluded. This MR changes data delivery and documentation, not model training, frozen test results or HTTP integration.

Target: `develop`. The user will review and merge; no automatic merge is configured.
