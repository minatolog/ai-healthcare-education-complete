# Backend

## Local patient API

Run `python backend/server.py` from the repository root. The read-only service
listens at `http://127.0.0.1:8000` and loads the 40 teaching identities from
[`data/`](data/README.md). It provides separate diabetes and kidney responses,
ordered CT series, and allowlisted CT image serving. No database installation
or credentials are needed. Follow [the English setup guide](../docs/JSON_PATIENT_SETUP.md)
to start both services and validate the migration.

## Kidney model module

The Kidney feature branches add the Kidney Tumour segmentation source module under [`kidney/`](kidney/README.md). They start from `develop` and keep the existing local learning starter and dataset separate from this repository checkout.

The module includes PNG loading, machine data auditing, deterministic patient-cohort locking, training/reference previews, a frozen ImageNet ResNet-18/U-Net training and validation path, checkpoint handling, frozen final testing, CT-only independent prediction and engineering tests. Local run05 training/validation, frozen test (10 patients/530 slices) and independent prediction (5 patients/676 CT slices) are complete. Read [the results summary](../docs/kidney/RUN05_MODEL_RESULTS.md) for evidence and limitations.

The model module exposes Python functions and learning commands. The patient HTTP service above serves data only; it does not invoke this model or the Diabetes adapter. Model inference integration remains follow-up work. Kidney and Diabetes remain independent tasks, without multimodal fusion.

From the repository root, use:

```sh
cd backend/kidney
```

Read [the module setup and commands](kidney/README.md) and [the feature-branch handoff](../docs/kidney/FEATURE_BRANCH_HANDOFF.md) before integration. The reviewed 40-patient processed CT/mask handoff is now available under [`data/kidney/`](../data/kidney/README.md). Raw NIfTI scans, encoder/model weights, caches, environments and generated outputs remain external.
