# Backend Contributions

The Kidney feature branches add the Kidney Tumour segmentation source module under [`kidney/`](kidney/README.md). They start from `develop` and keep the existing local learning starter and dataset separate from this repository checkout.

The module includes PNG loading, machine data auditing, deterministic patient-cohort locking, training/reference previews, a frozen ImageNet ResNet-18/U-Net training and validation path, checkpoint handling, frozen final testing, CT-only independent prediction and engineering tests. Local run05 training/validation, frozen test (10 patients/530 slices) and independent prediction (5 patients/676 CT slices) are complete. Read [the results summary](../docs/kidney/RUN05_MODEL_RESULTS.md) for evidence and limitations.

This contribution currently exposes Python module functions and learning commands. It does not implement the team's backend HTTP service or establish compatibility with the frontend. Shared API contracts and the separate Diabetes model adapter are follow-up work. Kidney and Diabetes remain independent tasks, without multimodal fusion.

From the repository root, use:

```sh
cd backend/kidney
```

Read [the module setup and commands](kidney/README.md) and [the feature-branch handoff](../docs/kidney/FEATURE_BRANCH_HANDOFF.md) before integration. The reviewed 40-patient processed CT/mask handoff is now available under [`data/kidney/`](../data/kidney/README.md). Raw NIfTI scans, encoder/model weights, caches, environments and generated outputs remain external.
