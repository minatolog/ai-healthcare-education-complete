# Backend Contributions

The `feat/yining-kidney-segmentation` feature branch adds the Kidney Tumour segmentation source module under [`kidney/`](kidney/README.md). It starts from `develop` and keeps the existing local learning starter and dataset separate from this repository checkout.

The module includes PNG loading, machine data auditing, deterministic patient-cohort locking, training reference previews, a frozen ImageNet ResNet-18/U-Net training and validation path, checkpoint handling, and engineering tests. Its real locked-cohort training and validation results are **pending**.

This contribution currently exposes Python module functions and learning commands. It does not implement the team's backend HTTP service or establish compatibility with the frontend. Shared API contracts and the separate Diabetes model adapter are follow-up work. Kidney and Diabetes remain independent tasks, without multimodal fusion.

From the repository root, use:

```sh
cd backend/kidney
```

Read [the module setup and commands](kidney/README.md) and [the feature-branch handoff](../docs/kidney/FEATURE_BRANCH_HANDOFF.md) before integration. Patient images, masks, encoder/model weights, caches, environments and generated outputs stay outside Git.
