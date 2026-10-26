# Backend

## Local patient API

Run `python backend/server.py` from the repository root. The service listens at
`http://127.0.0.1:8000` and loads the 40 teaching identities from
[`data/`](data/README.md). It provides separate diabetes and kidney responses,
ordered CT series, and allowlisted CT image serving. No database installation
or credentials are needed. Follow [the English setup guide](../docs/JSON_PATIENT_SETUP.md)
to start both services and validate the migration.

## Model endpoints

[`model_service.py`](model_service.py) is the layer between the HTTP API and the
project models. It imports the real diabetes package from
[`services/diabetes_model`](../services/diabetes_model/README.md), keeps the
fitted estimator in memory for the running session, and reports the kidney model
status honestly.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/model/status` | Which models can run here, plus the committed artifact metadata |
| `POST /api/diabetes/train` | Body `{"patient_ids": ["P001", …]}` — exactly 20 development patients |
| `POST /api/diabetes/predict` | Body `{"patient_code": "P031"}` — one held-out patient |
| `POST /api/kidney/train` | Recorded U-Net epochs, validation Dice and held-out test Dice |
| `POST /api/kidney/predict` | Per-slice tumour segmentation for one patient |
| `GET /api/kidney/patients/P006/overlay/106.png` | CT slice with predicted kidney/tumour mask |

The training response contains the real 20/10/10 split, the measured tree-growth
curve, split feature importance, per-patient evaluation predictions, accuracy and
the untouched held-out result. Prediction uses the session model when one exists
and otherwise falls back to the committed demonstration artifact; only the session
model can return local feature effects.

Kidney training and prediction use
[`kidney_web_model.py`](kidney_web_model.py), a small 3-class U-Net (background,
kidney, tumour) trained on the reviewed PNG handoff in [`data/kidney`](../data/kidney/README.md).
Build the checkpoint once, then the API serves real segmentation:

```bash
python backend/train_kidney_model.py   # ~100s on Apple MPS, writes artifacts/kidney/
```

This GitHub distribution includes the reviewed teaching checkpoint and metrics. If it is missing, `/api/kidney/*` returns `503 model_unavailable` with that
exact command, and the website displays the reason rather than inventing output.
The research pipeline under [`kidney/`](kidney/README.md) is unchanged and still
needs PyTorch plus its own external checkpoint.

```bash
python3 -m unittest discover -s backend -p 'test_*.py' -v
```

## Kidney model module

The Kidney feature branches add the Kidney Tumour segmentation source module under [`kidney/`](kidney/README.md). They start from `develop` and keep the existing local learning starter and dataset separate from this repository checkout.

The module includes PNG loading, machine data auditing, deterministic patient-cohort locking, training/reference previews, a frozen ImageNet ResNet-18/U-Net training and validation path, checkpoint handling, frozen final testing, CT-only independent prediction and engineering tests. Local run05 training/validation, frozen test (10 patients/530 slices) and independent prediction (5 patients/676 CT slices) are complete. Read [the results summary](../docs/kidney/RUN05_MODEL_RESULTS.md) for evidence and limitations.

The model module exposes Python functions and learning commands. The research CLI remains separate. The HTTP service invokes the Diabetes adapter and the teaching SmallUNet, rather than this ResNet18 research model. Kidney and Diabetes remain independent tasks, without multimodal fusion.

From the repository root, use:

```sh
cd backend/kidney
```

Read [the module setup and commands](kidney/README.md) and [the feature-branch handoff](../docs/kidney/FEATURE_BRANCH_HANDOFF.md) before integration. The reviewed 40-patient processed CT/mask handoff is now available under [`data/kidney/`](../data/kidney/README.md). Raw NIfTI scans, encoder/model weights, caches, environments and generated outputs remain external.
