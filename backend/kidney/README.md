# Kidney Tumour Segmentation Module

This is the Kidney source contribution based on `develop`. The default project route is **single-channel 2D PNG segmentation** using a frozen, verified ImageNet ResNet-18 encoder and a trainable U-Net decoder/head. Classes are `0=background`, `1=kidney`, `2=tumour`.

The original local Starter remains the manual learning environment and the `kits_project` dataset stays external. Patient data, pretrained/model weights, environments and generated run assets are excluded; a small reviewed results summary is included under `docs/kidney`. The patient lock and original model/data hashes are preserved; placing source here does not recreate the cohort or run training.

## Layout

```text
backend/kidney/
  kidney_segmentation/       Python model, PNG loader/audit/cohort/training modules
  learning/                 Step-by-step commands, English guides and CLI tests
  tests/                    Synthetic engineering fixtures; no private CTs required
  scripts/                  Read-only metadata audit
  requirements.txt          Supported direct dependency ranges
  requirements-tested.txt   Exact direct dependency snapshot used locally
```

## Prepare a checkout environment

From the repository root:

```sh
cd backend/kidney
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
```

Select this module's `.venv/bin/python` in VS Code. The branch has no committed environment; the existing starter's environment is a separate local environment. The tested requirements are a direct dependency snapshot, not a complete transitive lock or a GPU environment. The strict offline encoder loader requires `segmentation-models-pytorch==0.5.0`.

Pretrained initialization reads the local Hugging Face cache for `smp-hub/resnet18.imagenet`, pinned revision `3f2325ff978283d47aa6a1d6878ca20565622683`. `kidney_segmentation/pretrained.py` verifies the pinned config/weight SHA-256 values and all encoder tensors. It never downloads or silently substitutes a random encoder. A new teammate must prepare that cache separately before real model initialization; installing Python dependencies alone does not supply the weights. The default cache follows `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME` or the standard local cache. Keep it outside Git.

To prepare that exact snapshot once, with network access, run separately in the active module environment:

```sh
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="smp-hub/resnet18.imagenet",
    revision="3f2325ff978283d47aa6a1d6878ca20565622683",
    allow_patterns=["config.json", "model.safetensors"],
)
PY
```

This is explicit cache preparation, not an automatic action by the training entry. If the cache is unavailable or fails its pinned hashes, model initialization stops. The downloaded files are not shipped or committed with this branch.

## Point every command at external data

Replace `/absolute/path/to/kits_project` below with your actual **absolute** path. The expected data is under its `data2d/` directory, with `manifest.csv`, the unchanged original `splits.json`, `images/` and `masks/`. Do not copy patient files into this module. Do not rely on the old `../kits_project` command default: it would refer to the wrong location from this checkout.

The local Step 5 audit checked 20,686 image/mask pairs across 210 cases. The reviewed seed-42 lock contains:

| Pool | Patients | Selected exported slices |
| --- | ---: | ---: |
| Training | 20 | 2,071 |
| Validation | 10 | 1,156 |
| Final test | 10 | 530 |
| Independent prediction | 5 | 676 |

The modelling cohort is 40 patients; the prediction pool is five additional patients. `case_00000`, inspected and updated during development, belongs to training. All slices of one patient stay in the same pool. Duplicate/source-content controls and bound fingerprints protect the reviewed assignments; external patient aliases and full-scan coverage remain unverified.

Existing local audit/lock/review evidence can be referenced by absolute paths. It need not be copied or committed. If preparing evidence locally in this checkout for the first time, these commands write to ignored `learning_outputs/` and preserve an existing cohort lock:

```sh
python learning/05_audit_and_lock_cohort.py --root /absolute/path/to/kits_project --output-dir learning_outputs/step5 --seed 42
python learning/06a_review_training_pairs.py --root /absolute/path/to/kits_project --cohort learning_outputs/step5/project40_splits.json --audit learning_outputs/step5/png_data_audit.json
```

Do not recreate a lock to conceal a changed dataset, inspection exposure or model decision. Review source/configuration changes and use deliberately versioned evidence. The original dataset's 147/32/31 split is preserved and is not the training assignment used here.

## Check the PNG training entry without training

From `backend/kidney`, with the environment active and reviewed evidence available:

```sh
python learning/06b_train_png.py --root /absolute/path/to/kits_project --cohort learning_outputs/step5/project40_splits.json --audit learning_outputs/step5/png_data_audit.json --reference-review learning_outputs/step6a/training_reference_review.json --check-only
```

This preflight checks metadata bindings and pinned offline-cache files. It decodes no CT/mask PNGs, initializes no model, performs no optimizer step and creates no run directory. Source pixels are checked against their audited hashes when a real run consumes them.

For a fresh reproduction experiment after reviewing training references, the first-epoch command is:

```sh
python learning/06b_train_png.py --root /absolute/path/to/kits_project --cohort learning_outputs/step5/project40_splits.json --audit learning_outputs/step5/png_data_audit.json --reference-review learning_outputs/step6a/training_reference_review.json --epochs 1 --batch-size 2 --device cpu --output-dir learning_outputs/step6b/run01
```

The entry uses all 2,071 training slices and all 1,156 validation slices. It does not load final-test or prediction pixels. It starts fresh, does not resume and refuses to replace existing run evidence; use an explicitly chosen `run02` for a new run. Ctrl+C records `interrupted`, not completed training. Local run01–run05 training and validation are complete. Run05 used ten epochs and retained epoch 10 by patient-mean validation tumour Dice. The completed frozen test and independent CT-only prediction use those same weights; preserve them rather than rerunning this reproduction example over existing outputs.

Training uses cross-entropy plus foreground soft-Dice loss. Optional `--tumour-ce-weight` sets relative CE weights [1,1,w], retaining the original unweighted call at the default w=1; foreground Dice is unchanged. Validation reports hard-argmax Dice/IoU/precision/recall/F1 per patient/class, with patient-balanced means and undefined zero-denominator rates as `null`. The default lowest-validation-loss rule selects `best_model.pt`; optional `--selection-metric tumour_dice` selects highest patient-mean validation tumour Dice with lower validation loss then earlier epoch as tie-breaks. A one-epoch run is a baseline execution, not a promise of useful segmentation or clinical performance.

The checkpoint schema is `png_segmentation_v1`, restored by `load_png_checkpoint`. Step 4's `learning_step4_v1` and legacy volume checkpoints are different formats. The preserved `python -m kidney_segmentation` full-volume/2.5D route is a separate reference path, not this module's default PNG command. The dedicated PNG path now has separate frozen-test and CT-only prediction entries below; the full-volume entry remains separate.

## Engineering checks and handoff

Run source/CLI fixture tests from the module directory:

```sh
python -m pytest tests learning -q
```

The original baseline passed 188 local tests and MR !3 server CI before merging to develop. The current source workflow passed 481 local engineering/CLI tests after the review repair; the next MR's GitLab pipeline must be checked after publication. Synthetic tests verify software behavior separately from the completed local real-data runs. See the next MR description for the current verified test count.

Read [the learning roadmap](learning/README.md), [the results summary](../../docs/kidney/RUN05_MODEL_RESULTS.md) and [the integration handoff](../../docs/kidney/FEATURE_BRANCH_HANDOFF.md). No Jira user-facing acceptance criterion is closed by this source contribution alone.

## Completed frozen evaluation and independent prediction

[Step 6C](learning/STEP6C.md) creates reference-guided diagnostic train/validation views; it is not the blind prediction route. [Step 7](learning/STEP7.md) freezes the selected PNG checkpoint and evaluates every locked test row once without model updates. [Step 8](learning/STEP8.md) processes every locked prediction CT once without opening reference-mask PNGs, producing predicted labels, model overlays, fixed identity-selected previews and a complete JSON manifest.

From the team layout, supply absolute external data, cohort, audit, run, freeze and completed-test paths. All `--check-only` entries are metadata/tensor preflights with no model initialization or PNG decoding. Keep output directories new. Existing real evidence belongs to the original Starter; importing this module does not run or repeat it.

Local run05 patient-mean final-test Dice is 0.876164 for kidney and 0.368032 for tumour. Four test patients have zero tumour Dice. CT-only prediction completed five patients/676 images, saving 676 class-ID masks, 676 model overlays and five fixed previews. The frozen parameter/buffer state and completed-test evidence stayed unchanged. No reference accuracy or disease diagnosis is claimed for those prediction outputs.

The shared HTTP service, request/job persistence, public asset registry, frontend wiring and separate Diabetes adapter remain follow-up work. Local prediction paths are not public URLs; an adapter must map approved identifiers and assets into the agreed shared contract.
