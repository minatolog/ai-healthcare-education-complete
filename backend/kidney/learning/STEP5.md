# Step 5: Audit the PNG Dataset and Lock the Patient Cohort

> Feature-branch layout: run commands from `backend/kidney` (use `cd backend/kidney` from the repository root). Replace `/absolute/path/to/kits_project` with your own absolute data path. Prepare the branch environment using [the module README](../README.md); environments, data, caches and generated results are not included in Git. The original local starter is unchanged.

Step 4 showed that one model update works on a development batch. It did not establish formal performance. Before a real training run, this step checks the dataset programmatically and fixes which patients may be used for learning, model selection, final testing and independent prediction.

The project cohort is **40 patients: 20 training + 10 validation + 10 final test**. A separate pool contains **5 prediction-only patients**, matching the prototype's new-patient workflow. These are patient groups, not counts of images. The model task remains three-class image segmentation: background, kidney and tumour. No patient-level Tumour/No Tumour classifier is created.

## Run in the existing VS Code environment

Prepare the module environment using its README. Select `backend/kidney/.venv/bin/python` in VS Code, then run from the repository root:

```sh
cd backend/kidney
source .venv/bin/activate
python learning/05_audit_and_lock_cohort.py --root /absolute/path/to/kits_project --output-dir learning_outputs/step5 --seed 42
```

Keep `kits_project` and its original `splits.json` unchanged. New audit/lock files go into ignored `learning_outputs/step5`, outside the data project. Existing reviewed audit/lock files are not bundled in Git; later commands can reference them by absolute path instead of regenerating evidence. The command performs the full programmatic PNG audit and reports progress. It does not train a model, calculate model-performance metrics or create image previews for human inspection.

## Why split by patient?

One patient can contribute many slices. Nearby slices share anatomy and acquisition details. Putting one slice into training and another from the same patient into test would let the model encounter that patient before the supposedly independent test.

Every slice belonging to a patient therefore follows the same assignment:

| Patient pool | Count | Purpose |
| --- | --- | --- |
| Training | 20 | Learn model parameters and inspect images during development |
| Validation | 10 | Select the model/checkpoint or settings during training |
| Final test | 10 | Evaluate the frozen selected model once at the final evaluation stage |
| Independent prediction | 5 | Demonstrate prediction on patients unused in training or evaluation |

Validation and final test have different roles. Once test performance has influenced changes to the model or its settings, that test is no longer an untouched final check. Prediction-only cases also cannot be recycled from any of the three modelling pools.

`case_00000` was inspected in Steps 2/3 and used in Step 4. It is development-exposed and is forced into training. Its other slices cannot become validation, final-test or new-patient examples.

The command registers development cases found in the saved Step 2/3/4 reports. If you inspected another patient outside those reports, register it before the first lock with `--exposed-case case_XXXXX`; repeat the option for additional patients. Do not omit a case merely because its inspection was manual. A changed exposure list also requires review and must not silently rewrite an existing lock.

## What the machine audit checks

The audit processes the manifest-selected CT/mask PNGs for all **210 indexed cases**. It checks file references and case/slice identities, image/mask pairing, dimensions, grayscale PNG format, decoded pixel values, and content hashes. Masks must contain only class IDs 0/1/2, with kidney/tumour counts and presence flags consistent with the manifest.

These are programmatic quality checks, independent of model predictions. Reading holdout pixels inside a fixed audit is different from displaying those images to developers or adjusting a model after seeing its performance. This script does not present holdout images for human inspection.

The audit does **not** establish anatomical correctness, verify external source aliases or independently confirm full-scan coverage. The PNGs were selected using reference masks in the earlier preprocessing. Even a complete match to the manifest is coverage of those selected PNGs, not proof that every slice of each original CT was retained. Later results must describe segmentation on selected exported slices.

## Duplicate content and deterministic selection

Different case IDs alone do not prove independent patients. The audit identifies groups sharing meaningful CT content. Uniform or otherwise uninformative repeated backgrounds must be distinguished from meaningful duplicates so common blank images are not treated as patient identity evidence.

The cohort uses one representative from a meaningful duplicate group, preventing members of that group from appearing in different pools. This is a conservative content-leakage control; it does not replace manual source-identity verification. If a group contains the development-exposed `case_00000`, that patient is retained as its training representative.

Eligible representative patient IDs are ranked deterministically using role-specific SHA-256 ranking and seed 42. This ranking does not use tumour labels, pixel counts, losses or model scores to choose favourable patients. Reference labels are checked for dataset integrity, not used to rank patients for a better result. The result locks 20/10/10 modelling patients and 5 separate prediction patients. Other patients remain outside the selected cohort. All selected IDs and source/content bindings must be unique and compatible with the duplicate checks.

The seed makes the procedure reproducible; it is not a quality score. Do not try multiple seeds and choose whichever gives the best test result.

## Read the audit and lock files

Open these generated files in VS Code:

| File under `learning_outputs/step5` | Purpose |
| --- | --- |
| `png_data_audit.json` | Programmatic file/pixel checks, content hashes, issues and duplicate groups |
| `project40_splits.json` | Immutable patient assignments and their audit/preprocessing bindings |
| `step5_summary.json` | Completion status, whether a lock was created or verified, and actual patient/slice counts |
| `latest_audit_attempt.json` | A rejected later audit; preserves the original lock and its bound audit |

The lock binds the assignment to the audit evidence and PNG preprocessing configuration. Those bindings identify which manifest, source content and normalization policy were reviewed.

The audit uses schema `png_dataset_audit_v1`; the lock uses `png_cohort_v1` and sampling policy `patient_sha256_rank_v1`. In the lock, `pools` contains the four patient groups, `exposed_patient_ids` records development exposure, and excluded-patient records explain cases outside the selection, including duplicate representatives. Configuration and manifest/audit fingerprints bind this record to its reviewed inputs. Source identifiers such as `kits19:case_00000` track the dataset case; they are not independent proof that external patient aliases have been resolved.

Check the following before continuing:

- The machine audit completed successfully, with no unresolved pairing, dimensions, label or metadata failures.
- The modelling pools contain exactly 20 training, 10 validation and 10 test patients; the prediction pool contains 5 additional patients.
- `case_00000` belongs to training, and patient IDs do not overlap between pools.
- Meaningful duplicate content cannot cross the pool boundaries.
- The lock records its seed and audit/preprocessing bindings rather than only listing patient names.
- The stated scope remains selected-slice segmentation, with manual anatomical/source/full-scan limitations visible.

No accuracy or expected Dice score can be inferred from these checks. They establish reproducible inputs and separation, not model performance.

## Repeating the command preserves the lock

Running the default command again verifies and reuses the existing lock when its audit, source and preprocessing bindings remain unchanged. It does not silently create a new random split or force-overwrite patient assignments.

If source files, the manifest, preprocessing or another bound input changes, verification must reject the existing lock. Keep the old evidence, review the change, and create a deliberately versioned replacement through the agreed process. Do not delete a lock merely to obtain a different split, reuse a seed to conceal changed data, or move patients after looking at final-test results.

The generated `learning_outputs/step5` material is local evidence and stays outside Git by default. Later, the team can review and commit appropriate small metadata copies for reproducibility. PNGs, masks, caches and model weights stay outside GitLab.

## Verified local result on 2026-10-06

The first local run passed machine checks for all 20,686 image/mask pairs and 210 cases, with no issues or informative exact cross-patient CT duplicate groups under the documented policy. It created the seed-42 lock with these counts:

| Pool | Patients | Selected slices |
| --- | ---: | ---: |
| Training | 20 | 2,071 |
| Validation | 10 | 1,156 |
| Final test | 10 | 530 |
| Independent prediction | 5 | 676 |

The default repeat command should report `verified_existing_lock` when the source data and registered development exposures remain unchanged. Source manifest and original split hashes were rechecked unchanged. No model ran in Step 5. Before the next model-development step, review training-only image/mask alignment and provenance, then add the formal PNG training entry that consumes this lock.

## Next: one epoch on the locked training pool

After the machine audit and split checks pass, Step 6 will run one real training epoch using only the locked training patients. The validation stage may help select the model; final test and the five independent prediction cases remain unused until their designated stages.

Broader manual checks of image appearance, anatomical alignment and normalization should use training patients, preserving holdout blindness. Record unresolved source-identity and coverage limitations rather than presenting the machine audit as medical validation. Formal training has not been completed by Step 5.
