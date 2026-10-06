Title: feat(kidney): add audited 2D segmentation baseline

Source branch: `feat/yining-kidney-segmentation`

Target branch: `develop`

Primary story: H18JDONUT-7

Supporting stories: H18JDONUT-5, H18JDONUT-20, H18JDONUT-15, H18JDONUT-21

This MR adds the Kidney Tumour model module under `backend/kidney`, giving the team a reproducible source baseline for single-slice CT segmentation. A verified ImageNet ResNet-18 encoder remains frozen while a U-Net decoder/head learns background, kidney and tumour labels.

The module includes explicit-patient PNG loading, file/pixel/CSV audits, an immutable 20/10/10 patient lock with five separate prediction cases, training-only reference previews, train/validation telemetry, patient-balanced overlap metrics and a compatible PNG checkpoint format. The read-only metadata checker rejects duplicate CSV headers and malformed records with structured diagnostics. English learning guides, dependencies, fixture tests and CPU CI jobs accompany the source.

Validation:

- 188 local engineering tests passed, including 16 metadata-audit tests.
- All 13 core model-package Python files retained their source hashes during the directory migration.
- The migrated entry passed metadata/offline-cache preflight against the existing local audit and cohort, without model initialization, CT/mask decoding or optimizer updates.
- CI YAML parsed locally; GitLab server CI must be checked on this branch after publication.

Scope and remaining work:

- This is a source/module contribution. Real locked-cohort training/validation results, final PNG testing, independent prediction assets and the shared HTTP/Diabetes/frontend adapters remain follow-up work.
- Existing PNGs were selected using reference labels during export. Results must describe selected exported slices, rather than blind complete-volume performance or patient-level disease diagnosis.
- CTs, masks, environments, pretrained/model weights and generated learning outputs remain local and are excluded from this MR.
- No Jira story or user-facing acceptance criterion is closed by this source merge alone.
