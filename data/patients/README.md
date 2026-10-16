# Patient data has moved

The canonical patient JSON files and source mapping now live in
[backend/data](../../backend/data/README.md). The local backend reads that
store directly; the duplicate files and SQL import scripts have been removed.

Follow the [English startup guide](../../docs/JSON_PATIENT_SETUP.md) to run the
API and frontend. Original diabetes CSVs and CT PNGs remain in their source
folders under data/. The migration audit is stored in
[backend/data/migration_report.json](../../backend/data/migration_report.json).
