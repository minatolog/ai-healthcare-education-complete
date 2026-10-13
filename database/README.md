# PostgreSQL Patient Data Import Guide

This guide covers the patient dataset and database setup. The 40 patients are
defined by `case` in `data/kidney/project40_manifest.csv`. Each patient retains
the original `pool`, and every selected CT slice and reference mask is included.
All data fields and values use English. Gender is stored as SQL `NULL`.

The schema follows the proposal's `PATIENT`, `PATIENT_IMAGE`, and `TRAINING_RUN`
tables and the provenance approach used in `D:/2026/healthAi/database`.
It uses lowercase table names and `created_at`, with additional kidney case IDs,
pools, slice indices and source metadata. Diabetes values are stored in the
`clinical_data` and `biomarker` JSONB columns. `training_run` remains empty.

## 1. Review the 40 patients

Open `D:/2026/project/data/patients/patient_mapping.csv`.
For example, `P001` corresponds to `case_00000`, with 322 CT slices and 322 masks.
All slices from one case belong to the same patient. Random names are generated
reproducibly, so rebuilding the data preserves their assignment.

Diabetes values come from `data/synthetic/diabetes_patients.csv` and are unchanged.
The sources are paired in sorted order for educational patient profiles;
they do not represent matched real patients. `age` comes from the synthetic
diabetes record. Gender is not inferred. `label` refers only to the diabetes target.

## 2. Start the existing PostgreSQL instance

PostgreSQL 17 is installed on this computer. The 9900 project documents the
existing instance on port `5433`, using `C:/Program Files/PostgreSQL/17/data`.
Open PowerShell and run:

```powershell
Set-Location 'D:/2026/project'
$pgBin = 'C:/Program Files/PostgreSQL/17/bin'
& "$pgBin/pg_isready.exe" -h localhost -p 5433
```

If the result says `accepting connections`, continue to Step 3. Otherwise, run:

```powershell
& "$pgBin/pg_ctl.exe" start -D 'C:/Program Files/PostgreSQL/17/data' -l "$env:TEMP/healthcare-postgres.log" -w
& "$pgBin/pg_isready.exe" -h localhost -p 5433
```

If startup reports insufficient file permissions, run the startup command in an
administrator PowerShell window. Do not run `initdb` on the existing data directory.
If the instance's port has changed, use its configured port throughout this guide.

## 3. Create a separate database for this project

The following command uses the `zzy` role documented in the 9900 project.
Enter that role's database password when prompted. If the actual role is named
`Zzy`, replace `zzy` in connection and owner arguments with `Zzy`; case must match.

```powershell
& "$pgBin/createdb.exe" -h localhost -p 5433 -U zzy -W -E UTF8 healthcare_education
```

This creates `healthcare_education` and leaves the old project's `aihealth`
database unchanged. If creation succeeds, continue to Step 4. If a database
with this name already exists, inspect its contents first; do not delete it.

### If zzy does not have permission to create a database

If the command reports `permission denied to create database`, create the
database using your PostgreSQL administrator account and assign `zzy` as owner.
The example below uses `postgres` as the administrator role; replace it if your
installation uses another administrator account. Windows administrator privileges
and PostgreSQL administrator privileges are separate.

Run this in PowerShell and enter the **PostgreSQL administrator's password**:

```powershell
$pgBin = 'C:/Program Files/PostgreSQL/17/bin'
& "$pgBin/createdb.exe" -h localhost -p 5433 -U postgres -W -O zzy -E UTF8 healthcare_education
```

`-U postgres` selects the account performing the operation; `-O zzy` sets the
new database's owner to `zzy`. This lets `zzy` own this database without granting
it cluster-wide database creation privileges. See the official
[createdb options](https://www.postgresql.org/docs/17/app-createdb.html).

If creation succeeds, skip the missing-role instructions below and verify the owner.

### Only if the zzy role does not exist

If the administrator command reports `role "zzy" does not exist`, connect as
the PostgreSQL administrator from PowerShell:

```powershell
& "$pgBin/psql.exe" -X -h localhost -p 5433 -U postgres -W -d postgres
```

At the `postgres=#` prompt, check the existing roles:

```text
\du
```

If `Zzy` already exists, use that exact name in `-U` and `-O` arguments instead
of creating a second role. If neither intended role exists, run the following
inside `psql`, one line at a time:

```text
CREATE ROLE zzy LOGIN;
\password zzy
\q
```

The `\password` command prompts twice for a new password for `zzy`. Choose a
password and keep it for the later import commands. The new role can log in;
it does not need superuser privileges. See
[CREATE ROLE](https://www.postgresql.org/docs/17/sql-createrole.html).

After `\q` returns you to PowerShell, retry the administrator's `createdb`
command with `-O zzy` shown above.

### Verify ownership and the zzy connection

Run this in PowerShell and enter **zzy's password**:

```powershell
& "$pgBin/psql.exe" -X -h localhost -p 5433 -U zzy -W -d healthcare_education -c "SELECT current_database(), current_user, pg_get_userbyid(datdba) AS database_owner FROM pg_database WHERE datname = current_database();"
```

Expected values are `healthcare_education`, `zzy`, and `zzy` respectively
(or `Zzy` if that is your actual role). Continue with `-U zzy` for Steps 4-6,
so the tables are also created and populated by the database owner.

## 4. Create the tables once in the new database

```powershell
& "$pgBin/psql.exe" -X -h localhost -p 5433 -U zzy -W -d healthcare_education -v ON_ERROR_STOP=1 -f database/schema.sql
```

A successful run ends with `COMMIT`. The script creates three tables:

| Table | Contents |
| --- | --- |
| `patient` | 40 patients, English names, diabetes values, kidney cases and pools |
| `patient_image` | CT and mask paths, slice indices, descriptions and source metadata |
| `training_run` | Empty table reserved for future training records |

It also creates two views without copying data: `patient_library` provides
patient and slice counts, and `diabetes_model_input` exposes the original
diabetes CSV fields.

## 5. Import the prepared data

The files are ready to import; regeneration is optional. Run:

```powershell
& "$pgBin/psql.exe" -X -h localhost -p 5433 -U zzy -W -d healthcare_education -v ON_ERROR_STOP=1 -f database/seed_patients.sql
```

A successful run ends with `COMMIT`. The import runs in one transaction.
Repeating the same seed does not add duplicate patients or image records.
If an existing record has the same identifier but different contents, the
import fails and rolls back rather than overwriting the record. Inspect any
reported error before retrying; do not clear existing tables.

The database stores image paths. Keep the project's `data/kidney/data2d`
directory in place. Resolve all stored paths from `D:/2026/project`.

## 6. Check the results

```powershell
& "$pgBin/psql.exe" -X -h localhost -p 5433 -U zzy -W -d healthcare_education -v ON_ERROR_STOP=1 -f database/verify.sql
```

Expected results:

| Item | Count |
| --- | ---: |
| Patients | 40 |
| Patients with NULL gender | 40 |
| Train / validation / test patients | 20 / 10 / 10 |
| CT image records | 3,757 |
| Reference mask records | 3,757 |
| Total image records | 7,514 |
| Training records | 0 |

You can also inspect the data in pgAdmin's Query Tool:

```sql
SELECT * FROM patient_library ORDER BY patient_code;

SELECT p.patient_code, p.name, i.img_type, i.slice_index, i.file_path
FROM patient p JOIN patient_image i ON i.patient_id = p.id
WHERE p.kidney_case_id = 'case_00000'
ORDER BY i.slice_index, i.img_type;
```

If you prefer pgAdmin, connect to `localhost:5433` and create
`healthcare_education` with owner `zzy`. Use the administrator connection for
database creation if required, then connect as `zzy` for the import. Open the
database's Query Tool and execute `schema.sql`, `seed_patients.sql`, and
`verify.sql` in that order. This replaces the command-line operations in Steps 3-6.

## Optional data checks and regeneration

The generated files can be imported directly. Use the following commands only
when checking or rebuilding the dataset. They require Python 3.10 or newer,
with no model installation or third-party Python packages:

```powershell
python database/build_patient_seed.py --check
python -m unittest discover -s database -p 'test_patient_data.py' -v
python database/build_patient_seed.py
python database/validate_postgres.py --pg-bin 'C:/Program Files/PostgreSQL/17/bin'
```

The last command creates a temporary PostgreSQL instance under the project's
`.local` directory with a random port and password. It verifies field-by-field
readback, repeat imports, conflict rollback and foreign keys, then stops and
removes the temporary instance. A successful report is written to
`data/patients/validation_report.json`. This does not import the data into your
regular database instance.

After regeneration, rerun validation. The report's SQL file hashes must match
the current files. Official PostgreSQL references:
[JSONB](https://www.postgresql.org/docs/17/datatype-json.html) and
[psql error handling](https://www.postgresql.org/docs/17/app-psql.html).
