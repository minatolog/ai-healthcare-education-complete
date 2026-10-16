# Run the patient data preview

## 1. Check prerequisites

Use Python 3.10 or newer, Node.js 24 (64-bit), and pnpm. The Python patient
service uses only the standard library. PostgreSQL is no longer required.

```powershell
python --version
node -p "process.version + ' ' + process.arch"
pnpm --version
```

The Node command should report `v24... x64`. This machine's older 32-bit Node
installation cannot run the current Vite dependencies. If needed, use the
bundled runtimes shown in step 4.

## 2. Start the backend

Open a PowerShell terminal and leave it running:

```powershell
Set-Location D:\2026\project
python backend/server.py
```

The JSON API listens on `http://127.0.0.1:8000`. Check it with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Expected output: `status: ok`, `storage: json`, `patients: 40`.

## 3. Start the frontend

Open a second PowerShell terminal and leave it running:

```powershell
Set-Location D:\2026\project\frontend
pnpm install --frozen-lockfile
pnpm dev --host 127.0.0.1
```

Open:

- [Diabetes patients](http://127.0.0.1:8443/diabetes/patients)
- [Kidney patients](http://127.0.0.1:8443/kidney/patients)

Both pages show 40 stored identities. Click the small data button on any card,
including validation/test cards. Diabetes details show only tabular data,
including original units and the reference target. Kidney details show the
case's actual CT images. Drag the slice slider left/right, use the arrow keys,
click Previous/Next, or use Play/Pause. The slider enumerates manifest rows;
the displayed source z can have gaps. Gender is unspecified and not displayed.

The frontend proxies `/api` to port 8000. Start the backend first; if the page
shows a data error, start it and click Retry. Stop either terminal with Ctrl+C.
Restart the backend after rebuilding or editing JSON. An occupied port should
be resolved by stopping the previous copy of this service. Avoid stopping
unrelated processes.

## 4. Bundled runtime commands for this computer

If the normal PATH commands use the old runtime, start the backend with:

```powershell
Set-Location D:\2026\project
& 'C:\Users\Zzy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' backend/server.py
```

In a second terminal, use:

```powershell
Set-Location D:\2026\project\frontend
$runtimeRoot = 'C:\Users\Zzy\.cache\codex-runtimes\codex-primary-runtime\dependencies'
$env:Path = "$runtimeRoot\node\bin;$env:Path"
& "$runtimeRoot\bin\fallback\pnpm.cmd" install --frozen-lockfile
& "$runtimeRoot\node\bin\node.exe" node_modules/vite/bin/vite.js --host 127.0.0.1
```

## 5. Validate data and build

```powershell
Set-Location D:\2026\project
python backend/build_patient_data.py --check
python -m unittest discover -s backend -p "test_patient*.py"
Set-Location frontend
pnpm test:patients
pnpm typecheck
pnpm build
```

Backend tests compare all 40 records and 7,514 image records to the CSV sources,
verify each patient's ordered CT series, check boundary images, and reject
mask, unknown and traversal paths. Frontend tests verify all diabetes table
values/units, modality isolation, and routes for IDs 1–40.

The patient data integration is live. Training, evaluation and prediction
screens still use the existing workflow simulation and are labeled as demos;
they do not run the ML models or write training results. Generated feature
contributions and CT attention overlays have been removed because no model
explanations are available for these stored records.

## JSON files and API

See [the backend store](../backend/data/README.md) for the full data contract,
provenance, fixed CT splits and migration audit. Patient edits are file-based;
no account management or database administrator account is needed.

| Endpoint | Response |
| --- | --- |
| `GET /api/health` | Service status and patient count |
| `GET /api/diabetes/patients` | 40 identities and diabetes tabular values |
| `GET /api/kidney/patients` | 40 identities, CT case IDs, pools and slice counts |
| `GET /api/kidney/patients/P001/ct` | Ordered CT slice URLs for one identity |
| `GET /api/ct/P001/4.png` | A manifest-selected CT image |

The local API binds only to loopback. It serves allowlisted CT PNGs, not
arbitrary files, reference masks, source JSON files or model weights.
