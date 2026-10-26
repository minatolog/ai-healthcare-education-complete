# Local setup (02_main)

Follow the Chinese [root README](../README.md) for complete download and setup instructions.

The website is React/TypeScript/Vite with a Python JSON/model API. Install root `requirements.txt`, run `python scripts/prepare_data.py`, then run `npx --yes pnpm@10 install --frozen-lockfile` inside `frontend`.

From the project root, start `./run_backend.sh` (port 8004) and `./run_frontend.sh` (port 8444). If occupied, use `./run_frontend.sh --port 8445`. Vite proxies `/api` to 8004 by default; `API_BASE_URL` overrides it.

The included teaching U-Net checkpoint is distinct from the research ResNet18/U-Net under `backend/kidney`. Its training/evaluation views replay saved evidence; per-patient CT inference runs the checkpoint. The GitHub source includes its weights; the reviewed PNGs are distributed in the versioned release and restored by the data preparation script.
