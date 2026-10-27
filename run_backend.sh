#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
PROJECT_PYTHON_EXEC="${PROJECT_PYTHON:-.venv/bin/python}"
if ! command -v "$PROJECT_PYTHON_EXEC" >/dev/null 2>&1; then
  printf '%s\n' 'Create the environment: python3 -m venv .venv' 'Install dependencies: .venv/bin/python -m pip install -r requirements.txt'
  exit 1
fi
"$PROJECT_PYTHON_EXEC" scripts/prepare_data.py
exec "$PROJECT_PYTHON_EXEC" backend/server.py --port "${BACKEND_PORT:-8004}" "$@"
