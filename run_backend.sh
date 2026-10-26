#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
PROJECT_PYTHON_EXEC="${PROJECT_PYTHON:-.venv/bin/python}"
"$PROJECT_PYTHON_EXEC" scripts/prepare_data.py
exec "$PROJECT_PYTHON_EXEC" backend/server.py --port "${BACKEND_PORT:-8004}" "$@"
