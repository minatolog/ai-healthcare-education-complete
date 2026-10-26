#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "${PROJECT_PYTHON:-.venv/bin/python}" backend/server.py --port "${BACKEND_PORT:-8004}" "$@"
