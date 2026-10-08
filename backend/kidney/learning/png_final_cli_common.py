"""Portable CLI helpers for immutable model freeze and a recorded final test."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_runtime():
    root = Path(__file__).resolve().parents[1]
    source = root / 'kidney-segmentation' if (root / 'kidney-segmentation').is_dir() else root
    sys.path.insert(0, str(source))
    return importlib.import_module('kidney_segmentation.png_final_test')


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_output(requested, protected):
    output = Path(os.path.abspath(str(Path(requested).expanduser())))
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError('Output paths must not contain symlinks.')
    for source in protected:
        source = Path(source).resolve()
        if output == source or source in output.parents:
            raise ValueError('Keep new output outside data, training-run and freeze directories.')
    if output.exists():
        raise ValueError('Output already exists; preserve the existing evidence.')
    return output


def json_bytes(record):
    return (json.dumps(record, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_new_json(path, record):
    with Path(path).open('xb') as handle:
        handle.write(json_bytes(record))


def atomic_json(path, record):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json_bytes(record))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
