"""Verify local teaching data or restore it from an explicitly provided archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def expected_images(root):
    expected = {}
    for line in (root / 'data/kidney/checksums.sha256').read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        if name.startswith(('data2d/images/', 'data2d/masks/')) and name.endswith('.png'):
            expected['data/kidney/' + name] = checksum
    if len(expected) != 7514:
        raise ValueError('The reviewed checksum manifest must contain exactly 7,514 PNGs.')
    distribution = json.loads((root / 'docs/distribution.json').read_text())
    expected.update(distribution.get('extra_files', {}))
    return expected


def restore(root, archive=None):
    expected = expected_images(root)
    missing = [name for name, digest in expected.items()
               if not (root / name).is_file() or sha256(root / name) != digest]
    if not missing:
        print('Data ready: 3,757 CT / reference-mask pairs; all PNG SHA-256 values match.')
        return
    if archive is None:
        preview = ', '.join(missing[:3])
        raise ValueError(
            f'{len(missing)} data files are missing or changed: {preview}. '
            'Restore the complete project data, or run '
            'python scripts/prepare_data.py --archive /path/to/kidney-data.zip. '
            'No data will be downloaded automatically.'
        )
    distribution = json.loads((root / 'docs/distribution.json').read_text())
    bundle = Path(archive)
    if sha256(bundle) != distribution['sha256']:
        raise ValueError('Local archive SHA-256 does not match the reviewed data bundle.')
    with ZipFile(bundle) as zip_file:
        names = zip_file.namelist()
        if len(names) != len(expected) or set(names) != set(expected):
            raise ValueError('Archive must contain exactly the reviewed CT, mask and metadata paths.')
        for name in names:
            data = zip_file.read(name)
            if hashlib.sha256(data).hexdigest() != expected[name]:
                raise ValueError('Data file checksum mismatch: ' + name)
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print('Restored all 7,514 verified PNGs and the pinned metadata.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Restore missing/changed files from a local reviewed data archive.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        restore(args.root.resolve(), args.archive)
    except (OSError, ValueError, BadZipFile) as error:
        parser.exit(1, f'Data preparation failed: {error}\n')
