"""Restore the 7,514 reviewed PNGs from the project's versioned release archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile


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
    return expected


def restore(root, archive=None):
    expected = expected_images(root)
    missing = [name for name, digest in expected.items()
               if not (root / name).is_file() or sha256(root / name) != digest]
    if not missing:
        print('Data ready: 3,757 CT / reference-mask pairs; all PNG SHA-256 values match.')
        return
    distribution = json.loads((root / 'docs/distribution.json').read_text())
    with tempfile.TemporaryDirectory(prefix='healthcare-data-') as folder:
        bundle = Path(archive) if archive else Path(folder) / 'kidney-data.zip'
        if archive is None:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
            print('Downloading reviewed CT/mask data:', distribution['url'], flush=True)
            with urlopen(distribution['url'], context=context, timeout=120) as response:
                with bundle.open('wb') as stream:
                    shutil.copyfileobj(response, stream)
        if sha256(bundle) != distribution['sha256']:
            raise ValueError('Downloaded archive SHA-256 does not match the pinned release.')
        with ZipFile(bundle) as zip_file:
            names = zip_file.namelist()
            if len(names) != len(expected) or set(names) != set(expected):
                raise ValueError('Archive must contain exactly the reviewed CT and mask paths.')
            for name in names:
                data = zip_file.read(name)
                if hashlib.sha256(data).hexdigest() != expected[name]:
                    raise ValueError('PNG checksum mismatch: ' + name)
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    print('Restored all 7,514 verified PNGs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded data archive.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    restore(args.root.resolve(), args.archive)
