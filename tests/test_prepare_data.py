"""Offline verification and explicit archive restoration for project data."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from scripts.prepare_data import restore


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'project'
        self.root.mkdir()
        self.files = {
            'data/kidney/data2d/images/example.png': b'fixture CT bytes',
            'backend/data/patient_images.json': b'[]',
        }
        self.expected = {name: hashlib.sha256(data).hexdigest()
                         for name, data in self.files.items()}
        self.manifest = patch('scripts.prepare_data.expected_images', return_value=self.expected)
        self.manifest.start()
        self.addCleanup(self.manifest.stop)
        self.archive = Path(self.temporary.name) / 'kidney-data.zip'
        self.make_archive(self.files)

    def make_archive(self, files):
        with ZipFile(self.archive, 'w') as bundle:
            for name, data in files.items():
                bundle.writestr(name, data)
        metadata = self.root / 'docs/distribution.json'
        metadata.parent.mkdir(exist_ok=True)
        metadata.write_text(json.dumps({
            'sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest(),
        }))

    def test_missing_data_requires_an_explicit_local_archive(self):
        with self.assertRaisesRegex(ValueError, '--archive') as error:
            restore(self.root)
        self.assertIn('No data will be downloaded automatically', str(error.exception))
        self.assertFalse((self.root / next(iter(self.files))).exists())

    def test_existing_valid_data_needs_no_archive_or_distribution_file(self):
        for name, data in self.files.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (self.root / 'docs/distribution.json').unlink()
        restore(self.root)

    def test_explicit_local_archive_restores_and_repairs_files(self):
        target = self.root / 'backend/data/patient_images.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'changed')
        restore(self.root, self.archive)
        for name, data in self.files.items():
            self.assertEqual((self.root / name).read_bytes(), data)

    def test_corrupt_archive_is_rejected_before_writing_data(self):
        self.archive.write_bytes(b'not the reviewed archive')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            restore(self.root, self.archive)
        self.assertFalse((self.root / 'backend').exists())

    def test_unlisted_archive_paths_are_rejected(self):
        self.make_archive({'../outside.txt': b'not a dataset asset'})
        with self.assertRaisesRegex(ValueError, 'exactly the reviewed'):
            restore(self.root, self.archive)
        self.assertFalse((self.root.parent / 'outside.txt').exists())


if __name__ == '__main__':
    unittest.main()
