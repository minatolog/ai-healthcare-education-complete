"""Tests for generated artifact checksums and verification."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.diabetes_model.artifact_manifest import (
    MANIFEST_SCHEMA_VERSION,
    ArtifactIntegrityError,
    create_artifact_manifest,
    verify_artifact_manifest,
)


class DiabetesArtifactManifestTests(unittest.TestCase):
    def test_manifest_records_and_verifies_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            directory = Path(temp_directory)
            model_path = directory / "model.joblib"
            report_path = directory / "report.json"
            model_path.write_bytes(b"model-content")
            report_path.write_text('{"result": true}\n', encoding="utf-8")

            manifest_path = create_artifact_manifest(
                {"model": model_path, "report": report_path},
                directory / "artifact_manifest.json",
            )
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            result = verify_artifact_manifest(manifest_path)

            self.assertEqual(
                payload["schema_version"], MANIFEST_SCHEMA_VERSION
            )
            self.assertTrue(payload["educational_only"])
            self.assertEqual(len(payload["artifacts"]), 2)
            self.assertEqual(
                {path.name for path in result.verified_files},
                {"model.joblib", "report.json"},
            )

    def test_changed_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            directory = Path(temp_directory)
            artifact_path = directory / "result.json"
            artifact_path.write_text("original", encoding="utf-8")
            manifest_path = create_artifact_manifest(
                {"result": artifact_path},
                directory / "artifact_manifest.json",
            )

            artifact_path.write_text("modified", encoding="utf-8")
            with self.assertRaisesRegex(
                ArtifactIntegrityError, "mismatch"
            ):
                verify_artifact_manifest(manifest_path)

    def test_missing_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            directory = Path(temp_directory)
            artifact_path = directory / "result.json"
            artifact_path.write_text("result", encoding="utf-8")
            manifest_path = create_artifact_manifest(
                {"result": artifact_path},
                directory / "artifact_manifest.json",
            )
            artifact_path.unlink()

            with self.assertRaisesRegex(ArtifactIntegrityError, "missing"):
                verify_artifact_manifest(manifest_path)

    def test_artifact_outside_manifest_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            directory = Path(temp_directory)
            manifest_directory = directory / "manifest"
            manifest_directory.mkdir()
            artifact_path = directory / "outside.json"
            artifact_path.write_text("result", encoding="utf-8")

            with self.assertRaisesRegex(
                ArtifactIntegrityError, "direct children"
            ):
                create_artifact_manifest(
                    {"outside": artifact_path},
                    manifest_directory / "artifact_manifest.json",
                )


if __name__ == "__main__":
    unittest.main()
