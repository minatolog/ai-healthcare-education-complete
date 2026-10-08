"""Create and verify checksums for generated diabetes model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping

MANIFEST_SCHEMA_VERSION = "1.0"


class ArtifactIntegrityError(ValueError):
    """Raised when an artifact manifest or listed file is invalid."""


@dataclass(frozen=True)
class ArtifactVerificationResult:
    """Names and paths successfully checked against a manifest."""

    manifest_path: Path
    verified_files: tuple[Path, ...]


def create_artifact_manifest(
    artifacts: Mapping[str, str | Path],
    output_path: str | Path,
) -> Path:
    """Write deterministic file metadata and SHA-256 checksums."""

    if not artifacts:
        raise ArtifactIntegrityError("At least one artifact is required.")

    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_directory = manifest_path.parent.resolve()
    entries: list[dict[str, str | int]] = []
    seen_filenames: set[str] = set()

    for logical_name, artifact_path_value in sorted(artifacts.items()):
        if type(logical_name) is not str or not logical_name.strip():
            raise ArtifactIntegrityError(
                "Every artifact must have a non-empty logical name."
            )
        artifact_path = Path(artifact_path_value)
        if not artifact_path.is_file():
            raise ArtifactIntegrityError(
                f"Artifact does not exist or is not a file: {artifact_path}"
            )
        if artifact_path.resolve().parent != output_directory:
            raise ArtifactIntegrityError(
                "Manifest artifacts must be direct children of its directory."
            )
        if artifact_path.name in seen_filenames:
            raise ArtifactIntegrityError(
                f"Duplicate artifact filename: {artifact_path.name}"
            )
        seen_filenames.add(artifact_path.name)
        entries.append(
            {
                "name": logical_name,
                "filename": artifact_path.name,
                "bytes": artifact_path.stat().st_size,
                "sha256": _sha256(artifact_path),
            }
        )

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "educational_only": True,
        "generated_by": "services.diabetes_model.generate_demo_artifacts",
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": _package_version("scikit-learn"),
            "matplotlib": _package_version("matplotlib"),
            "joblib": _package_version("joblib"),
        },
        "artifacts": entries,
        "integrity_note": (
            "SHA-256 detects file changes; this manifest is not a digital signature."
        ),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def verify_artifact_manifest(
    manifest_path: str | Path,
) -> ArtifactVerificationResult:
    """Verify file existence, size, and SHA-256 against a trusted manifest."""

    path = Path(manifest_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Could not read manifest: {path}") from exc

    if not isinstance(payload, dict):
        raise ArtifactIntegrityError("Manifest root must be a JSON object.")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactIntegrityError("Unsupported manifest schema version.")
    if payload.get("educational_only") is not True:
        raise ArtifactIntegrityError("Manifest must be marked educational_only.")
    entries = payload.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ArtifactIntegrityError("Manifest must contain artifact entries.")

    verified_files: list[Path] = []
    seen_names: set[str] = set()
    seen_filenames: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ArtifactIntegrityError("Each artifact entry must be an object.")
        try:
            logical_name = entry["name"]
            filename = entry["filename"]
            expected_bytes = entry["bytes"]
            expected_sha256 = entry["sha256"]
        except KeyError as exc:
            raise ArtifactIntegrityError(
                "Artifact entry is missing a required field."
            ) from exc

        if type(logical_name) is not str or not logical_name:
            raise ArtifactIntegrityError("Artifact name must be a non-empty string.")
        if logical_name in seen_names:
            raise ArtifactIntegrityError(f"Duplicate artifact name: {logical_name}")
        seen_names.add(logical_name)

        if (
            type(filename) is not str
            or not filename
            or Path(filename).name != filename
        ):
            raise ArtifactIntegrityError("Artifact filename must be a safe basename.")
        if filename in seen_filenames:
            raise ArtifactIntegrityError(f"Duplicate artifact filename: {filename}")
        seen_filenames.add(filename)

        artifact_path = path.parent / filename
        if not artifact_path.is_file():
            raise ArtifactIntegrityError(f"Artifact is missing: {filename}")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ArtifactIntegrityError(f"Invalid byte count for {filename}.")
        if artifact_path.stat().st_size != expected_bytes:
            raise ArtifactIntegrityError(f"Artifact size mismatch: {filename}")
        if (
            type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or _sha256(artifact_path) != expected_sha256
        ):
            raise ArtifactIntegrityError(f"Artifact checksum mismatch: {filename}")
        verified_files.append(artifact_path)

    return ArtifactVerificationResult(
        manifest_path=path,
        verified_files=tuple(verified_files),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError as exc:
        raise ArtifactIntegrityError(
            f"Required package is not installed: {distribution_name}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify generated diabetes model artifact checksums."
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    result = verify_artifact_manifest(args.manifest)
    print(f"Verified {len(result.verified_files)} artifacts:")
    for verified_path in result.verified_files:
        print(f"- {verified_path.name}")


if __name__ == "__main__":
    main()
