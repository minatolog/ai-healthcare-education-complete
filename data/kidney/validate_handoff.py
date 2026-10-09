#!/usr/bin/env python3
"""Verify the immutable 40-patient Kidney data handoff without running a model.

The default checks file bytes, processed CT/mask pixels and reference counts.
``--metadata-only`` reads text metadata and file sizes, never PNG contents.
Checksums detect changes relative to this published snapshot; they are not a
signature or independent proof of the upstream dataset's identity.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


PATIENT_RE = re.compile(r"case_[0-9]{5}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
ROLES = ("train", "validation", "test")
EXPECTED_POOLS = {
    "train": {"patients": 20, "slice_pairs": 2071},
    "validation": {"patients": 10, "slice_pairs": 1156},
    "test": {"patients": 10, "slice_pairs": 530},
}
EXPECTED_COUNTS = {
    "patients": 40, "slice_pairs": 3757, "ct_pngs": 3757,
    "mask_pngs": 3757, "png_bytes": 87913810,
}
REQUIRED_MANIFEST_FIELDS = {
    "case", "z", "image", "mask", "has_kidney", "has_tumor",
    "kidney_px", "tumor_px", "n_slices", "sp_z", "sp_y", "sp_x",
}
PATIENT_FIELDS = {
    "patient_id", "source_patient_id", "pool", "selected_slice_count",
    "content_fingerprint_sha256", "demo_prediction_eligible",
}


class HandoffError(ValueError):
    """A data handoff failed a specific integrity or consistency check."""


def require(condition, message):
    if not condition:
        raise HandoffError(message)


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def fingerprint(payload, field):
    require(isinstance(payload, dict), "Fingerprint input must be a JSON object.")
    content = {key: value for key, value in payload.items() if key != field}
    return sha256_bytes(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8"))


def safe_file(root, relative):
    """Require a canonical relative path to a contained, regular, real file."""
    require(isinstance(relative, str) and relative and "\\" not in relative
            and "\x00" not in relative and "\n" not in relative and "\r" not in relative,
            "Invalid package path: %r." % relative)
    part_path = PurePosixPath(relative)
    require(not part_path.is_absolute() and part_path.as_posix() == relative
            and all(part not in (".", "..") for part in part_path.parts),
            "Package paths must be canonical and relative: %r." % relative)
    cursor = root
    for part in part_path.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "Symlinks are not allowed: %s." % relative)
    require(cursor.is_file() and stat.S_ISREG(cursor.stat().st_mode),
            "Missing regular package file: %s." % relative)
    require(root in cursor.resolve().parents,
            "Package path leaves its root: %s." % relative)
    return cursor


def inventory(root):
    files = set()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        for name in directories + filenames:
            path = Path(directory) / name
            require(not path.is_symlink(), "Symlinks are not allowed: %s." % path)
        directories[:] = [name for name in directories if name != "__pycache__"]
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                continue
            relative = (Path(directory) / name).relative_to(root).as_posix()
            safe_file(root, relative)
            files.add(relative)
    return files


def read_checksums(root):
    index = {}
    text = safe_file(root, "checksums.sha256").read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        require(match is not None, "Malformed checksum at line %d." % number)
        digest, relative = match.groups()
        safe_file(root, relative)
        require(relative != "checksums.sha256" and relative not in index,
                "Duplicate or self-referential checksum path: %s." % relative)
        index[relative] = digest
    actual = inventory(root) - {"checksums.sha256"}
    require(set(index) == actual,
            "Checksum coverage differs from package files; missing=%s, extra=%s."
            % (sorted(actual - set(index))[:5], sorted(set(index) - actual)[:5]))
    return index


def verify_checksums(root, checksums, *, metadata_only):
    checked = 0
    for relative, expected in checksums.items():
        if metadata_only and relative.endswith(".png"):
            continue
        observed = sha256_bytes(safe_file(root, relative).read_bytes())
        require(observed == expected, "Checksum mismatch: %s." % relative)
        checked += 1
    return checked


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: %s." % key)
        result[key] = value
    return result


def read_json(root, relative):
    def reject_constant(value):
        raise HandoffError("Non-finite JSON value: %s." % value)
    return json.loads(safe_file(root, relative).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_json, parse_constant=reject_constant)


def read_csv(root, relative):
    text = safe_file(root, relative).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    headers = reader.fieldnames or []
    require(headers and all(headers) and len(set(headers)) == len(headers),
            "CSV headers must be nonempty and unique: %s." % relative)
    rows = list(reader)
    require(rows and all(None not in row and all(value is not None for value in row.values())
                         for row in rows), "Malformed or empty CSV: %s." % relative)
    return headers, rows


def nonnegative_integer(row, field):
    value = row.get(field)
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None,
            "Manifest %s must be a non-negative integer." % field)
    return int(value)


def manifest_index(headers, rows):
    require(REQUIRED_MANIFEST_FIELDS.issubset(headers), "Source manifest columns are incomplete.")
    index, scan_sizes = {}, {}
    for row in rows:
        patient = row["case"]
        require(PATIENT_RE.fullmatch(patient) is not None, "Invalid manifest patient: %s." % patient)
        z, size = nonnegative_integer(row, "z"), nonnegative_integer(row, "n_slices")
        require(size > 0 and z < size, "Invalid source slice index for %s." % patient)
        require(patient not in scan_sizes or scan_sizes[patient] == size,
                "Original scan size changes within %s." % patient)
        scan_sizes[patient] = size
        identity = (patient, z)
        require(identity not in index, "Duplicate patient/slice identity: %s:z%d." % identity)
        for kind, folder in (("image", "images"), ("mask", "masks")):
            require(row[kind] == "%s/%s_z%03d.png" % (folder, patient, z),
                    "Unexpected %s path for %s:z%d." % (kind, patient, z))
        kidney = nonnegative_integer(row, "kidney_px")
        tumour = nonnegative_integer(row, "tumor_px")
        require(kidney + tumour <= 256 * 256, "Foreground pixel counts exceed image size.")
        require(nonnegative_integer(row, "has_kidney") == int(kidney > 0)
                and nonnegative_integer(row, "has_tumor") == int(tumour > 0),
                "Reference presence flags disagree with pixel counts.")
        for field in ("sp_z", "sp_y", "sp_x"):
            try:
                spacing = float(row[field])
            except (TypeError, ValueError):
                raise HandoffError("Invalid original voxel spacing: %s." % field) from None
            require(math.isfinite(spacing) and spacing > 0,
                    "Original voxel spacing must be positive and finite.")
        index[identity] = row
    return index


def verify_projection(source_headers, source_index, projected_headers, projected_rows, patients):
    require(projected_headers == source_headers + ["pool"],
            "Project manifest must preserve source columns and append only pool.")
    projected = manifest_index(source_headers, projected_rows)
    require(set(projected) == {identity for identity in source_index if identity[0] in patients},
            "Project manifest does not contain exactly every selected source slice.")
    for identity, row in projected.items():
        require({field: row[field] for field in source_headers} == source_index[identity],
                "Project/source metadata mismatch: %s:z%d." % identity)
        require(row["pool"] == patients[identity[0]]["pool"],
                "Patient/slice pool mismatch: %s:z%d." % identity)
    return projected


def verify_png(root, relative, facts, row):
    from PIL import Image
    content = safe_file(root, relative).read_bytes()
    require(sha256_bytes(content) == facts["byte_sha256"],
            "PNG differs from the preserved source audit: %s." % relative)
    with Image.open(io.BytesIO(content)) as image:
        require(image.format == "PNG" and image.mode == "L" and image.size == (256, 256)
                and getattr(image, "n_frames", 1) == 1,
                "Expected a single grayscale L-mode 256x256 PNG: %s." % relative)
        image.load()
        require(sha256_bytes(image.tobytes()) == facts["decoded_pixel_sha256"],
                "Decoded pixels differ from the source audit: %s." % relative)
        if facts["kind"] == "mask":
            histogram = image.histogram()
            require(not any(histogram[3:]), "Unknown mask class ID: %s." % relative)
            counts = {str(label): histogram[label] for label in (0, 1, 2)}
            require(counts == facts["label_counts"], "Mask counts differ from the source audit.")
            require(counts["1"] == int(row["kidney_px"]) and counts["2"] == int(row["tumor_px"])
                    and int(counts["1"] > 0) == int(row["has_kidney"])
                    and int(counts["2"] > 0) == int(row["has_tumor"]),
                    "Mask counts differ from manifest metadata: %s." % relative)


def _verify_source_binding(root, info, audit, cohort):
    require(audit.get("schema_version") == "png_dataset_audit_v1"
            and audit.get("status") == "passed" and not audit.get("issues")
            and not audit.get("global_issues"), "A passed original PNG audit is required.")
    require(cohort.get("schema_version") == "png_cohort_v1", "Unknown original cohort schema.")
    for document, field in ((audit, "audit_fingerprint_sha256"),
                            (cohort, "cohort_fingerprint_sha256")):
        require(HASH_RE.fullmatch(str(document.get(field))) is not None
                and fingerprint(document, field) == document[field],
                "The preserved %s is stale or has changed." % field)
    bindings = info.get("source_metadata", {})
    for field, relative in (
            ("manifest_sha256", "data2d/manifest.csv"),
            ("original_split_sha256", "data2d/splits.json"),
            ("cohort_file_sha256", "provenance/project40_splits.json"),
            ("audit_file_sha256", "provenance/png_data_audit.json")):
        require(bindings.get(field) == sha256_bytes(safe_file(root, relative).read_bytes()),
                "Source metadata binding mismatch: %s." % field)
    require(bindings.get("manifest_sha256") == audit.get("manifest_sha256")
            == cohort.get("manifest_sha256"), "Source manifest differs from the original audit/lock.")
    require(bindings.get("original_split_sha256") == audit.get("original_split_sha256"),
            "Original split differs from the source audit.")
    require(bindings.get("cohort_fingerprint_sha256") == cohort["cohort_fingerprint_sha256"]
            and bindings.get("audit_fingerprint_sha256") == audit["audit_fingerprint_sha256"]
            == cohort.get("audit_fingerprint_sha256"), "Source audit/cohort fingerprints disagree.")
    require(info.get("preprocessing") == {"input_mode": "single_2d_slice",
                                          "config": cohort.get("preprocessing", {}).get("config")},
            "Handoff preprocessing differs from the immutable cohort.")
    require(info["preprocessing"]["config"] == {
        "input_channels": 1, "size": 256, "mean": 0.456, "std": 0.224,
        "version": "png-single-slice-gray8-normalized-v1"}, "Unknown PNG preprocessing configuration.")
    # These are archived source snapshots, not runnable instructions to regenerate
    # or overwrite this already locked data package.
    safe_file(root, "provenance/source_processing/preprocess.py")
    safe_file(root, "provenance/source_processing/config.py")


def _patient_index(root, info, audit, cohort):
    headers, rows = read_csv(root, "patient_splits.csv")
    require(set(headers) == PATIENT_FIELDS, "Patient index columns are incomplete or unexpected.")
    patients, sources, fingerprints = {}, {}, {}
    expected = {record["patient_id"]: (role, record)
                for role in ROLES for record in cohort.get("pools", {}).get(role, [])}
    require(len(expected) == 40 and len(rows) == 40, "The handoff must contain exactly 40 formal patients.")
    audited = {record["patient_id"]: record for record in audit.get("patients", [])}
    for row in rows:
        patient = row["patient_id"]
        require(patient not in patients and patient in expected,
                "Duplicate or unlocked formal patient: %s." % patient)
        role, locked = expected[patient]
        require(row["pool"] == role and role in ROLES, "Patient moved from its locked pool: %s." % patient)
        for field in ("source_patient_id", "content_fingerprint_sha256"):
            require(row[field] == locked[field] == audited[patient][field],
                    "Patient source binding changed: %s / %s." % (patient, field))
        require(HASH_RE.fullmatch(row["content_fingerprint_sha256"]) is not None,
                "Invalid patient fingerprint.")
        count = nonnegative_integer(row, "selected_slice_count")
        require(count > 0 and count == locked["selected_slice_count"]
                == audited[patient]["selected_slice_count"], "Patient selected-slice count changed.")
        expected_eligibility = {"true", "1"} if role == "test" else {"false", "0"}
        require(row["demo_prediction_eligible"].lower() in expected_eligibility,
                "Only locked test patients are eligible for current demo prediction.")
        for mapping, field in ((sources, "source_patient_id"),
                               (fingerprints, "content_fingerprint_sha256")):
            require(row[field] not in mapping, "Repeated formal source/content identity: %s." % patient)
            mapping[row[field]] = role
        patients[patient] = row
    require(set(patients) == set(expected), "Formal patient set differs from the immutable cohort.")
    for role in ROLES:
        records = [row for row in rows if row["pool"] == role]
        require(len(records) == EXPECTED_POOLS[role]["patients"], "Locked role patient count changed.")
    exposed = cohort.get("exposed_patient_ids", [])
    require(all(patient in patients and patients[patient]["pool"] == "train" for patient in exposed),
            "Development-exposed patients must remain training patients.")
    legacy = [record["patient_id"] for record in cohort["pools"].get("prediction", [])]
    require(len(legacy) == 5 and len(set(legacy)) == 5 and not set(legacy) & set(patients)
            and info.get("legacy_prediction_patient_ids") == legacy,
            "Legacy prediction provenance changed or entered the formal cohort.")
    for component in audit.get("duplicate_patient_components", []):
        present_roles = {patients[patient]["pool"] for patient in component if patient in patients}
        require(len(present_roles) <= 1, "A duplicate-content component crosses formal pools.")
    return patients


def validate_handoff(root, *, metadata_only=False):
    root = Path(root).expanduser().absolute()
    require(root.is_dir() and not root.is_symlink(), "A real data handoff directory is required.")
    root = root.resolve()
    checksums = read_checksums(root)
    checked_hashes = verify_checksums(root, checksums, metadata_only=metadata_only)
    info = read_json(root, "dataset_info.json")
    require(info.get("schema_version") == "kidney_data_handoff_v1", "Unknown data handoff schema.")
    require(info.get("counts") == EXPECTED_COUNTS and info.get("pools") == EXPECTED_POOLS,
            "The published 40-patient handoff counts changed.")
    cohort = read_json(root, "provenance/project40_splits.json")
    audit = read_json(root, "provenance/png_data_audit.json")
    _verify_source_binding(root, info, audit, cohort)
    patients = _patient_index(root, info, audit, cohort)
    source_headers, source_rows = read_csv(root, "data2d/manifest.csv")
    source = manifest_index(source_headers, source_rows)
    source_patients = {identity[0] for identity in source}
    require(len(source) == 20686 and len(source_patients) == 210,
            "The source manifest must remain the original 210-case / 20,686-slice index.")
    old_split = read_json(root, "data2d/splits.json")
    require(set(old_split) == {"train", "val", "test"}, "Unexpected old split schema.")
    old_ids = [patient for records in old_split.values() for patient in records]
    require(len(set(old_ids)) == len(old_ids) and set(old_ids) == source_patients,
            "The original split must partition the original source patients.")
    require({key: len(value) for key, value in old_split.items()}
            == {"train": 147, "val": 32, "test": 31}, "The archived old split counts changed.")
    stats = read_json(root, "data2d/dataset_stats.json")
    require(stats.get("n_slices") == len(source) and stats.get("n_cases") == len(source_patients),
            "Original dataset statistics disagree with the source manifest.")
    projected_headers, projected_rows = read_csv(root, "project40_manifest.csv")
    projected = verify_projection(source_headers, source, projected_headers, projected_rows, patients)
    require(len(projected) == EXPECTED_COUNTS["slice_pairs"], "Formal slice-pair count changed.")
    for role in ROLES:
        count = sum(row["pool"] == role for row in projected.values())
        require(count == EXPECTED_POOLS[role]["slice_pairs"], "Selected slice count changed for %s." % role)
    for patient, record in patients.items():
        count = sum(identity[0] == patient for identity in projected)
        require(count == int(record["selected_slice_count"]), "Patient/project slice count differs.")
    expected_assets = {"data2d/" + row[kind] for row in projected.values() for kind in ("image", "mask")}
    asset_prefixes = ("data2d/images/", "data2d/masks/")
    actual_assets = {path for path in inventory(root) if path.startswith(asset_prefixes)}
    require(actual_assets == expected_assets, "PNG inventory differs from exactly the 40 formal patients.")
    png_bytes = 0
    checked_pngs = 0
    for identity, row in projected.items():
        for kind in ("image", "mask"):
            relative = "data2d/" + row[kind]
            path = safe_file(root, relative)
            facts = audit.get("assets", {}).get(row[kind], {})
            require(facts.get("patient_id") == identity[0] and facts.get("slice_index") == identity[1]
                    and facts.get("kind") == kind and facts.get("mode") == "L"
                    and facts.get("width") == 256 and facts.get("height") == 256
                    and checksums.get(relative) == facts.get("byte_sha256"),
                    "Selected asset identity/checksum differs from source audit: %s." % relative)
            png_bytes += path.stat().st_size
            if not metadata_only:
                verify_png(root, relative, facts, row)
                checked_pngs += 1
    require(png_bytes == EXPECTED_COUNTS["png_bytes"],
            "PNG byte count changed; ensure actual images are present, including Git LFS downloads if used.")
    return {"mode": "metadata_only" if metadata_only else "full",
            "patients": len(patients), "slice_pairs": len(projected),
            "files_hash_checked": checked_hashes, "pngs_decoded": checked_pngs,
            "png_bytes": png_bytes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Data handoff directory (default: this script's directory).")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Check metadata, identity, inventory and file sizes without reading PNG contents.")
    args = parser.parse_args(argv)
    try:
        report = validate_handoff(args.root, metadata_only=args.metadata_only)
    except (HandoffError, OSError, KeyError, TypeError, ValueError, ImportError) as error:
        print("[DATA HANDOFF VALIDATION FAILED] %s" % error, file=sys.stderr)
        return 1
    print("[DATA HANDOFF VALIDATION PASSED]")
    print("Mode: %(mode)s; patients: %(patients)d; slice pairs: %(slice_pairs)d" % report)
    print("Files hash-checked: %(files_hash_checked)d; PNGs decoded: %(pngs_decoded)d; PNG bytes: %(png_bytes)d" % report)
    print("Formal pools: 20 train / 10 validation / 10 test. Demo prediction uses the same 10 test patients.")
    if args.metadata_only:
        print("PNG bytes and pixels were not read. Run without --metadata-only to validate image contents.")
    print("No model, training, evaluation or source-file modification occurred.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
