"""Deterministic machine QA for selected KiTS PNGs; no Torch or model imports.

All images/masks are decoded sequentially. The audit checks file contents and
reference-label metadata, not anatomical correctness, full scans, or diagnosis.
It reports duplicate groups but does not choose patients or assign new splits.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

SCHEMA_VERSION = "png_dataset_audit_v1"
REQUIRED_COLUMNS = {"case", "z", "image", "mask", "n_slices", "kidney_px",
                    "tumor_px", "has_kidney", "has_tumor"}
CASE_PATTERN = re.compile(r"case_[0-9]{5}\Z")
MINIMUM_DUPLICATE_CENTRAL_UNIQUE_VALUES = 16
MINIMUM_DUPLICATE_CENTRAL_STD = 8.0
MINIMUM_DUPLICATE_CENTRAL_BODY_PIXELS = 1000


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def audit_fingerprint(audit):
    """Hash every report field except the fingerprint itself; no extra exclusions."""
    return _canonical_sha256({key: value for key, value in audit.items()
                              if key != "audit_fingerprint_sha256"})


def _bytes_sha256(content):
    return hashlib.sha256(content).hexdigest()


def _integer(row, key):
    value = int(row[key])
    if value < 0:
        raise ValueError("negative metadata")
    return value


def _resolve_asset(root, relative, folder, case, z):
    expected = "%s_z%03d.png" % (case, z)
    if not isinstance(relative, str) or Path(relative).parts != (folder, expected):
        raise ValueError("unexpected asset path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("asset outside root")
    if not path.is_file():
        raise FileNotFoundError("missing asset")
    return path


def _decode_png(path):
    content = path.read_bytes()
    if (len(content) < 33 or content[:8] != b"\x89PNG\r\n\x1a\n"
            or content[12:16] != b"IHDR" or content[24:26] != b"\x08\x00"):
        raise ValueError("expected grayscale 8-bit PNG header")
    if int.from_bytes(content[16:20], "big") != 256 or int.from_bytes(content[20:24], "big") != 256:
        raise ValueError("expected 256x256 PNG header dimensions")
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (256, 256):
            raise ValueError("expected grayscale L-mode 256x256 PNG")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("animated PNG unsupported")
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        pixels = np.array(image, dtype=np.uint8)
    return pixels, {
        "byte_sha256": _bytes_sha256(content),
        "decoded_pixel_sha256": _bytes_sha256(pixels.tobytes(order="C")),
        "mode": "L", "width": 256, "height": 256,
        "minimum_pixel": int(pixels.min()), "maximum_pixel": int(pixels.max()),
    }


def _duplicate_components(groups):
    parents = {}

    def find(case):
        parents.setdefault(case, case)
        if parents[case] != case:
            parents[case] = find(parents[case])
        return parents[case]

    for group in groups:
        ordered = sorted(set(group))
        if len(ordered) < 2:
            continue
        head = find(ordered[0])
        for case in ordered[1:]:
            parents[find(case)] = head
    components = defaultdict(list)
    for case in sorted(parents):
        components[find(case)].append(case)
    return sorted(sorted(group) for group in components.values() if len(group) > 1)


def audit_png_dataset(root, progress=None):
    """Audit a kits_project root or its data2d directory and return stable JSON data.

    ``progress`` optionally receives a dict with completed_slice_count and
    total_slice_count every 1,000 rows and on completion. Progress is not stored
    in the hashed report. Only one image/mask pair's arrays are held at a time.
    """
    root = Path(root).expanduser().resolve()
    data = root if (root / "manifest.csv").is_file() else root / "data2d"
    data = data.resolve()
    error_counts, error_samples = Counter(), defaultdict(list)
    bad_cases = set()

    def issue(code, message, case=None, row_number=None, asset=None):
        error_counts[(code, message)] += 1
        if case:
            bad_cases.add(case)
        sample = {}
        if case:
            sample["patient_id"] = case
        if row_number is not None:
            sample["row_number"] = row_number
        if asset is not None:
            sample["asset"] = asset
        if sample and len(error_samples[(code, message)]) < 5:
            error_samples[(code, message)].append(sample)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "scope": "selected_manifest_slices",
        "machine_file_pixel_checks_only": True,
        "visual_anatomical_alignment_verified": False,
        "full_scan_coverage": False,
        "source_patient_aliases_verified": False,
        "duplicate_detection_uses_reference_presence": False,
        "dataset_namespace": "kits19",
        "manifest_sha256": None,
        "original_split_sha256": None,
        "patients": [], "assets": {}, "issues": [], "warnings": [],
        "duplicate_patient_components": [],
        "preprocessing": {
            "input_mode": "single_2d_slice", "input_channels": 1,
            "size": 256, "mean": 0.456, "std": 0.224,
            "version": "png-single-slice-gray8-normalized-v1",
            "units": "already_processed_uint8_not_HU", "mask_labels": {"0": "background", "1": "kidney", "2": "tumour"},
        },
        "duplicate_detection_policy": {
            "image_fingerprint": "SHA-256 of decoded row-major uint8 CT pixels",
            "patient_fingerprint": "Canonical ordered selected CT pixel hashes, without patient IDs or z values",
            "central_ct_crop": {"row_start": 64, "row_stop": 192, "column_start": 64, "column_stop": 192},
            "minimum_central_unique_pixel_values": MINIMUM_DUPLICATE_CENTRAL_UNIQUE_VALUES,
            "minimum_central_pixel_std": MINIMUM_DUPLICATE_CENTRAL_STD,
            "minimum_central_pixels_above_32": MINIMUM_DUPLICATE_CENTRAL_BODY_PIXELS,
            "uses_reference_foreground_presence": False,
            "duplicate_detection_uses_reference_presence": False,
            "criterion_uses_only_ct_pixels": True,
            "constant_background_duplicates_ignored": True,
            "whole_series_requires_at_least_one_informative_ct_slice": True,
            "duplicates_are_review_groups_not_automatic_patient_selection": True,
        },
        "limitations": [
            "Machine dimensions and metadata agreement do not establish anatomical image/mask alignment.",
            "The source_patient_id is a kits19 case namespace; independent aliases outside this dataset are not verified.",
            "The source preprocessing selected foreground slices using reference masks; complete original scans are not established.",
            "Hash duplicates detect exact selected pixel content, not every transformed duplicate or repeated source patient.",
            "Recorded hashes describe the bytes read during this audit; selected assets must be rechecked before later training.",
            "This audit does not select a formal cohort, inspect model performance, train, or display held-out reference images.",
        ],
    }
    audit["preprocessing_sha256"] = _canonical_sha256(audit["preprocessing"])
    rows = []
    manifest = data / "manifest.csv"
    try:
        content = manifest.read_bytes()
        audit["manifest_sha256"] = _bytes_sha256(content)
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not REQUIRED_COLUMNS.issubset(headers):
            issue("invalid_manifest_headers", "Required manifest column names must be present and unique.")
        else:
            rows = list(reader)
            if not rows:
                issue("empty_manifest", "The manifest contains no slice rows.")
    except (OSError, UnicodeError, csv.Error):
        issue("unreadable_manifest", "The manifest could not be read as UTF-8 CSV.")
    split_path = data / "splits.json"
    original_splits = {}
    try:
        content = split_path.read_bytes()
        audit["original_split_sha256"] = _bytes_sha256(content)
        original_splits = json.loads(content.decode("utf-8"))
        if (not isinstance(original_splits, dict) or set(original_splits) != {"train", "val", "test"}
                or any(not isinstance(ids, list) or any(not isinstance(case, str) or not CASE_PATTERN.fullmatch(case) for case in ids)
                       for ids in original_splits.values())):
            issue("invalid_original_splits", "Original split JSON must contain train/val/test lists of case IDs.")
            original_splits = {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        issue("unreadable_original_splits", "Original split JSON could not be read.")

    seen = set()
    by_case = defaultdict(list)
    expected_paths = set()
    decoded_image_groups = defaultdict(set)
    decoded_pair_groups = defaultdict(set)
    informative_slices = Counter()
    last_progress = 0
    for offset, row in enumerate(rows, start=2):
        case = row.get("case")
        valid_case = isinstance(case, str) and CASE_PATTERN.fullmatch(case)
        if not valid_case:
            issue("invalid_patient_id", "Manifest patient IDs must use case_XXXXX.", row_number=offset)
            continue
        by_case[case].append({"raw": {key: value for key, value in row.items() if key is not None},
                              "slice_index": None, "image_hash": None, "mask_hash": None})
        record = by_case[case][-1]
        if None in row or any(row.get(key) is None for key in REQUIRED_COLUMNS):
            issue("invalid_csv_row", "CSV row field count differs from its declared columns.", case, offset)
            continue
        try:
            z, n = _integer(row, "z"), _integer(row, "n_slices")
            counts = {key: _integer(row, key) for key in ("kidney_px", "tumor_px", "has_kidney", "has_tumor")}
            if n <= 0 or z >= n or counts["has_kidney"] not in (0, 1) or counts["has_tumor"] not in (0, 1):
                raise ValueError("invalid range")
        except (KeyError, TypeError, ValueError):
            issue("invalid_manifest_numbers", "Slice/count/flag metadata must be valid nonnegative integers.", case, offset)
            continue
        record.update(slice_index=z, n_slices=n)
        if (case, z) in seen:
            issue("duplicate_patient_slice", "A case/z pair appears more than once in the manifest.", case, offset)
        seen.add((case, z))
        decoded = {}
        for kind, folder in (("image", "images"), ("mask", "masks")):
            relative = "%s/%s_z%03d.png" % (folder, case, z)
            expected_paths.add(relative)
            record[kind] = relative
            try:
                path = _resolve_asset(data, row.get(kind), folder, case, z)
            except FileNotFoundError:
                issue("missing_png", "A referenced PNG file is missing.", case, offset, relative)
                continue
            except ValueError:
                issue("invalid_asset_path", "PNG paths must match the case/z filename and remain inside the data root.", case, offset)
                continue
            try:
                pixels, facts = _decode_png(path)
            except (OSError, ValueError, SyntaxError):
                issue("invalid_png", "A PNG failed header, format, dimensions or complete pixel decoding.", case, offset, relative)
                continue
            facts.update(kind=kind, patient_id=case, slice_index=z)
            audit["assets"][relative] = facts
            record[kind + "_hash"] = facts["decoded_pixel_sha256"]
            decoded[kind] = pixels
        informative_ct = False
        if "image" in decoded:
            central = decoded["image"][64:192, 64:192]
            informative_ct = (len(np.unique(central)) >= MINIMUM_DUPLICATE_CENTRAL_UNIQUE_VALUES
                              and float(central.std()) >= MINIMUM_DUPLICATE_CENTRAL_STD
                              and int(np.count_nonzero(central > 32)) >= MINIMUM_DUPLICATE_CENTRAL_BODY_PIXELS)
            if informative_ct:
                informative_slices[case] += 1
                decoded_image_groups[record["image_hash"]].add(case)
                if record["mask_hash"] is not None:
                    decoded_pair_groups[(record["image_hash"], record["mask_hash"])].add(case)
            audit["assets"][record["image"]]["informative_ct_for_duplicate_check"] = informative_ct
        if "mask" in decoded:
            mask = decoded["mask"]
            if not np.isin(mask, [0, 1, 2]).all():
                issue("invalid_mask_labels", "Mask pixels must be labels 0/1/2.", case, offset, record["mask"])
            else:
                label_counts = {str(label): int(np.count_nonzero(mask == label)) for label in (0, 1, 2)}
                audit["assets"][record["mask"]]["label_counts"] = label_counts
                actual = {"kidney_px": label_counts["1"], "tumor_px": label_counts["2"],
                          "has_kidney": int(label_counts["1"] > 0), "has_tumor": int(label_counts["2"] > 0)}
                if counts != actual:
                    issue("mask_metadata_mismatch", "Decoded mask counts/flags differ from the CSV.", case, offset, record["mask"])
                record["label_counts"] = label_counts
        if progress is not None and ((offset - 1) % 1000 == 0 or offset - 1 == len(rows)):
            progress({"completed_slice_count": offset - 1, "total_slice_count": len(rows)})
            last_progress = offset - 1
    if progress is not None and last_progress != len(rows):
        progress({"completed_slice_count": len(rows), "total_slice_count": len(rows)})

    inventory = set()
    for folder in ("images", "masks"):
        for path in (data / folder).rglob("*"):
            if path.is_file() and path.name != ".DS_Store":
                relative = path.relative_to(data).as_posix()
                inventory.add(relative)
                if path.name.endswith((".aria2", ".part", ".partial", ".tmp")):
                    issue("partial_download_file", "A download/partial-file marker remains in the PNG cache.", asset=relative)
    for relative in sorted(inventory - expected_paths):
        issue("orphan_asset", "A file in images/masks is outside the expected manifest paths.", asset=relative)
    known_cases = set(by_case)
    original_membership = defaultdict(list)
    for split, ids in original_splits.items():
        if len(ids) != len(set(ids)):
            issue("duplicate_original_split_member", "An original split repeats a patient ID.")
        for case in ids:
            original_membership[case].append(split)
    if original_splits:
        if set(original_membership) != known_cases:
            issue("original_split_membership_mismatch", "Original splits and manifest patient IDs do not cover the same patients.")
        for case, memberships in sorted(original_membership.items()):
            if len(memberships) > 1:
                issue("original_split_leakage", "A patient appears in multiple original splits.", case)

    series_groups = defaultdict(set)
    for case, records in sorted(by_case.items()):
        records.sort(key=lambda record: (record["slice_index"] is None, record["slice_index"] or 0))
        n_values = {record.get("n_slices") for record in records}
        if len(n_values) != 1 or None in n_values:
            issue("inconsistent_patient_slice_count", "A patient's n_slices metadata is inconsistent or invalid.", case)
        all_decoded = all(record["image_hash"] is not None and record["mask_hash"] is not None for record in records)
        image_series = [record["image_hash"] for record in records]
        fingerprint = _canonical_sha256(image_series) if all_decoded else None
        eligible = case not in bad_cases and all_decoded
        if eligible and informative_slices[case] > 0:
            series_groups[fingerprint].add(case)
        patient = {
            "patient_id": case, "source_patient_id": "kits19:" + case,
            "eligible": eligible, "selected_slice_count": len(records),
            "content_fingerprint_sha256": fingerprint,
            "manifest_metadata_sha256": _canonical_sha256([record["raw"] for record in records]),
            "original_split": original_membership.get(case, [None])[0],
            "original_scan_slice_count": next(iter(n_values)) if len(n_values) == 1 else None,
            "informative_ct_slice_count": informative_slices[case],
            "slice_assets": [{"slice_index": record["slice_index"],
                              "image": record.get("image"), "mask": record.get("mask")}
                             for record in records],
        }
        audit["patients"].append(patient)
    groups = [group for group in decoded_image_groups.values() if len(group) > 1]
    whole_series = [group for group in series_groups.values() if len(group) > 1]
    pair_groups = [group for group in decoded_pair_groups.values() if len(group) > 1]
    components = _duplicate_components(groups + whole_series)
    audit["duplicate_patient_components"] = components
    audit["duplicate_content_summary"] = {
        "meaningful_ct_hash_group_count": len(groups),
        "meaningful_ct_mask_pair_hash_group_count": len(pair_groups),
        "identical_selected_image_series_group_count": len(whole_series),
        "connected_patient_component_count": len(components),
        "patient_ids_in_components": sorted({case for component in components for case in component}),
    }
    if components:
        audit["warnings"].append({
            "code": "duplicate_patient_content_review", "severity": "warning",
            "message": "Exact meaningful selected CT content links patient IDs; use at most one representative per connected component in a later cohort.",
            "component_count": len(components),
        })
    audit["summary"] = {
        "manifest_slice_count": len(rows), "patient_count": len(by_case),
        "eligible_patient_count": sum(patient["eligible"] for patient in audit["patients"]),
        "decoded_image_count": sum(asset["kind"] == "image" for asset in audit["assets"].values()),
        "decoded_mask_count": sum(asset["kind"] == "mask" for asset in audit["assets"].values()),
        "inventory_file_count": len(inventory), "orphan_file_count": len(inventory - expected_paths),
        "original_split_counts": {split: len(ids) for split, ids in sorted(original_splits.items())},
    }
    audit["issues"] = [{"code": code, "severity": "error", "message": message,
                        "count": count, "examples": error_samples[(code, message)]}
                       for (code, message), count in sorted(error_counts.items())]
    audit["status"] = "passed" if not audit["issues"] else "failed"
    audit["audit_fingerprint_sha256"] = audit_fingerprint(audit)
    return audit
