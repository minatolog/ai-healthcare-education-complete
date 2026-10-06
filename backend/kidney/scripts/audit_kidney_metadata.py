#!/usr/bin/env python3
"""Read-only KiTS metadata/cache audit; never opens a raw CT volume.

Usage: python3 scripts/audit_kidney_metadata.py --root /path/to/kits_project
The report omits the input root and absolute paths. No files are changed unless
an explicit --output is supplied. PNG inspection reads only each file's header;
it cannot establish mask pixel correctness or full NIfTI integrity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REQUIRED_COLUMNS = {
    "case", "z", "image", "mask", "has_kidney", "has_tumor",
    "kidney_px", "tumor_px", "n_slices",
}
EXPECTED_SPLITS = {"train": 20, "val": 10, "test": 10}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_path(base: Path, value: str) -> Path | None:
    relative = Path(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        return None
    path = (base / relative).resolve()
    return path if path.is_relative_to(base.resolve()) else None


def _png_header(path: Path) -> tuple[int, int, int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(33)
    except OSError:
        return None
    if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR" or header[8:12] != b"\0\0\0\r"):
        return None
    width, height, depth, colour = struct.unpack(">IIBB", header[16:26])
    return width, height, depth, colour


def _png_inventory(directory: Path) -> tuple[set[str], dict]:
    files = sorted(p for p in directory.glob("*.png") if p.is_file())
    headers = Counter()
    unreadable = 0
    for path in files:
        header = _png_header(path)
        if header is None:
            unreadable += 1
        else:
            headers[str(header)] += 1
    return {p.stem for p in files}, {
        "file_count": len(files), "invalid_or_unreadable_headers": unreadable,
        "patient_stem_count": len({p.stem.rsplit("_z", 1)[0] for p in files}),
        "width_height_bit_depth_colour_type": dict(sorted(headers.items())),
    }


def audit(root: Path, expected_splits: dict[str, int] | None = None) -> dict:
    """Audit CSV/JSON metadata, split identities, PNG pairs and raw file presence."""
    expected = EXPECTED_SPLITS if expected_splits is None else expected_splits
    data = root / "data2d"
    issues: list[dict] = []

    def issue(code: str, severity: str = "error", **details) -> None:
        issues.append({"code": code, "severity": severity, **details})

    report = {
        "schema_version": "1.0", "read_only": True,
        "raw_volume_contents_loaded": False, "absolute_paths_included": False,
        "expected_splits": dict(expected), "issues": issues,
    }
    manifest = data / "manifest.csv"
    split_path = data / "splits.json"
    for path in (manifest, split_path):
        if not path.is_file():
            issue("missing_metadata", file=path.name)
    if issues:
        report["status"] = "action_required"
        return report
    try:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, strict=True)
            headers = reader.fieldnames or []
            empty_headers = sum(not isinstance(name, str) or not name.strip()
                                for name in headers)
            duplicate_headers = len(headers) - len(set(headers))
            # Check the header before DictReader can overwrite conflicting IDs.
            if empty_headers or duplicate_headers:
                issue("invalid_manifest_headers", empty_header_count=empty_headers,
                      duplicate_header_count=duplicate_headers)
                report["status"] = "action_required"
                return report
            columns = set(headers)
            rows = list(reader)
        splits = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as error:
        issue("unreadable_metadata", error_type=type(error).__name__)
        report["status"] = "action_required"
        return report
    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    if missing_columns:
        issue("missing_columns", columns=missing_columns)
    else:
        # DictReader uses None for absent fields and as the key for extra fields.
        # Reject malformed records before file-path handling or patient counting.
        malformed_rows = sum(
            None in row or any(value is None for value in row.values())
            or any(not isinstance(row[kind], str) or not row[kind].strip()
                   for kind in ("image", "mask"))
            for row in rows)
        if malformed_rows:
            issue("malformed_manifest_rows", count=malformed_rows)
    if not rows:
        issue("empty_manifest")
    if (not isinstance(splits, dict)
            or any(not isinstance(name, str) or not isinstance(ids, list)
                   or any(not isinstance(case, str) for case in ids)
                   for name, ids in splits.items())):
        issue("invalid_split_schema")
    if issues:
        report["status"] = "action_required"
        return report

    report["metadata_sha256"] = {
        "manifest.csv": _sha256(manifest), "splits.json": _sha256(split_path),
    }
    cases = {row["case"] for row in rows}
    tumour_cases: set[str] = set()
    case_rows = Counter()
    available_pairs = Counter()
    source_ids = defaultdict(set)
    slice_keys = Counter()
    paths_seen = {"image": defaultdict(set), "mask": defaultdict(set)}
    invalid_rows = 0
    missing_images = missing_masks = unsafe_paths = stem_mismatches = 0
    row_tumour_count = row_kidney_count = 0
    n_slices_by_case = defaultdict(set)
    context_needed: set[str] = set()
    for row in rows:
        case = row["case"]
        case_rows[case] += 1
        source_ids[row.get("source_patient_id") or case].add(case)
        try:
            z, n = int(row["z"]), int(row["n_slices"])
            kidney, tumour = int(row["has_kidney"]), int(row["has_tumor"])
            kidney_px, tumour_px = int(row["kidney_px"]), int(row["tumor_px"])
            if (not case or n <= 0 or z < 0 or z >= n
                    or kidney not in (0, 1) or tumour not in (0, 1)
                    or kidney_px < 0 or tumour_px < 0
                    or kidney != int(kidney_px > 0)
                    or tumour != int(tumour_px > 0)):
                raise ValueError("invalid metadata")
        except (TypeError, ValueError):
            invalid_rows += 1
            continue
        slice_keys[(case, z)] += 1
        n_slices_by_case[case].add(n)
        row_kidney_count += kidney
        row_tumour_count += tumour
        if tumour:
            tumour_cases.add(case)
        stem = f"{case}_z{z:03d}"
        for neighbour in (max(0, z - 1), z, min(n - 1, z + 1)):
            context_needed.add(f"{case}_z{neighbour:03d}")
        exists = {}
        for kind in ("image", "mask"):
            value = row[kind]
            path = _safe_path(data, value)
            paths_seen[kind][value].add(case)
            if path is None:
                unsafe_paths += 1
            if Path(value).stem != stem:
                stem_mismatches += 1
            exists[kind] = path is not None and path.is_file()
        missing_images += not exists["image"]
        missing_masks += not exists["mask"]
        if exists["image"] and exists["mask"]:
            available_pairs[case] += 1
    if invalid_rows:
        issue("invalid_manifest_rows", count=invalid_rows)
    if unsafe_paths:
        issue("unsafe_manifest_paths", count=unsafe_paths)
    if stem_mismatches:
        issue("case_slice_filename_mismatch", count=stem_mismatches)
    duplicate_slices = sum(count - 1 for count in slice_keys.values())
    if duplicate_slices:
        issue("duplicate_case_slice", count=duplicate_slices)
    inconsistent_shapes = [case for case, values in n_slices_by_case.items()
                           if len(values) > 1]
    if inconsistent_shapes:
        issue("inconsistent_n_slices", count=len(inconsistent_shapes))

    split_sets = {name: set(ids) for name, ids in splits.items()}
    split_union = set().union(*split_sets.values())
    overlaps = {f"{a}/{b}": sorted(split_sets[a] & split_sets[b])
                for a, b in combinations(sorted(split_sets), 2)}
    for name, ids in splits.items():
        if len(ids) != len(set(ids)):
            issue("duplicate_patient_in_split", split=name,
                  count=len(ids) - len(set(ids)))
    if any(overlaps.values()):
        issue("patient_split_leakage", overlaps=overlaps)
    expected_mismatches = {
        name: {"expected": count, "actual": len(splits.get(name, []))}
        for name, count in expected.items()
        if len(splits.get(name, [])) != count
    }
    if expected_mismatches:
        issue("split_count_mismatch", splits=expected_mismatches)
    unknown = sorted(split_union - cases)
    unassigned = sorted(cases - split_union)
    if unknown:
        issue("split_patient_missing_from_manifest", count=len(unknown))
    if unassigned:
        issue("unassigned_manifest_patients", "warning", count=len(unassigned))
    source_leaks = {}
    for source, related_cases in source_ids.items():
        memberships = sorted(name for name, ids in split_sets.items()
                             if related_cases & ids)
        if len(memberships) > 1:
            source_leaks[source] = memberships
    if source_leaks:
        issue("source_patient_split_leakage", groups=source_leaks)
    repeated_paths = {
        kind: sum(len(ids) > 1 for ids in mapping.values())
        for kind, mapping in paths_seen.items()
    }
    if any(repeated_paths.values()):
        issue("file_path_reused_across_patients", counts=repeated_paths)

    images, image_headers = _png_inventory(data / "images")
    masks, mask_headers = _png_inventory(data / "masks")
    manifest_images = {Path(row["image"]).stem for row in rows}
    manifest_masks = {Path(row["mask"]).stem for row in rows}
    complete_cases = sorted(case for case in cases
                            if available_pairs[case] == case_rows[case]
                            and case_rows[case] > 0)
    cache = {
        "images": image_headers, "masks": mask_headers,
        "paired_png_stems": len(images & masks),
        "image_only_stems": len(images - masks),
        "mask_only_stems": len(masks - images),
        "images_outside_manifest": len(images - manifest_images),
        "masks_outside_manifest": len(masks - manifest_masks),
        "missing_manifest_image_rows": missing_images,
        "missing_manifest_mask_rows": missing_masks,
        "complete_manifest_patient_count": len(complete_cases),
        "complete_manifest_patient_ids": complete_cases,
        "context_2_5d_required_images": len(context_needed),
        "context_2_5d_missing_images": len(context_needed - images),
    }
    if missing_images or missing_masks:
        issue("incomplete_slice_cache", images=missing_images, masks=missing_masks)
    for kind, headers in (("images", image_headers), ("masks", mask_headers)):
        if headers["invalid_or_unreadable_headers"]:
            issue("invalid_png_headers", kind=kind,
                  count=headers["invalid_or_unreadable_headers"])

    raw = root / "kits19" / "data"
    raw_cases = sorted(p for p in raw.glob("case_*") if p.is_dir())
    raw_images = {p.name for p in raw_cases if (p / "imaging.nii.gz").is_file()}
    raw_masks = {p.name for p in raw_cases if (p / "segmentation.nii.gz").is_file()}
    raw_ready = {p.name for p in raw_cases
                 if p.name in raw_images & raw_masks
                 and not (p / "imaging.nii.gz.aria2").exists()}
    report["raw_inventory"] = {
        "patient_directories": len(raw_cases), "imaging_files": len(raw_images),
        "segmentation_files": len(raw_masks),
        "paired_files": len(raw_images & raw_masks),
        "ready_patient_file_pairs": len(raw_ready),
        "file_existence_only_not_integrity_verified": True,
    }
    if not raw_ready and missing_masks:
        issue("cannot_rebuild_labelled_cache_from_local_raw_files")
    report["manifest"] = {
        "rows": len(rows), "patients": len(cases),
        "slices_with_kidney": row_kidney_count,
        "slices_with_tumour": row_tumour_count,
        "patients_with_tumour_slice": len(tumour_cases),
        "patients_without_observed_tumour_slice": len(cases - tumour_cases),
        "duplicate_case_slice_rows": duplicate_slices,
    }
    report["splits"] = {
        "patient_counts": {name: len(ids) for name, ids in splits.items()},
        "pairwise_patient_overlaps": overlaps,
        "source_identity_field": "source_patient_id" if "source_patient_id" in columns else "case",
        "source_identity_not_available_beyond_manifest": "source_patient_id" not in columns,
        "source_patient_overlap_groups": source_leaks,
        "unassigned_manifest_patient_count": len(unassigned),
        "unknown_split_patient_count": len(unknown),
        "per_split": {
            name: {
                "slice_rows": sum(case_rows[c] for c in ids),
                "tumour_positive_patients": len(ids & tumour_cases),
                "complete_cached_patients": len(ids & set(complete_cases)),
            } for name, ids in split_sets.items()
        },
    }
    report["slice_cache"] = cache
    report["patient_binary_classification"] = {
        "confirmed_negative_patients": None,
        "all_observed_patients_have_tumour": bool(cases) and cases == tumour_cases,
        "can_establish_two_patient_classes_from_this_manifest": False,
        "reason": ("Every observed patient has a tumour-containing slice."
                   if cases and cases == tumour_cases else
                   "Absence of tumour in selected slices does not establish a negative patient."),
        "non_tumour_slice_is_not_a_negative_patient": True,
    }
    if cases and cases == tumour_cases:
        issue("no_negative_patient_cohort", "warning", count=len(cases))
    stats_path = data / "dataset_stats.json"
    if stats_path.is_file():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if not isinstance(stats, dict):
                raise TypeError("dataset stats must be an object")
            stats_counts = {key: stats.get(key) for key in
                            ("n_slices", "n_cases", "slices_with_kidney",
                             "slices_with_tumor", "cases_with_tumor")}
            report["dataset_stats_counts"] = stats_counts
            actual = {
                "n_slices": len(rows), "n_cases": len(cases),
                "slices_with_kidney": row_kidney_count,
                "slices_with_tumor": row_tumour_count,
                "cases_with_tumor": len(tumour_cases),
            }
            mismatch = {k: {"stored": stats[k], "observed": value}
                        for k, value in actual.items() if k in stats and stats[k] != value}
            if mismatch:
                issue("stale_dataset_stats", counts=mismatch)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            issue("unreadable_dataset_stats", "warning")
    report["status"] = ("action_required" if any(i["severity"] == "error" for i in issues)
                        else "audited_with_warnings" if issues else "metadata_checks_passed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected", default="train=20,val=10,test=10",
                        help="comma-separated split=count pairs")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when the audit reports an error")
    args = parser.parse_args()
    try:
        expected = {name: int(count) for name, count in
                    (item.split("=", 1) for item in args.expected.split(","))}
        if any(not name or count < 0 for name, count in expected.items()):
            raise ValueError
    except ValueError:
        parser.error("--expected requires nonnegative split=count pairs")
    report = audit(args.root, expected)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return int(args.strict and report["status"] == "action_required")


if __name__ == "__main__":
    raise SystemExit(main())
