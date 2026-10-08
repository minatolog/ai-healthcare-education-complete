#!/usr/bin/env python3
"""Step 1: inspect the existing 2D files without changing data or training.

Place this file in the starter's learning/ directory and run in VS Code:
python3 learning/01_check_2d_data.py --root ../kits_project

Only the Python standard library is required; PyTorch is not needed yet.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path


def load_auditor(starter_root):
    # Reuse the tested auditor rather than adding another split/training policy.
    script = starter_root / "scripts/audit_kidney_metadata.py"
    if not script.is_file():
        raise ValueError("Cannot find scripts/audit_kidney_metadata.py. Keep this entry in backend/kidney/learning/.")
    spec = importlib.util.spec_from_file_location("kidney_metadata_auditor", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example_pair(auditor, root):
    # Select a labelled example for inspection only, never to select test cases.
    data = root / "data2d"
    fallback = None
    with (data / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            image = auditor._safe_path(data, row.get("image", ""))
            mask = auditor._safe_path(data, row.get("mask", ""))
            if image is None or mask is None or not image.is_file() or not mask.is_file():
                continue
            if fallback is None:
                fallback = row
            if row.get("has_tumor") == "1":
                return row
    return fallback


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 1: check existing 2D CT PNGs, masks and manifest references")
    parser.add_argument("--root", type=Path, required=True, help="The kits_project directory, not the data2d directory")
    parser.add_argument("--output", type=Path, help="Report destination; default: step1_data_audit.json in the current directory")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output = args.output or Path("step1_data_audit.json")
    if not root.is_dir():
        parser.error("The --root directory does not exist: %s" % root)
    # Keep reports outside the original project; save them in the starter.
    destination = output.expanduser().resolve()
    if destination == root or root in destination.parents:
        parser.error("Save the report in the starter, outside kits_project.")
    try:
        auditor = load_auditor(Path(__file__).resolve().parents[1])
        report = auditor.audit(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print("\nRead-only file/index check. No model was trained and no CT/mask was modified.")
    print("Full report: %s" % output.resolve())
    if "manifest" not in report:
        print("Could not read complete metadata. Check that manifest.csv and splits.json exist and have valid formats.")
        for issue in report.get("issues", []):
            print("- %s" % issue["code"])
        return 1

    # Each case identifies a patient; each CSV row references an exported slice.
    manifest = report["manifest"]
    cache = report["slice_cache"]
    print("\n[CSV INDEX]")
    print("Manifest slice records: %s" % manifest["rows"])
    print("Manifest patients: %s" % manifest["patients"])
    print("The CSV is an index, not a 3D pixel array: case identifies the patient; z is the original CT slice index.")
    print("\n[LOCAL 2D FILES]")
    print("CT PNGs: %s" % cache["images"]["file_count"])
    print("Mask PNGs: %s" % cache["masks"]["file_count"])
    print("Matching CT/mask filename pairs: %s" % cache["paired_png_stems"])
    print("CT files without a matching mask: %s" % cache["image_only_stems"])
    print("Mask files without a matching CT: %s" % cache["mask_only_stems"])
    print("Manifest references missing CTs: %s; missing masks: %s" %
          (cache["missing_manifest_image_rows"], cache["missing_manifest_mask_rows"]))
    print("Patients with complete coverage of their selected manifest slices: %s" % cache["complete_manifest_patient_count"])
    print("This completeness refers to selected manifest slices, not all slices of the original scan.")
    print("\n[ORIGINAL PATIENT SPLIT]")
    print(json.dumps(report["splits"]["patient_counts"], ensure_ascii=False))
    print("Create a separate project split of 20 train / 10 validation / 10 test; preserve the original splits.json.")

    try:
        row = example_pair(auditor, root)
    except (OSError, csv.Error, UnicodeError):
        row = None
    if row:
        print("\n[MATCHING EXAMPLE TO OPEN NEXT]")
        print("case=%s; z=%s; CSV has_tumor=%s (metadata only; mask pixels have not been verified yet)" %
              (row["case"], row["z"], row.get("has_tumor", "unknown")))
        print("CT: %s" % (root / "data2d" / row["image"]))
        print("Mask: %s" % (root / "data2d" / row["mask"]))
        print("Expected mask labels are 0/1/2, so a greyscale preview may look nearly black. Colour overlays are for display only; keep the labels unchanged.")
    print("\n[CURRENT FINDINGS]")
    if cache["missing_manifest_image_rows"] or cache["missing_manifest_mask_rows"]:
        print("The local cache does not fully match the manifest. One real pair can demonstrate loading; the formal cohort requires restored files or a separately reviewed data scope.")
    else:
        print("All manifest CT/mask references exist. Pixel labels, dimensions, duplicate content and patient splits still require checks before formal training.")
    print("Pure 2D reads one PNG at a time. Missing neighbouring slices for 2.5D are not an input requirement for this step.")
    print("Complete processed PNGs do not require raw NIfTI downloads just for loading. Raw volumes support regeneration or later full-scan evaluation.")
    print("The original cache used reference masks to select slices. Report its results as segmentation evaluation on selected 2D slices.")
    print("\nCheck completed. Review this output and the issues in step1_data_audit.json before proceeding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
