#!/usr/bin/env python3
"""Step 2: read one real 2D CT/mask pair and display reference annotations.

Run from the starter directory, for example:
python3 learning/02_inspect_2d_pair.py --root ../kits_project

This script does not train a model, import PyTorch, read NIfTI, or modify data.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXPECTED_SIZE = (256, 256)
LABELS = {0: "background", 1: "kidney", 2: "tumour"}
COLOURS = np.array([[0, 0, 0], [0, 200, 0], [255, 0, 0]], dtype=np.uint8)


# 1. Locate exactly one CSV record and keep every input inside the data directory.
def find_record(root, case, slice_index):
    if not re.fullmatch(r"case_[0-9]{5}", case) or slice_index < 0:
        raise ValueError("Use a patient ID such as case_00000 and a non-negative slice index.")
    data = (root / "data2d").resolve()
    manifest = (data / "manifest.csv").resolve()
    if root not in data.parents or data not in manifest.parents:
        raise ValueError("The data2d directory and manifest must stay inside kits_project.")
    content = manifest.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    required = {"case", "z", "image", "mask", "kidney_px", "tumor_px", "has_kidney", "has_tumor", "n_slices"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("The manifest is missing required patient, slice or mask-metadata columns.")
    matches = []
    for row in reader:
        if row["case"] == case and integer_field(row, "z") == slice_index:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError("Expected exactly one manifest row for %s at z=%d; found %d." % (case, slice_index, len(matches)))
    row = matches[0]
    if slice_index >= integer_field(row, "n_slices"):
        raise ValueError("The slice index is outside the volume length recorded in the manifest.")
    return data, manifest, sha256(content), row


def integer_field(row, field):
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Manifest field '%s' must contain an integer." % field) from None


def checked_asset(data, relative, folder, case, slice_index):
    if not isinstance(relative, str) or not relative:
        raise ValueError("The selected manifest %s path is missing." % folder)
    relative = Path(relative)
    expected = "%s_z%03d.png" % (case, slice_index)
    if relative.is_absolute() or relative.parts != (folder, expected):
        raise ValueError("Expected the manifest %s path to be %s/%s." % (folder, folder, expected))
    path = (data / relative).resolve()
    if data not in path.parents or not path.is_file():
        raise ValueError("The selected PNG is missing or resolves outside data2d: %s" % relative)
    return path


def sha256(content):
    return hashlib.sha256(content).hexdigest()


# 2. Decode the original PNG bytes. Conversion to RGB is for display only.
def read_greyscale_png(path):
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != EXPECTED_SIZE:
            raise ValueError("Expected a grayscale L-mode 256x256 PNG: %s" % path.name)
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated PNGs are not supported: %s" % path.name)
        pixels = np.array(image, dtype=np.uint8)
    return pixels, sha256(content)


def array_statistics(pixels):
    return {"shape": list(pixels.shape), "dtype": str(pixels.dtype),
            "minimum": int(pixels.min()), "maximum": int(pixels.max())}


# 3. Compare decoded label counts with the CSV, rather than trusting metadata alone.
def verify_mask(mask, row):
    values, frequencies = np.unique(mask, return_counts=True)
    if not np.isin(values, [0, 1, 2]).all():
        raise ValueError("Mask labels must be 0/1/2; observed %s." % values.tolist())
    counts = {str(label): int((mask == label).sum()) for label in LABELS}
    observed = {"kidney_px": counts["1"], "tumor_px": counts["2"],
                "has_kidney": int(counts["1"] > 0), "has_tumor": int(counts["2"] > 0)}
    comparisons = {field: {"manifest_value": integer_field(row, field), "decoded_mask_value": value,
                            "matches": integer_field(row, field) == value}
                   for field, value in observed.items()}
    mismatches = [field for field, comparison in comparisons.items() if not comparison["matches"]]
    if mismatches:
        raise ValueError("Mask metadata does not match decoded pixels: %s. Comparisons: %s" %
                         (", ".join(mismatches), json.dumps(comparisons)))
    unique_counts = [{"label": int(label), "name": LABELS[int(label)], "pixels": int(count)}
                     for label, count in zip(values, frequencies)]
    return counts, unique_counts, comparisons


# 4. Build reference displays; no output here is a model prediction.
def reference_images(ct, mask, alpha=0.4):
    ct_rgb = np.repeat(ct[..., None], 3, axis=2)
    coloured = COLOURS[mask]
    overlay = ct_rgb.copy()
    foreground = mask > 0
    blended = (1 - alpha) * ct_rgb[foreground].astype(np.float32) + alpha * coloured[foreground].astype(np.float32)
    overlay[foreground] = np.round(blended).astype(np.uint8)
    # Background pixels remain identical to the CT, with no darkening.
    return Image.fromarray(ct_rgb), Image.fromarray(coloured), Image.fromarray(overlay)


def labelled_preview(images, case, slice_index):
    margin, panel_size = 16, 256
    canvas = Image.new("RGB", (3 * panel_size + 4 * margin, 432), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=18)
    label_font = ImageFont.load_default(size=14)
    draw.text((margin, 14), "Ground-truth/reference annotation only | No model prediction", fill="black", font=title_font)
    draw.text((margin, 43), "Patient: %s | Original slice index: %d" % (case, slice_index), fill="black", font=label_font)
    titles = ("CT image (processed PNG)", "Reference mask (annotation)", "Reference overlay (annotation)")
    for index, (image, title) in enumerate(zip(images, titles)):
        x = margin + index * (panel_size + margin)
        draw.text((x, 75), title, fill="black", font=label_font)
        canvas.paste(image, (x, 102))
    draw.text((margin, 374), "Labels: 0 background = black | 1 kidney = green | 2 tumour = red", fill="black", font=label_font)
    draw.text((margin, 401), "One selected exported 2D slice; not full-volume inference. No model was trained.", fill="black", font=label_font)
    return canvas


def checked_destination(output, name):
    path = output / name
    if output not in path.resolve().parents:
        raise ValueError("An output file resolves outside the output directory: %s" % name)
    return path


# 5. Save learning outputs outside the original project and record their provenance.
def inspect(args):
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The --root kits_project directory does not exist: %s" % root)
    if output == root or root in output.parents:
        raise ValueError("Save --output-dir outside kits_project; original data must remain unchanged.")
    data, manifest_path, manifest_hash, row = find_record(root, args.case, args.slice_index)
    image_path = checked_asset(data, row["image"], "images", args.case, args.slice_index)
    mask_path = checked_asset(data, row["mask"], "masks", args.case, args.slice_index)
    ct, image_hash = read_greyscale_png(image_path)
    mask, mask_hash = read_greyscale_png(mask_path)
    if ct.shape != mask.shape:
        raise ValueError("The CT and reference mask must have identical pixel dimensions.")
    counts, unique_counts, comparisons = verify_mask(mask, row)
    ct_stats, mask_stats = array_statistics(ct), array_statistics(mask)
    print("\n[DECODED CT PNG]")
    print("shape=%s; dtype=%s; range=%d..%d" % (ct.shape, ct.dtype, ct.min(), ct.max()))
    print("These are processed display intensities, not raw HU values.")
    print("\n[DECODED REFERENCE MASK]")
    print("shape=%s; dtype=%s; range=%d..%d" % (mask.shape, mask.dtype, mask.min(), mask.max()))
    for label in unique_counts:
        print("label=%d (%s); pixels=%d" % (label["label"], label["name"], label["pixels"]))
    print("CSV kidney_px/tumor_px/has_kidney/has_tumor match the decoded mask.")

    output.mkdir(parents=True, exist_ok=True)
    images = reference_images(ct, mask)
    files = {"ct_display": "ct_display.png", "reference_mask_coloured": "reference_mask_coloured.png",
             "reference_overlay": "reference_overlay.png", "reference_preview": "reference_preview.png"}
    destinations = {key: checked_destination(output, name) for key, name in files.items()}
    report_path = checked_destination(output, "inspection.json")
    for image, key in zip(images, ("ct_display", "reference_mask_coloured", "reference_overlay")):
        image.save(destinations[key])
    labelled_preview(images, args.case, args.slice_index).save(destinations["reference_preview"])
    report = {
        "step": "02_inspect_2d_pair", "status": "completed", "patient_id": args.case, "slice_index": args.slice_index,
        "model_trained": False, "contains_model_prediction": False,
        "source_files": {"manifest": str(manifest_path), "ct_image": str(image_path), "reference_mask": str(mask_path)},
        "source_sha256": {"manifest": manifest_hash, "ct_image": image_hash, "reference_mask": mask_hash},
        "ct_statistics": ct_stats, "reference_mask_statistics": mask_stats,
        "mask_label_map": {str(label): name for label, name in LABELS.items()}, "mask_label_counts": counts,
        "observed_unique_mask_labels": unique_counts, "manifest_record": row, "metadata_comparisons": comparisons,
        "visualization_identity": {"patient_id": args.case, "slice_index": args.slice_index,
                                   "source_ct_sha256": image_hash, "source_reference_mask_sha256": mask_hash,
                                   "coordinate_space": "processed_png_256x256", "type": "ground_truth_reference_annotation",
                                   "contains_model_prediction": False, "output_files": files,
                                   "foreground_overlay_alpha": 0.4, "background_pixels_unchanged": True},
        "selection_scope": "One exported 2D slice selected by case and original z index from manifest.csv; not a full-volume evaluation.",
        "development_inspection": {"case_inspected_for_development": args.case, "cohort_selected": False,
                                   "recommended_future_role": "training_patient_or_outside_formal_cohort",
                                   "must_not_be_presented_as_unseen_final_test": True,
                                   "must_not_be_presented_as_independent_prediction": True},
        "educational_message": "These colours display a ground-truth/reference annotation. They are not model predictions or a patient diagnosis."
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("\nReference displays and inspection.json saved to: %s" % output)
    print("No model was trained. No source CT, mask or manifest was modified.")
    print("This patient was inspected for development; later assign it to training or keep it outside the formal cohort.")
    print("Do not present this case as an unseen final-test patient or an independent new-patient example.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 2: inspect one CT PNG and its ground-truth/reference mask without training")
    parser.add_argument("--root", type=Path, required=True, help="The kits_project directory, not data2d")
    parser.add_argument("--case", default="case_00000", help="Patient/case ID (default: case_00000)")
    parser.add_argument("--slice-index", type=int, default=294, help="Original CT slice index z (default: 294)")
    parser.add_argument("--output-dir", type=Path, default=Path("learning_outputs/step2"), help="Display/report directory outside kits_project")
    args = parser.parse_args(argv)
    try:
        inspect(args)
    except (OSError, ValueError, csv.Error, UnicodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
