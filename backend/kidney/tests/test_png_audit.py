"""Small real PNG fixtures for deterministic machine QA; no model or patients."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from kidney_segmentation.png_audit import audit_fingerprint, audit_png_dataset


def informative_image(offset=0):
    y, x = np.indices((256, 256))
    return ((x + y * 3 + offset) % 256).astype(np.uint8)


def mask_fixture():
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[90:110, 90:110] = 1
    mask[100:105, 100:105] = 2
    return mask


def dataset(tmp_path, cases=("case_00000",), images=None, masks=None, slices=1):
    data = tmp_path / "data2d"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir()
    rows = []
    for index, case in enumerate(cases):
        for z in range(slices):
            image = images[index] if images is not None else informative_image(index + z)
            mask = masks[index] if masks is not None else mask_fixture()
            paths = {kind: "%s/%s_z%03d.png" % (folder, case, z)
                     for kind, folder in (("image", "images"), ("mask", "masks"))}
            Image.fromarray(image).save(data / paths["image"])
            Image.fromarray(mask).save(data / paths["mask"])
            rows.append({"case": case, "z": z, **paths, "n_slices": slices,
                         "kidney_px": int(np.count_nonzero(mask == 1)),
                         "tumor_px": int(np.count_nonzero(mask == 2)),
                         "has_kidney": int(np.any(mask == 1)), "has_tumor": int(np.any(mask == 2))})
    write_rows(data, rows)
    (data / "splits.json").write_text(json.dumps({"train": list(cases), "val": [], "test": []}))
    return data, rows


def write_rows(data, rows):
    with (data / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def codes(audit):
    return {issue["code"] for issue in audit["issues"]}


def test_complete_pixel_audit_is_deterministic_readonly_and_scoped(tmp_path):
    data, rows = dataset(tmp_path)
    before = {p.relative_to(tmp_path).as_posix(): p.read_bytes()
              for p in tmp_path.rglob("*") if p.is_file()}
    progress = []
    first = audit_png_dataset(tmp_path, progress.append)
    second = audit_png_dataset(data)
    assert first == second
    assert first["status"] == "passed" and first["issues"] == []
    assert first["schema_version"] == "png_dataset_audit_v1"
    assert first["scope"] == "selected_manifest_slices"
    assert first["visual_anatomical_alignment_verified"] is False
    assert first["full_scan_coverage"] is False
    assert first["source_patient_aliases_verified"] is False
    assert first["patients"][0]["source_patient_id"] == "kits19:case_00000"
    assert first["patients"][0]["eligible"] is True
    assert first["patients"][0]["selected_slice_count"] == 1
    assert progress[-1] == {"completed_slice_count": 1, "total_slice_count": 1}
    asset = first["assets"][rows[0]["image"]]
    assert asset["decoded_pixel_sha256"] == hashlib.sha256(informative_image().tobytes()).hexdigest()
    assert asset["byte_sha256"] == hashlib.sha256((data / rows[0]["image"]).read_bytes()).hexdigest()
    assert first["audit_fingerprint_sha256"] == audit_fingerprint(first)
    assert str(tmp_path) not in json.dumps(first)
    after = {p.relative_to(tmp_path).as_posix(): p.read_bytes()
             for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_corrupt_png_reports_error_and_ineligible_patient(tmp_path):
    data, rows = dataset(tmp_path)
    (data / rows[0]["image"]).write_bytes(b"not a PNG")
    audit = audit_png_dataset(tmp_path)
    assert "invalid_png" in codes(audit)
    assert audit["status"] == "failed"
    assert audit["patients"][0]["eligible"] is False
    assert audit["patients"][0]["content_fingerprint_sha256"] is None


def test_decoded_labels_and_counts_are_checked(tmp_path):
    data, rows = dataset(tmp_path)
    rows[0]["tumor_px"] += 1
    write_rows(data, rows)
    audit = audit_png_dataset(tmp_path)
    assert "mask_metadata_mismatch" in codes(audit)
    assert audit["patients"][0]["eligible"] is False
    mask = mask_fixture()
    mask[0, 0] = 3
    Image.fromarray(mask).save(data / rows[0]["mask"])
    assert "invalid_mask_labels" in codes(audit_png_dataset(tmp_path))


def test_duplicate_case_z_and_filename_path_mismatch_rejected(tmp_path):
    data, rows = dataset(tmp_path)
    write_rows(data, rows + rows)
    assert "duplicate_patient_slice" in codes(audit_png_dataset(tmp_path))
    rows[0]["image"] = "images/case_99999_z000.png"
    write_rows(data, rows)
    assert "invalid_asset_path" in codes(audit_png_dataset(tmp_path))


def test_outside_symlink_is_rejected(tmp_path):
    data, rows = dataset(tmp_path)
    outside = tmp_path / "outside.png"
    Image.fromarray(informative_image()).save(outside)
    image = data / rows[0]["image"]
    image.unlink()
    image.symlink_to(outside)
    assert "invalid_asset_path" in codes(audit_png_dataset(tmp_path))


def test_duplicate_informative_images_use_ct_only_not_mask_labels(tmp_path):
    blank = np.zeros((256, 256), dtype=np.uint8)
    data, rows = dataset(tmp_path, ("case_00000", "case_00001"),
                         images=[informative_image(), informative_image()],
                         masks=[blank, mask_fixture()])
    audit = audit_png_dataset(tmp_path)
    assert audit["status"] == "passed" and not audit["issues"]
    assert audit["duplicate_patient_components"] == [["case_00000", "case_00001"]]
    assert audit["duplicate_detection_policy"]["uses_reference_foreground_presence"] is False
    assert audit["warnings"][0]["code"] == "duplicate_patient_content_review"
    assert all(patient["eligible"] for patient in audit["patients"])


def test_constant_and_peripheral_only_duplicate_images_ignored(tmp_path):
    image = np.zeros((256, 256), dtype=np.uint8)
    image[:32] = informative_image()[:32]
    dataset(tmp_path, ("case_00000", "case_00001"), images=[image, image])
    audit = audit_png_dataset(tmp_path)
    assert audit["status"] == "passed"
    assert audit["duplicate_patient_components"] == []
    assert all(patient["informative_ct_slice_count"] == 0 for patient in audit["patients"])


def test_duplicate_groups_are_transitive_patient_components(tmp_path):
    data, rows = dataset(tmp_path, ("case_00000", "case_00001", "case_00002"), slices=2)
    a, b, c = informative_image(0), informative_image(20), informative_image(40)
    images = [a, b, b, c, c, informative_image(60)]
    for row, image in zip(rows, images):
        Image.fromarray(image).save(data / row["image"])
    audit = audit_png_dataset(tmp_path)
    assert audit["duplicate_patient_components"] == [["case_00000", "case_00001", "case_00002"]]


def test_orphans_partial_markers_and_missing_assets_rejected(tmp_path):
    data, rows = dataset(tmp_path)
    (data / "images" / ".DS_Store").write_bytes(b"ignored")
    assert audit_png_dataset(tmp_path)["status"] == "passed"
    (data / "images" / "orphan.png").write_bytes(b"unused")
    (data / "masks" / "download.png.aria2").write_bytes(b"partial")
    (data / rows[0]["mask"]).unlink()
    audit = audit_png_dataset(tmp_path)
    assert {"orphan_asset", "partial_download_file", "missing_png"} <= codes(audit)
    assert audit["summary"]["orphan_file_count"] == 2


def test_bad_headers_and_original_split_leakage_rejected(tmp_path):
    data, rows = dataset(tmp_path)
    (data / "splits.json").write_text(json.dumps({"train": ["case_00000"], "val": ["case_00000"], "test": []}))
    assert "original_split_leakage" in codes(audit_png_dataset(tmp_path))
    with (data / "manifest.csv").open("w") as handle:
        handle.write("case,case,z\ncase_00000,case_00000,0\n")
    assert "invalid_manifest_headers" in codes(audit_png_dataset(tmp_path))


def test_out_of_range_z_wrong_dimensions_and_mode_rejected(tmp_path):
    data, rows = dataset(tmp_path)
    rows[0]["z"] = 1
    write_rows(data, rows)
    assert "invalid_manifest_numbers" in codes(audit_png_dataset(tmp_path))
    rows[0]["z"] = 0
    write_rows(data, rows)
    Image.new("L", (128, 128)).save(data / rows[0]["image"])
    assert "invalid_png" in codes(audit_png_dataset(tmp_path))
    Image.new("RGB", (256, 256)).save(data / rows[0]["image"])
    assert "invalid_png" in codes(audit_png_dataset(tmp_path))


def test_fingerprint_changes_with_pixels_and_covers_all_metadata(tmp_path):
    data, rows = dataset(tmp_path)
    first = audit_png_dataset(tmp_path)
    Image.fromarray(informative_image(1)).save(data / rows[0]["image"])
    second = audit_png_dataset(tmp_path)
    assert first["audit_fingerprint_sha256"] != second["audit_fingerprint_sha256"]
    second["limitations"].append("An added limitation must change the fingerprint.")
    assert second["audit_fingerprint_sha256"] != audit_fingerprint(second)
