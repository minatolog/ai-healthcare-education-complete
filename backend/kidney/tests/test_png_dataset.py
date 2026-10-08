"""Small PNG fixtures verify the adapter without selecting or training a cohort."""
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from kidney_segmentation.png_dataset import PNGConfig, PNGPatientDataset


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def png_files(tmp_path):
    data = tmp_path / "data2d"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir()
    rows = []
    for case, z in [("case_00000", 294), ("case_00000", 300), ("case_00001", 7)]:
        image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        if case == "case_00001":
            image = np.flip(image, axis=1).copy()
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10:12, 10:14] = 1
        mask[20:22, 20:22] = 2
        stem = "%s_z%03d.png" % (case, z)
        Image.fromarray(image).save(data / "images" / stem)
        Image.fromarray(mask).save(data / "masks" / stem)
        rows.append({"case": case, "z": z, "image": "images/" + stem, "mask": "masks/" + stem,
                     "kidney_px": 8, "tumor_px": 4, "has_kidney": 1, "has_tumor": 1, "n_slices": 611})
    manifest = data / "manifest.csv"
    write_manifest(manifest, rows)
    return data, manifest, rows


def test_single_channel_normalization_target_and_source_identity(png_files):
    data, manifest, _ = png_files
    dataset = PNGPatientDataset(manifest, ["case_00000"])
    sample = dataset[0]
    assert len(dataset) == 2
    assert sample["input"].shape == (1, 256, 256) and sample["input"].dtype == torch.float32
    assert sample["target"].shape == (256, 256) and sample["target"].dtype == torch.int64
    assert sample["input"][0, 0, 0].item() == pytest.approx((0 - .456) / .224)
    assert sample["input"][0, 0, 255].item() == pytest.approx((1 - .456) / .224)
    assert set(sample["target"].unique().tolist()) == {0, 1, 2}
    assert sample["target"][20, 20].item() == 2
    assert sample["mask_label_counts"] == {"0": 65524, "1": 8, "2": 4}
    assert sample["patient_id"] == "case_00000" and sample["slice_index"] == 294
    assert sample["sample_id"] == "case_00000:z294"
    assert sample["source_paths"]["image"] == str(data / "images/case_00000_z294.png")
    assert sample["source_sha256"]["image"] == hashlib.sha256((data / "images/case_00000_z294.png").read_bytes()).hexdigest()
    assert dataset.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert sample["config_sha256"] == dataset.config_sha256
    assert sample["annotation_type"] == "ground_truth_reference_segmentation"
    assert sample["contains_model_prediction"] is False


def test_mask_changes_do_not_change_input_and_source_files_are_read_only(png_files):
    data, manifest, rows = png_files
    dataset = PNGPatientDataset(manifest, ["case_00000"])
    first = dataset[0]
    image_path = data / rows[0]["image"]
    image_bytes = image_path.read_bytes()
    empty = np.zeros((256, 256), dtype=np.uint8)
    Image.fromarray(empty).save(data / rows[0]["mask"])
    rows[0].update(kidney_px=0, tumor_px=0, has_kidney=0, has_tumor=0)
    write_manifest(manifest, rows)
    # The fixture deliberately changes its reference annotation, not its image.
    second = PNGPatientDataset(manifest, ["case_00000"])[0]
    torch.testing.assert_close(first["input"], second["input"], rtol=0, atol=0)
    assert first["target"].sum() > 0 and second["target"].sum() == 0
    assert image_path.read_bytes() == image_bytes
    observed_paths = [manifest, image_path, data / rows[0]["mask"]]
    before = {path: path.read_bytes() for path in observed_paths}
    _ = PNGPatientDataset(manifest, ["case_00000"])[0]
    assert before == {path: path.read_bytes() for path in observed_paths}


def test_explicit_patient_order_batching_and_no_neighbour_or_split_requirement(png_files):
    data, manifest, _ = png_files
    # A deliberately invalid old split file must never be read by this adapter.
    (data / "splits.json").write_text("not a JSON split")
    dataset = PNGPatientDataset(manifest, ["case_00001", "case_00000"])
    assert [row["case"] for row in dataset.rows] == ["case_00001", "case_00000", "case_00000"]
    assert [row["z"] for row in dataset.rows] == [7, 294, 300]
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    assert batch["input"].shape == (2, 1, 256, 256)
    assert batch["target"].shape == (2, 256, 256)
    assert batch["patient_id"] == ["case_00001", "case_00000"]
    assert batch["slice_index"].tolist() == [7, 294]
    assert all(isinstance(value, str) for value in batch["source_paths"]["image"])
    assert "no split assignment" in dataset.selection_scope


@pytest.mark.parametrize("patients", [[], None, "case_00000", ["case_00000", "case_00000"], ["case_99999"], ["invalid"]])
def test_empty_duplicate_or_unknown_patient_selection_is_rejected(png_files, patients):
    _, manifest, _ = png_files
    with pytest.raises(ValueError):
        PNGPatientDataset(manifest, patients)


def test_duplicate_case_slice_identity_is_rejected_even_outside_selection(png_files):
    _, manifest, rows = png_files
    write_manifest(manifest, rows + [rows[-1]])
    with pytest.raises(ValueError, match="Duplicate patient/slice"):
        PNGPatientDataset(manifest, ["case_00000"])


@pytest.mark.parametrize("n_slices", [0, 294])
def test_existing_png_cannot_make_an_out_of_bounds_original_slice_valid(png_files, n_slices):
    _, manifest, rows = png_files
    rows[0]["n_slices"] = n_slices
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="positive n_slices"):
        PNGPatientDataset(manifest, ["case_00000"])


@pytest.mark.parametrize("path", ["../outside.png", "/tmp/outside.png", "images/case_00001_z294.png", "images/case_00000_z295.png", "images/case_00000_z294.jpg"])
def test_path_traversal_and_filename_patient_slice_mismatch_are_rejected(png_files, path):
    _, manifest, rows = png_files
    rows[0]["image"] = path
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Expected the selected"):
        PNGPatientDataset(manifest, ["case_00000"])


def test_symlink_escape_is_rechecked_when_a_sample_is_loaded(png_files, tmp_path):
    data, manifest, rows = png_files
    dataset = PNGPatientDataset(manifest, ["case_00000"])
    outside = tmp_path / "outside.png"
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(outside)
    image_path = data / rows[0]["image"]
    image_path.unlink()
    image_path.symlink_to(outside)
    with pytest.raises(ValueError, match="outside the approved"):
        dataset[0]


@pytest.mark.parametrize("invalid_kind", ["rgb_image", "wrong_size", "palette_mask", "invalid_mask_label"])
def test_invalid_png_modes_sizes_and_mask_labels_are_rejected(png_files, invalid_kind):
    data, manifest, rows = png_files
    if invalid_kind == "rgb_image":
        Image.new("RGB", (256, 256)).save(data / rows[0]["image"])
    elif invalid_kind == "wrong_size":
        Image.new("L", (128, 256)).save(data / rows[0]["image"])
    elif invalid_kind == "palette_mask":
        Image.new("P", (256, 256)).save(data / rows[0]["mask"])
    else:
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[0, 0] = 255
        Image.fromarray(mask).save(data / rows[0]["mask"])
    with pytest.raises(ValueError):
        PNGPatientDataset(manifest, ["case_00000"])[0]


@pytest.mark.parametrize("field,wrong", [("kidney_px", 7), ("tumor_px", 5), ("has_kidney", 0), ("has_tumor", 0)])
def test_reference_mask_metadata_mismatches_are_rejected(png_files, field, wrong):
    _, manifest, rows = png_files
    rows[0][field] = wrong
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="disagree with manifest"):
        PNGPatientDataset(manifest, ["case_00000"])[0]


def test_missing_uninspected_rows_are_retained_and_fail_on_access(png_files):
    data, manifest, rows = png_files
    (data / rows[1]["image"]).unlink()
    dataset = PNGPatientDataset(manifest, ["case_00000"])
    assert len(dataset) == 2
    assert dataset[0]["slice_index"] == 294
    with pytest.raises(ValueError, match="requested PNG file is missing"):
        dataset[1]


@pytest.mark.parametrize("kwargs", [{"input_channels": 3}, {"input_channels": True}, {"size": 64},
                                    {"mean": float("nan")}, {"std": 0}, {"std": float("inf")}, {"version": "unknown"}])
def test_unknown_or_invalid_normalization_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PNGConfig(**kwargs)
