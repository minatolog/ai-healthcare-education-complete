"""Training-only reference review checks using synthetic PNGs and real bindings."""
from argparse import Namespace
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import pytest

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from kidney_segmentation.cohort import make_cohort, validate_cohort
from kidney_segmentation.png_audit import audit_png_dataset

SPEC = importlib.util.spec_from_file_location(
    "step6a_cli_under_test", WORKSPACE / "learning" / "06a_review_training_pairs.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def json_bytes(content):
    return (json.dumps(content, indent=2, allow_nan=False) + "\n").encode("utf-8")


def snapshot(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def audited_template(tmp_path_factory):
    """Audit 90 small artificial pairs once; no actual patient data is accessed."""
    template = tmp_path_factory.mktemp("step6a_synthetic")
    root = template / "original_data"
    data = root / "data2d"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir()
    rows, cases = [], ["case_%05d" % index for index in range(45)]
    yy, xx = np.indices((256, 256))
    for index, case in enumerate(cases):
        for position, z in enumerate((2, 7)):
            # Ninety unique detailed images prevent cross-patient duplicate groups.
            ct = ((xx + 3 * yy + index * 2 + position) % 256).astype(np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask[80:100, 80:100] = 1
            mask[90:92 + position, 90:92 + position] = 2
            image = "images/%s_z%03d.png" % (case, z)
            mask_path = "masks/%s_z%03d.png" % (case, z)
            Image.fromarray(ct).save(data / image)
            Image.fromarray(mask).save(data / mask_path)
            rows.append({"case": case, "z": z, "image": image, "mask": mask_path,
                         "n_slices": 10, "kidney_px": int((mask == 1).sum()),
                         "tumor_px": int((mask == 2).sum()), "has_kidney": 1, "has_tumor": 1})
    with (data / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (data / "splits.json").write_bytes(json_bytes({
        "train": cases[:30], "val": cases[30:38], "test": cases[38:]}))
    audit = audit_png_dataset(root)
    assert audit["status"] == "passed" and audit["duplicate_patient_components"] == []
    lock = make_cohort(audit, seed=42, exposed_patient_ids=("case_00000",), prediction_count=5)
    validate_cohort(lock, audit)
    (template / "png_data_audit.json").write_bytes(json_bytes(audit))
    (template / "project40_splits.json").write_bytes(json_bytes(lock))
    return template


@pytest.fixture
def environment(tmp_path, monkeypatch, audited_template):
    # Copy only the synthetic fixture; tests may mutate their private copy.
    fixture = tmp_path / "fixture"
    shutil.copytree(audited_template, fixture)
    args = Namespace(root=fixture / "original_data", cohort=fixture / "project40_splits.json",
                     audit=fixture / "png_data_audit.json", output_dir=tmp_path / "review")
    lock = json.loads(args.cohort.read_bytes())
    audit = json.loads(args.audit.read_bytes())
    cohort_module, reference = CLI.load_helpers()
    decode = reference.read_greyscale_png
    reads = []
    allowed = {record["patient_id"] for record in lock["pools"]["train"]}
    forbidden = {record["patient_id"] for role in ("validation", "test", "prediction")
                 for record in lock["pools"][role]}

    def training_only_decode(path):
        patient = path.stem.split("_z")[0]
        assert patient in allowed and patient not in forbidden, "A held-out PNG was opened."
        assert args.root / "data2d" in path.parents
        reads.append(path.relative_to(args.root / "data2d").as_posix())
        return decode(path)

    monkeypatch.setattr(reference, "read_greyscale_png", training_only_decode)
    monkeypatch.setattr(CLI, "load_helpers", lambda: (cohort_module, reference))
    return args, lock, audit, reads


def test_review_opens_only_two_pairs_per_training_patient_and_keeps_inputs(environment):
    args, lock, audit, reads = environment
    before = snapshot(args.root)
    binding_bytes = (args.cohort.read_bytes(), args.audit.read_bytes())
    report = CLI.review(args)
    assert len(reads) == 80  # Forty CT/mask pairs, not eighty pairs.
    assert Counter(path.split("/")[0] for path in reads) == {"images": 40, "masks": 40}
    assert Counter(Path(path).stem.split("_z")[0] for path in reads) == {
        record["patient_id"]: 4 for record in lock["pools"]["train"]}
    assert report["patient_count"] == 20 and report["preview_pair_count"] == 40
    assert report["patient_ids"] == [record["patient_id"] for record in lock["pools"]["train"]]
    assert report["loaded_patient_pool"] == "train"
    assert report["manual_review_status"] == "pending"
    assert report["anatomical_alignment_verified"] is False
    assert report["contains_model_prediction"] is False
    assert report["model_run"] is False and report["training_started"] is False
    assert report["audit_fingerprint_sha256"] == audit["audit_fingerprint_sha256"]
    assert all(item["manual_review_status"] == "pending" for item in report["patients"])
    assert all(len(item["samples"]) == 2 for item in report["patients"])
    assert all(sample["source_assets"][kind]["matches_saved_audit"]
               for item in report["patients"] for sample in item["samples"]
               for kind in ("image", "mask"))
    report_path = args.output_dir / "training_reference_review.json"
    assert json.loads(report_path.read_bytes()) == report
    assert len(list((args.output_dir / "patient_previews").glob("*.png"))) == 20
    assert snapshot(args.root) == before
    assert (args.cohort.read_bytes(), args.audit.read_bytes()) == binding_bytes


def test_repeat_identical_review_reuses_exact_report_and_preview_bytes(environment):
    args, _, _, reads = environment
    first = CLI.review(args)
    before = snapshot(args.output_dir)
    second = CLI.review(args)
    assert second == first
    assert snapshot(args.output_dir) == before
    assert len(reads) == 160
    assert not list(args.output_dir.rglob("*.review.tmp"))


@pytest.mark.parametrize("mutation", ["image_pixels", "mask_pixels", "image_encoding"])
def test_selected_png_change_rejected_without_rebinding_saved_audit(environment, mutation):
    args, lock, audit, reads = environment
    patient = lock["pools"]["train"][0]["patient_id"]
    kind = "mask" if mutation == "mask_pixels" else "image"
    folder = "masks" if kind == "mask" else "images"
    relative = "%s/%s_z007.png" % (folder, patient)
    path = args.root / "data2d" / relative
    with Image.open(path) as image:
        pixels = np.array(image, dtype=np.uint8)
    previous_pixels = pixels.tobytes()
    if mutation == "image_encoding":
        metadata = PngInfo()
        metadata.add_text("fixture", "Same pixels, different saved PNG bytes.")
        Image.fromarray(pixels).save(path, pnginfo=metadata)
    else:
        pixels[0, 0] = int(pixels[0, 0]) ^ 1
        Image.fromarray(pixels).save(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() != audit["assets"][relative]["byte_sha256"]
    if mutation == "image_encoding":
        assert pixels.tobytes() == previous_pixels
    bound = (args.audit.read_bytes(), args.cohort.read_bytes())
    with pytest.raises(ValueError, match="Selected %s content/identity changed" % kind):
        CLI.review(args)
    assert reads and len(reads) <= 2
    assert not args.output_dir.exists()
    assert (args.audit.read_bytes(), args.cohort.read_bytes()) == bound


@pytest.mark.parametrize("filename", ["manifest.csv", "splits.json"])
def test_bound_metadata_drift_rejected_before_any_png_decode(environment, filename):
    args, _, _, reads = environment
    path = args.root / "data2d" / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Original manifest/split changed after Step 5"):
        CLI.review(args)
    assert reads == []
    assert not args.output_dir.exists()


@pytest.mark.parametrize("relative", ["", "data2d/review"])
def test_output_inside_original_project_rejected_before_png_decode(environment, relative):
    args, _, _, reads = environment
    before = snapshot(args.root)
    args.output_dir = args.root / relative
    with pytest.raises(ValueError, match="Save reference previews outside kits_project"):
        CLI.review(args)
    assert reads == [] and snapshot(args.root) == before


@pytest.mark.parametrize("held_out_pool", ["validation", "test", "prediction"])
def test_training_asset_symlink_to_held_out_png_rejected_before_any_decode(environment, held_out_pool):
    args, lock, _, reads = environment
    # The final training patient proves preflight covers the whole plan before decoding.
    training = lock["pools"]["train"][-1]["patient_id"]
    held_out = lock["pools"][held_out_pool][0]["patient_id"]
    images = args.root / "data2d" / "images"
    source = images / (training + "_z002.png")
    target = images / (held_out + "_z002.png")
    source.unlink()
    source.symlink_to(target)
    target_bytes = target.read_bytes()
    with pytest.raises(ValueError, match="must not be symlinks"):
        CLI.review(args)
    assert reads == [] and not args.output_dir.exists()
    assert source.is_symlink() and target.read_bytes() == target_bytes


def test_edited_existing_report_rejected_and_all_existing_outputs_preserved(environment):
    args, _, _, reads = environment
    CLI.review(args)
    report_path = args.output_dir / "training_reference_review.json"
    report_path.write_bytes(report_path.read_bytes() + b"\n")
    edited_outputs = snapshot(args.output_dir)
    with pytest.raises(ValueError, match="Existing review evidence differs"):
        CLI.review(args)
    assert snapshot(args.output_dir) == edited_outputs
    assert len(reads) == 160
    assert not list(args.output_dir.rglob("*.review.tmp"))


def row(z, tumour=0, kidney=0):
    return {"z": str(z), "tumor_px": str(tumour), "kidney_px": str(kidney)}


def chosen_z(selections):
    return [int(item[0]["z"]) for item in selections]


def test_selection_largest_tumour_tie_uses_smallest_z_independent_of_row_order():
    rows = [row(91, tumour=8), row(35, tumour=20), row(4, tumour=20), row(11, tumour=5)]
    first = CLI.select_rows(rows)
    assert chosen_z(first) == [4, 35]
    assert first[0][1]["reason"] == "largest_reference_tumour_area"
    assert first[1][1]["requested_axial_quantile"] == 0.75
    assert first[1][1]["requested_original_slice_index"] == 49.0
    assert CLI.select_rows(list(reversed(rows))) == first


def test_selection_without_tumour_uses_maximum_kidney_then_smallest_z():
    rows = [row(40, kidney=90), row(80, kidney=12), row(3, kidney=90), row(10, kidney=60)]
    selections = CLI.select_rows(rows)
    assert chosen_z(selections)[0] == 3
    assert selections[0][1]["reason"] == "largest_reference_kidney_area_no_tumour_rows"
    assert selections[1][1]["reason"] == "opposite_foreground_axial_quartile"


def test_irregular_axial_quartile_uses_actual_z_and_ignores_background_rows():
    rows = [row(0), row(5, kidney=1), row(10, kidney=1), row(40, kidney=1),
            row(100, tumour=50), row(999)]
    selections = CLI.select_rows(rows)
    assert chosen_z(selections) == [100, 10]
    assert selections[1][1]["requested_axial_quantile"] == 0.25
    assert selections[1][1]["requested_original_slice_index"] == 8.75


def test_selection_nearest_quartile_tie_uses_smallest_actual_z():
    selections = CLI.select_rows([row(1, tumour=5), row(5, kidney=1), row(9, kidney=1)])
    assert chosen_z(selections) == [1, 5]
    assert selections[1][1]["requested_original_slice_index"] == 7.0


@pytest.mark.parametrize("rows", [[], [row(1)], [row(1), row(1)]])
def test_selection_rejects_fewer_than_two_distinct_exported_slices(rows):
    with pytest.raises(ValueError, match="at least two distinct exported slices"):
        CLI.select_rows(rows)


def test_selection_single_foreground_uses_distinct_background_context():
    selections = CLI.select_rows([row(2), row(17, tumour=4), row(80)])
    assert chosen_z(selections) == [17, 2]
    assert selections[1][1]["reason"] == "distinct_exported_background_context_fallback"


def test_selection_all_background_reports_fallback_without_claiming_foreground():
    selections = CLI.select_rows([row(1), row(10), row(100)])
    assert chosen_z(selections) == [1, 10]
    assert selections[1][1]["requested_original_slice_index"] == 55.0
    assert selections[0][1]["reason"] == "no_reference_foreground_available_background_fallback"
    assert selections[1][1]["reason"] == "distinct_exported_background_context_fallback"
