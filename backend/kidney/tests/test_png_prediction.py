"""CT-only frozen-prediction checks using declared synthetic checkpoints."""
from collections import Counter
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from kidney_segmentation import png_prediction as prediction
from kidney_segmentation import png_training as training
from test_png_final_test import completed_run, digest, save_freeze
from test_png_training import fixture_data, tiny_builder


def inspect(run, **kwargs):
    return prediction.inspect_prediction_inputs(run["root"], run["cohort_path"], run["audit_path"],
                                                 run["run_dir"], run["freeze_path"], allow_fixture=True, **kwargs)


def prepare(run):
    save_freeze(run)
    inspected = inspect(run)
    directory = run["tmp"] / "prediction_output"
    directory.mkdir()
    return inspected, directory


def prediction_rows(run):
    ids = [record["patient_id"] for record in run["cohort"]["pools"]["prediction"]]
    return ids, [row for row in run["rows"] if row["case"] in ids]


def fail(*args, **kwargs):
    pytest.fail("This operation must not initialize a model or decode a PNG")


def test_preflight_is_metadata_only_and_masks_need_not_exist(completed_run, monkeypatch):
    save_freeze(completed_run)
    for path in (completed_run["data"] / "masks").iterdir():
        path.unlink()
    (completed_run["data"] / "masks").rmdir()
    monkeypatch.setattr(Image, "open", fail)
    monkeypatch.setattr(training, "KidneyUNet", fail)
    monkeypatch.setattr(training, "build_verified_model", fail)
    original_read = Path.read_bytes

    def read(path):
        assert path.suffix != ".png", "Preflight read CT or reference pixels"
        assert "masks" not in path.parts
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    inspected = inspect(completed_run)
    dataset = prediction.build_prediction_dataset(inspected)
    ids, rows = prediction_rows(completed_run)
    assert list(dataset.patient_ids) == ids
    assert len(dataset) == len(rows) == completed_run["slices"]["prediction"]
    assert inspected["prediction_inputs"]["slice_count"] == len(rows)
    assert all(set(row) == {"case", "z", "image", "n_slices"} for row in dataset.rows)
    assert inspected["prediction_inputs"]["mask_pixels_loaded"] == 0


def test_csv_label_metadata_cannot_change_rows_or_fixed_examples(completed_run):
    inspected, _ = prepare(completed_run)
    dataset = prediction.build_prediction_dataset(inspected)
    original = completed_run["data"].joinpath("manifest.csv").read_bytes()
    reader = csv.DictReader(io.StringIO(original.decode()))
    changed = io.StringIO()
    writer = csv.DictWriter(changed, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        for field in ("mask", "kidney_px", "tumor_px", "has_kidney", "has_tumor"):
            row[field] = "invalid label metadata ignored by CT loader"
        writer.writerow(row)
    assert prediction._ct_manifest_rows(changed.getvalue().encode(), dataset.patient_ids) == dataset.rows
    # A CSV with only the four CT fields has the same identities as well.
    ct_only = io.StringIO()
    writer = csv.DictWriter(ct_only, fieldnames=["case", "z", "image", "n_slices"])
    writer.writeheader()
    writer.writerows(dataset.rows)
    assert prediction._ct_manifest_rows(ct_only.getvalue().encode(), dataset.patient_ids) == dataset.rows
    for patient in dataset.patient_ids:
        rows = [row for row in dataset.rows if row["case"] == patient]
        assert dataset.preview_sample_ids[patient] == [dataset._sample_id(rows[0]), dataset._sample_id(rows[len(rows) // 2])]


def test_normalization_uses_ct_only_without_resizing(completed_run):
    inspected, _ = prepare(completed_run)
    dataset = prediction.build_prediction_dataset(inspected)
    item = dataset[0]
    assert item["input"].shape == (1, 256, 256) and item["input"].dtype == torch.float32
    assert item["ct_pixels"].dtype == torch.uint8 and item["ct_pixels"].shape == (256, 256)
    expected = (item["ct_pixels"].numpy().astype(np.float32) / 255 - 0.456) / 0.224
    np.testing.assert_array_equal(item["input"][0].numpy(), expected)
    assert item["input"].min().item() == pytest.approx(-0.456 / 0.224)
    assert item["input"].max().item() == pytest.approx((1 - 0.456) / 0.224)
    assert not {"target", "mask", "mask_label_counts", "reference_metadata_matches"}.intersection(item)


def test_all_fixed_ct_images_once_outputs_hashes_no_reference_or_optimizer(completed_run, monkeypatch):
    inspected, directory = prepare(completed_run)
    ids, rows = prediction_rows(completed_run)
    # Remove every reference mask, including unrelated train/test references.
    for path in (completed_run["data"] / "masks").iterdir():
        path.unlink()
    original_read = Path.read_bytes
    original_open = Image.open
    reads, decodes, built, restored = Counter(), Counter(), [], []

    def guarded_read(path):
        assert "masks" not in path.parts, "Prediction tried to read a reference mask"
        if path.suffix == ".png" and completed_run["data"] in path.parents:
            assert path.parent.name == "images" and path.stem.split("_z")[0] in ids
            reads[str(path)] += 1
        return original_read(path)

    def guarded_open(content, *args, **kwargs):
        image = original_open(content, *args, **kwargs)
        decodes[digest(content.getvalue())] += 1
        return image

    def builder(**kwargs):
        model, proof = tiny_builder(**kwargs)
        built.append(model)
        load, forward = model.load_state_dict, model.forward

        def restore(state, *args, **options):
            restored.append(state)
            return load(state, *args, **options)

        def infer(inputs):
            assert not torch.is_grad_enabled() and not model.training
            assert all(not parameter.requires_grad for parameter in model.parameters())
            assert all(torch.equal(model.state_dict()[key], value) for key, value in completed_run["state"].items())
            logits = forward(inputs)
            # Predict both foreground classes without consulting any reference.
            logits[:, 1, :128, :] = 10
            logits[:, 2, 128:, :] = 10
            return logits

        model.load_state_dict, model.forward = restore, infer
        return model, proof

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    monkeypatch.setattr(Image, "open", guarded_open)
    monkeypatch.setattr(torch.optim, "Adam", fail)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", fail)
    monkeypatch.setattr(torch.optim.Optimizer, "step", fail)

    def progress(event):
        # CLI-owned evidence may appear after the initial empty-output check.
        (directory / "prediction_summary.json").write_text(json.dumps(event))
        with (directory / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")

    report = prediction.predict_frozen_pool(inspected, directory, batch_size=3, progress=progress, model_builder=builder)
    monkeypatch.setattr(Image, "open", original_open)
    expected_reads = {str(completed_run["data"] / row["image"]): 1 for row in rows}
    assert reads == expected_reads and sum(decodes.values()) == len(rows)
    assert len(built) == len(restored) == 1
    assert all(torch.equal(restored[0][key], tensor) for key, tensor in completed_run["state"].items())
    assert json.loads((directory / "prediction_report.json").read_text()) == report
    assert report["patient_ids"] == ids and report["patient_count"] == 5
    assert report["prediction_ct_count"] == report["slice_count"] == len(rows)
    assert report["sample_counts"] == {"%s:z%d" % (row["case"], row["z"]): 1 for row in rows}
    assert report["decoded_ct_images"] == {"train": 0, "validation": 0, "test": 0, "prediction": len(rows)}
    assert report["test_pixels_loaded"] == report["mask_pixels_loaded"] == report["optimizer_steps"] == 0
    assert report["model_state_before_sha256"] == report["model_state_after_sha256"] == training._state_hash(completed_run["state"])
    assert report["model_state_unchanged"] and report["all_rows_seen_once"]
    assert not report["ground_truth_metrics_computed"] and not report["patient_level_diagnosis"]
    assert not report["confidence_estimate_provided"] and not report["full_scan_coverage"]
    assert not {"loss", "dice", "confusion_matrix", "patient_mean"}.intersection(report)
    for patient in report["patients"]:
        assert patient["preview"]["sample_ids"] == inspected["prediction_inputs"]["preview_sample_ids"][patient["patient_id"]]
        assert digest(Path(patient["preview"]["path"]).read_bytes()) == patient["preview"]["sha256"]
        for row in patient["slices"]:
            assert row["predicted_class_counts"] == {"0": 0, "1": 32768, "2": 32768}
            for field in ("predicted_mask", "model_overlay"):
                asset = row[field]
                assert digest(Path(asset["path"]).read_bytes()) == asset["sha256"]
                with Image.open(asset["path"]) as image:
                    assert image.size == (256, 256)
                    assert image.mode == ("L" if field == "predicted_mask" else "RGB")
                    if field == "predicted_mask":
                        assert set(np.unique(np.asarray(image))) == {1, 2}


@pytest.mark.parametrize("field,value", [("batch_size", True), ("batch_size", 0), ("batch_size", 1.5),
                                        ("num_threads", False), ("num_threads", -1), ("num_threads", "2")])
def test_invalid_inference_settings_fail_before_model(completed_run, monkeypatch, field, value):
    inspected, directory = prepare(completed_run)
    monkeypatch.setattr(training, "load_png_checkpoint", fail)
    with pytest.raises(ValueError, match="positive integer"):
        prediction.predict_frozen_pool(inspected, directory, model_builder=tiny_builder, **{field: value})
    assert not list(directory.iterdir())


def test_output_directory_must_be_new_empty_and_outside_protected_inputs(completed_run, monkeypatch):
    inspected, directory = prepare(completed_run)
    marker = directory / "existing_result.txt"
    marker.write_bytes(b"preserve exactly")
    monkeypatch.setattr(training, "load_png_checkpoint", fail)
    with pytest.raises(FileExistsError, match="empty"):
        prediction.predict_frozen_pool(inspected, directory, model_builder=tiny_builder)
    assert marker.read_bytes() == b"preserve exactly" and list(directory.iterdir()) == [marker]
    for destination in (completed_run["root"] / "new_output", completed_run["run_dir"] / "new_output",
                        completed_run["tmp"] / "not_created"):
        with pytest.raises(ValueError, match="protected|empty"):
            prediction.predict_frozen_pool(inspected, destination, model_builder=tiny_builder)
        assert not destination.exists()
    alias = completed_run["tmp"] / "alias_output"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prediction.predict_frozen_pool(inspected, alias, model_builder=tiny_builder)


@pytest.mark.parametrize("changed", ["manifest", "original_split", "cohort", "audit", "freeze", "checkpoint"])
def test_frozen_evidence_drift_rejected_before_model_or_pixels(completed_run, monkeypatch, changed):
    inspected, directory = prepare(completed_run)
    path = {"freeze": completed_run["freeze_path"], "checkpoint": completed_run["checkpoint_path"]}.get(
        changed, inspected["context"]["paths"].get(changed))
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(training, "load_png_checkpoint", fail)
    monkeypatch.setattr(Image, "open", fail)
    with pytest.raises(ValueError):
        prediction.predict_frozen_pool(inspected, directory, model_builder=tiny_builder)
    assert not list(directory.iterdir())


@pytest.mark.parametrize("drift", ["byte", "pixel", "identity", "path"])
def test_ct_source_hash_and_identity_checks(completed_run, monkeypatch, drift):
    inspected, _ = prepare(completed_run)
    dataset = prediction.build_prediction_dataset(inspected)
    row = dataset.rows[0]
    facts = dataset.context["audit"]["assets"][row["image"]]
    if drift == "byte":
        path = dataset.data_root / row["image"]
        path.write_bytes(path.read_bytes() + b"drift")
        monkeypatch.setattr(Image, "open", fail)
    elif drift == "pixel":
        facts["decoded_pixel_sha256"] = "0" * 64
    elif drift == "identity":
        facts["patient_id"] = "case_99999"
        monkeypatch.setattr(Image, "open", fail)
    else:
        row["image"] = row["image"].replace("images/", "masks/")
        monkeypatch.setattr(Image, "open", fail)
    with pytest.raises(ValueError, match="bytes|hash|identity"):
        dataset[0]


@pytest.mark.parametrize("mutation", ["tensor", "provenance", "gradients"])
def test_model_mutation_cannot_produce_completed_report(completed_run, mutation):
    inspected, directory = prepare(completed_run)

    def builder(**kwargs):
        model, proof = tiny_builder(**kwargs)
        forward = model.forward

        def infer(inputs):
            result = forward(inputs)
            if mutation == "tensor":
                model.network.segmentation_head.bias.add_(0.01)
            elif mutation == "provenance":
                model.pretrained_proof["pretrained_encoder"] = True
            else:
                model.network.segmentation_head.weight.requires_grad_(True)
            return result

        model.forward = infer
        return model, proof

    with pytest.raises(ValueError, match="changed|provenance"):
        prediction.predict_frozen_pool(inspected, directory, model_builder=builder)
    assert not (directory / "prediction_report.json").exists()


@pytest.mark.parametrize("output", ["shape", "integer", "nonfinite", "object"])
def test_invalid_logits_cannot_produce_completed_report(completed_run, output):
    inspected, directory = prepare(completed_run)

    def builder(**kwargs):
        model, proof = tiny_builder(**kwargs)
        forward = model.forward

        def infer(inputs):
            logits = forward(inputs)
            if output == "shape":
                return logits[:, :2]
            if output == "integer":
                return logits.to(torch.int64)
            if output == "nonfinite":
                logits[0, 0, 0, 0] = torch.nan
                return logits
            return {"logits": logits}

        model.forward = infer
        return model, proof

    with pytest.raises(ValueError, match="logits|floating"):
        prediction.predict_frozen_pool(inspected, directory, model_builder=builder)
    assert not (directory / "prediction_report.json").exists()


def test_completion_callback_is_inside_protected_interval(completed_run):
    inspected, directory = prepare(completed_run)

    def progress(event):
        if event["event"] == "completed":
            path = completed_run["freeze_path"]
            path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="freeze record changed"):
        prediction.predict_frozen_pool(inspected, directory, model_builder=tiny_builder, progress=progress)
    assert not (directory / "prediction_report.json").exists()


def test_fixture_injection_requires_explicit_gate(completed_run):
    inspected, directory = prepare(completed_run)
    with pytest.raises(ValueError, match="builder"):
        prediction.predict_frozen_pool(inspected, directory)
    inspected["allow_fixture"] = False
    with pytest.raises(ValueError, match="explicitly allowed"):
        prediction.predict_frozen_pool(inspected, directory, model_builder=tiny_builder)
