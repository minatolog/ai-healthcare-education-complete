"""All arrays and models in this suite are synthetic/untrained engineering fixtures."""
from dataclasses import replace
import json
from pathlib import Path
import cv2
import nibabel as nib
import numpy as np
from PIL import Image
import pytest
import torch
from kidney_segmentation.data import (AUDIT_FLAGS, FullSliceDataset, file_hash, load_patient, manifest_hash, read_array,
                                      synthetic_patients, validate_real_training, volume_fingerprint)
from kidney_segmentation.engine import (PatientBalancedSampler, evaluate_dataset, load_checkpoint,
                                       predict_volume, save_checkpoint, train_model)
from kidney_segmentation.metrics import confusion_matrix, metrics_from_confusion, patient_report, summarize_patients
from kidney_segmentation.model import KidneyUNet
from kidney_segmentation.overlays import save_overlays
from kidney_segmentation.preprocessing import PreprocessConfig, preprocess_mask, preprocess_slice
from kidney_segmentation.__main__ import write_prediction

torch.set_num_threads(2)
CONFIG = PreprocessConfig(size=64)


def fixture_lineage(patients):
    lineage = {split: {p.patient_id: p.source_patient_id for p in patients if p.split == split}
               for split in ("train", "validation", "test", "prediction")}
    lineage["canonical_volume_fingerprints"] = [volume_fingerprint(p.volume) for p in patients if p.split != "prediction"]
    return lineage


@pytest.fixture(scope="module")
def trained_fixture():
    patients = synthetic_patients(n_slices=2)
    events = []
    training = FullSliceDataset([p for p in patients if p.split == "train"], config=CONFIG)
    validation = FullSliceDataset([p for p in patients if p.split == "validation"], config=CONFIG)
    model, metadata = train_model(training, validation, epochs=2, batch_size=4, pretrained=False,
                                  lineage=fixture_lineage(patients), data_mode="synthetic_demo", progress=events.append)
    return model, metadata, patients, events


def test_preprocessing_matches_verified_legacy_sequence_and_edges():
    volume = np.stack([np.linspace(-100, 350, 120).reshape(10, 12) + offset for offset in (0, 10, 20)]).astype(np.float32)
    processed, indices = preprocess_slice(volume, 0, CONFIG)
    expected = (np.clip(volume, -79, 304) + 79) / 383
    expected = (expected * 255).astype(np.uint8)
    expected = np.stack([cv2.resize(expected[index], (64, 64), interpolation=cv2.INTER_LINEAR) for index in (0, 0, 1)])
    expected = (expected.astype(np.float32) / 255 - .456) / .224
    assert indices == (0, 0, 1)
    np.testing.assert_array_equal(processed.numpy(), expected)
    assert torch.equal(processed[0], processed[1])
    assert preprocess_slice(volume, 2, CONFIG)[1] == (1, 2, 2)
    assert preprocess_slice(volume[:1], 0, CONFIG)[1] == (0, 0, 0)


def test_masks_nearest_labels_and_no_image_label_dependence():
    patient = synthetic_patients()[0]
    first = FullSliceDataset([patient], config=CONFIG)
    second = FullSliceDataset([replace(patient, mask=np.zeros_like(patient.mask))], config=CONFIG)
    assert torch.equal(first[1]["image"], second[1]["image"])
    assert set(preprocess_mask(patient.mask[0], CONFIG).unique().tolist()) == {0, 1, 2}
    assert first[1]["patient_id"] == patient.patient_id
    assert first[1]["context_indices"].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="labels"):
        preprocess_mask(np.array([[3]], dtype=np.uint8), CONFIG)


def test_pixel_metrics_known_values_and_undefined_classes():
    matrix = np.array([[8, 1, 0], [2, 3, 1], [0, 1, 4]])
    metrics = metrics_from_confusion(matrix)
    assert metrics["tumour"]["dice"] == pytest.approx(8 / 10)
    assert metrics["tumour"]["iou"] == pytest.approx(4 / 6)
    assert metrics["tumour"]["precision"] == pytest.approx(4 / 5)
    assert metrics["tumour"]["recall"] == pytest.approx(4 / 5)
    assert metrics["tumour"]["f1"] == metrics["tumour"]["dice"]
    absent = metrics_from_confusion(np.diag([10, 0, 0]))["tumour"]
    assert all(absent[name] is None for name in ("dice", "iou", "precision", "recall", "f1"))
    false_positive = metrics_from_confusion(np.array([[9, 0, 1], [0, 0, 0], [0, 0, 0]]))["tumour"]
    assert false_positive["dice"] == 0 and false_positive["precision"] == 0
    assert false_positive["recall"] is None
    no_prediction = metrics_from_confusion(np.array([[9, 0, 0], [0, 0, 0], [1, 0, 0]]))["tumour"]
    assert no_prediction["precision"] is None and no_prediction["recall"] == 0
    # Strict JSON must remain serializable; no NaN substitutes for undefined metrics.
    json.dumps(metrics_from_confusion(np.diag([10, 0, 0])), allow_nan=False)


def test_patient_mean_is_equal_weight_and_excludes_null():
    reports = [patient_report("large", np.diag([100, 1000, 0])),
               patient_report("small", np.array([[1, 0, 0], [10, 0, 0], [0, 0, 0]]))]
    summary = summarize_patients(reports)
    assert summary["patient_mean"]["kidney"]["dice"] == .5
    assert summary["patient_mean"]["tumour"]["dice"] is None
    assert summary["contributing_patient_counts"]["tumour"]["dice"] == 0
    assert confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 2])).trace() == 3


def test_forward_frozen_encoder_eval_and_decoder_training(trained_fixture):
    model, metadata, patients, events = trained_fixture
    model.train()
    assert not model.network.encoder.training and model.network.decoder.training
    assert all(not parameter.requires_grad for parameter in model.network.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.network.decoder.parameters())
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 3, 64, 64)
    torch.manual_seed(42)
    initial = KidneyUNet(pretrained=False)
    for key, value in model.network.encoder.state_dict().items():
        assert torch.equal(value.cpu(), initial.network.encoder.state_dict()[key]), key
    assert any(not torch.equal(value.cpu(), initial.network.decoder.state_dict()[key])
               for key, value in model.network.decoder.state_dict().items())
    assert len(events) == 2 and events[-1]["progress"] == 1
    assert metadata["best_epoch"] in (1, 2)
    assert metadata["data_mode"] == "synthetic_demo" and not metadata["pretrained_encoder"]
    assert metadata["clinical_performance_evidence"] is False
    assert np.isfinite([event["train_loss"] for event in events]).all()


def test_patient_balanced_sampler_uses_only_patient_membership():
    sampler = PatientBalancedSampler([2, 12], seed=42)
    indices = list(sampler)
    assert sum(index < 2 for index in indices) == 6
    assert sum(index >= 2 for index in indices) == 6
    assert indices == list(PatientBalancedSampler([2, 12], seed=42))


def test_evaluation_rejects_relabeled_training_and_noncanonical_test(trained_fixture):
    model, metadata, patients, _ = trained_fixture
    training = next(p for p in patients if p.split == "train")
    with pytest.raises(ValueError, match="Training patients"):
        evaluate_dataset(model, FullSliceDataset([replace(training, split="test")], config=CONFIG), expected_split="test")
    tests = [p for p in patients if p.split == "test"]
    with pytest.raises(ValueError, match="membership"):
        evaluate_dataset(model, FullSliceDataset(tests[:1], config=CONFIG), expected_split="test",
                         expected_patient_ids=set(metadata["lineage"]["test"]))
    report = evaluate_dataset(model, FullSliceDataset(tests, config=CONFIG), expected_split="test",
                              expected_patient_ids=set(metadata["lineage"]["test"]))
    assert report["patient_count"] == 10 and report["slice_count"] == 20
    assert all(sum(map(sum, row["confusion_matrix"])) == 2 * 64 * 64 for row in report["patients"])


def test_checkpoint_save_load_logits_and_new_inference_parity(trained_fixture, tmp_path):
    model, metadata, patients, _ = trained_fixture
    model.eval()
    save_checkpoint(tmp_path / "fixture.pt", model, metadata)
    loaded, config, restored = load_checkpoint(tmp_path / "fixture.pt")
    assert config == CONFIG and restored == metadata
    image = preprocess_slice(patients[-1].volume, 0, config)[0][None]
    with torch.inference_mode():
        torch.testing.assert_close(model(image), loaded(image), rtol=0, atol=0)
    first = predict_volume(model, patients[-1].volume, patients[-1].patient_id, patients[-1].source_patient_id, config)
    second = predict_volume(loaded, patients[-1].volume, patients[-1].patient_id, patients[-1].source_patient_id, config)
    np.testing.assert_array_equal(first["predicted_mask"], second["predicted_mask"])
    assert second["predicted_mask"].shape == (2, 64, 64) and second["slice_indices"] == [0, 1]


def test_new_prediction_rejects_all_known_ids_sources_and_duplicate_content(trained_fixture):
    model, _, patients, _ = trained_fixture
    for split in ("train", "validation", "test"):
        known = next(p for p in patients if p.split == split)
        with pytest.raises(ValueError, match="outside all"):
            predict_volume(model, known.volume, known.patient_id, known.source_patient_id, CONFIG)
        with pytest.raises(ValueError, match="duplicates"):
            predict_volume(model, known.volume, "new-id", "new-source", CONFIG)


def test_prediction_and_reference_overlay_assets_are_separate(tmp_path):
    patient = synthetic_patients(n_slices=2)[0]
    prediction = np.zeros((2, 64, 64), dtype=np.uint8)
    assets = save_overlays(tmp_path, patient.volume, prediction, patient.patient_id, CONFIG, patient.mask)
    row = assets["assets"][0]
    assert row["predicted_overlay"] != row["ground_truth_overlay"]
    assert row["predicted_overlay_type"] == "model_segmentation"
    assert row["ground_truth_overlay_type"] == "reference_annotation"
    predicted = np.array(Image.open(tmp_path / row["predicted_overlay"]))
    reference = np.array(Image.open(tmp_path / row["ground_truth_overlay"]))
    assert predicted.shape == reference.shape == (64, 64, 3)
    assert not np.array_equal(predicted, reference)


@pytest.fixture
def audited_fixture_files(tmp_path):
    # Fake audited-real metadata tests gates only; never launches real/pretrained training.
    patients = synthetic_patients(n_slices=1)
    rows = []
    assets = {}
    for patient in patients:
        image = patient.patient_id + "_image.npy"
        np.save(tmp_path / image, patient.volume)
        assets[image] = file_hash(tmp_path / image)
        row = {"patient_id": patient.patient_id, "source_patient_id": patient.source_patient_id,
               "split": patient.split, "image": image, "slice_axis": 0}
        if patient.mask is not None:
            mask = patient.patient_id + "_mask.npy"
            np.save(tmp_path / mask, patient.mask)
            assets[mask] = file_hash(tmp_path / mask)
            row["mask"] = mask
        rows.append(row)
    manifest_path = tmp_path / "manifest.json"
    manifest = {"schema_version": 1, "task": "kidney_tumour_segmentation", "units": "HU", "data_mode": "real",
                "full_volume_coverage": True, "patients": rows}
    manifest_path.write_text(json.dumps(manifest))
    audit = {"status": "passed", "manifest_sha256": manifest_hash(manifest_path),
             "verified_patient_ids": [p.patient_id for p in patients], "asset_sha256": assets,
             **{flag: True for flag in AUDIT_FLAGS}}
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    return manifest_path, audit_path, manifest, audit


def test_real_training_audit_blocks_missing_incomplete_or_changed_files(audited_fixture_files):
    manifest_path, audit_path, manifest, audit = audited_fixture_files
    verified, root = validate_real_training(manifest_path, audit_path)
    assert len(verified["patients"]) == 41
    audit["all_axial_slices_available"] = False
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_real_training(manifest_path, audit_path)
    audit["all_axial_slices_available"] = True
    audit_path.write_text(json.dumps(audit))
    image_path = root / manifest["patients"][0]["image"]
    image = np.load(image_path)
    image[0, 0, 0] += 1
    np.save(image_path, image)
    with pytest.raises(ValueError, match="checksums"):
        validate_real_training(manifest_path, audit_path)
    image_path.unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_real_training(manifest_path, audit_path)


def test_real_training_blocks_duplicate_source_and_prediction_pool_overlap(audited_fixture_files):
    manifest_path, audit_path, manifest, _ = audited_fixture_files
    manifest["patients"][-1]["source_patient_id"] = manifest["patients"][0]["source_patient_id"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unique"):
        validate_real_training(manifest_path, audit_path)


def test_real_training_blocks_without_audit_even_if_arrays_are_available():
    patients = synthetic_patients()
    train = FullSliceDataset([p for p in patients if p.split == "train"], config=CONFIG)
    val = FullSliceDataset([p for p in patients if p.split == "validation"], config=CONFIG)
    with pytest.raises(ValueError, match="audit lineage"):
        train_model(train, val, pretrained=False, data_mode="real")


def test_duplicate_patient_ids_cannot_collapse_evaluation_cases():
    patients = synthetic_patients(n_slices=1)
    duplicate = replace(patients[1], patient_id=patients[0].patient_id)
    with pytest.raises(ValueError, match="unique patient"):
        FullSliceDataset([patients[0], duplicate], config=CONFIG)


def test_saved_preprocessing_is_used_by_default_and_mismatch_rejected(trained_fixture):
    model, _, patients, _ = trained_fixture
    new = patients[-1]
    assert predict_volume(model, new.volume, new.patient_id, new.source_patient_id)["predicted_mask"].shape == (2, 64, 64)
    with pytest.raises(ValueError, match="Prediction preprocessing"):
        predict_volume(model, new.volume, new.patient_id, new.source_patient_id, PreprocessConfig(size=256))
    tests = [p for p in patients if p.split == "test"]
    with pytest.raises(ValueError, match="Evaluation preprocessing"):
        evaluate_dataset(model, FullSliceDataset(tests, config=PreprocessConfig(size=256)), expected_split="test")


def test_missing_test_or_content_provenance_fails_closed(trained_fixture, monkeypatch):
    model, metadata, patients, _ = trained_fixture
    lineage = dict(metadata["lineage"])
    lineage.pop("test")
    monkeypatch.setattr(model, "lineage", lineage)
    tests = [p for p in patients if p.split == "test"]
    with pytest.raises(ValueError, match="Canonical evaluation lineage"):
        evaluate_dataset(model, FullSliceDataset(tests, config=CONFIG), expected_split="test")
    with pytest.raises(ValueError, match="Complete canonical"):
        predict_volume(model, patients[-1].volume, patients[-1].patient_id, patients[-1].source_patient_id, CONFIG)
    lineage = dict(metadata["lineage"])
    lineage.pop("canonical_volume_fingerprints")
    monkeypatch.setattr(model, "lineage", lineage)
    with pytest.raises(ValueError, match="content fingerprints"):
        predict_volume(model, patients[-1].volume, patients[-1].patient_id, patients[-1].source_patient_id, CONFIG)


def test_checkpoint_rejects_stale_lineage_and_preprocessing_metadata(trained_fixture, tmp_path):
    model, metadata, _, _ = trained_fixture
    stale = {**metadata, "lineage": {}}
    with pytest.raises(ValueError, match="metadata lineage"):
        save_checkpoint(tmp_path / "stale.pt", model, stale)
    stale = {**metadata, "preprocessing": PreprocessConfig(size=256).to_dict()}
    with pytest.raises(ValueError, match="preprocessing metadata"):
        save_checkpoint(tmp_path / "stale.pt", model, stale)
    save_checkpoint(tmp_path / "valid.pt", model, metadata)
    restored, _, _ = load_checkpoint(tmp_path / "valid.pt")
    assert restored.checkpoint_sha256 == model.checkpoint_sha256 == file_hash(tmp_path / "valid.pt")


def test_nifti_geometry_alignment_and_explicit_slice_axis(tmp_path):
    image = np.zeros((3, 4, 5), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[:, 1:3, 1:3] = 1
    affine = np.diag([2.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(image, affine), tmp_path / "image.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), tmp_path / "mask.nii.gz")
    row = {"patient_id": "geometry-fixture", "source_patient_id": "geometry-source", "split": "train",
           "image": "image.nii.gz", "mask": "mask.nii.gz", "slice_axis": 0}
    assert load_patient(row, tmp_path).volume.shape == (3, 4, 5)
    assert read_array(tmp_path / "image.nii.gz", slice_axis=2)[0].shape == (5, 3, 4)
    shifted = affine.copy()
    shifted[0, 3] = 2
    nib.save(nib.Nifti1Image(mask, shifted), tmp_path / "mask.nii.gz")
    with pytest.raises(ValueError, match="affines"):
        load_patient(row, tmp_path)
    with pytest.raises(ValueError, match="slice_axis"):
        read_array(tmp_path / "image.nii.gz", slice_axis=True)


def test_nonfinite_preprocessing_configuration_is_rejected():
    for kwargs in ({"hu_min": float("nan")}, {"hu_max": float("inf")}, {"mean": float("nan")}, {"std": float("inf")}, {"size": 64.0}):
        with pytest.raises(ValueError):
            PreprocessConfig(**kwargs)


def test_cli_prediction_records_exact_checkpoint_identity(trained_fixture, tmp_path):
    model, metadata, patients, _ = trained_fixture
    save_checkpoint(tmp_path / "fixture.pt", model, metadata)
    loaded, config, metadata = load_checkpoint(tmp_path / "fixture.pt")
    write_prediction(tmp_path / "prediction", patients[-1], loaded, config, metadata)
    result = json.loads((tmp_path / "prediction" / "prediction.json").read_text())
    assert result["model_version"] == result["checkpoint_sha256"] == file_hash(tmp_path / "fixture.pt")
    assert result["prediction_context"] == "new_patient"
