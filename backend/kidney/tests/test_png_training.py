"""Synthetic PNG/model engineering checks; no real CT or pretrained download."""
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from kidney_segmentation.cohort import canonical_fingerprint, make_cohort
from kidney_segmentation.engine import load_checkpoint as load_legacy_checkpoint
from kidney_segmentation.png_audit import audit_png_dataset
from kidney_segmentation import png_dataset
from kidney_segmentation import png_training as training


class TinyFixtureModel(nn.Module):
    """One-channel test network with a frozen Conv/BN encoder and small head."""
    architecture = "synthetic_tiny_segmentation"
    input_channels = 1
    input_mode = "single_2d_slice"
    pretrained = False

    def __init__(self):
        super().__init__()
        self.network = nn.Module()
        self.network.encoder = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
        self.network.decoder = nn.Sequential(nn.Conv2d(2, 4, 1), nn.ReLU())
        self.network.segmentation_head = nn.Conv2d(4, 3, 1)
        with torch.no_grad():
            self.network.encoder[0].weight.fill_(0.25)
            self.network.encoder[0].bias.fill_(0.1)
            self.network.decoder[0].weight.fill_(0.1)
            self.network.decoder[0].bias.fill_(0.5)
        for parameter in self.network.encoder.parameters():
            parameter.requires_grad_(False)
        self.network.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.network.encoder.eval()
        return self

    def forward(self, inputs):
        return self.network.segmentation_head(self.network.decoder(self.network.encoder(inputs)))


def tiny_builder(input_channels=1, cache_root=None):
    assert input_channels == 1
    return TinyFixtureModel(), {"initialization_kind": "synthetic_fixture", "pretrained_encoder": False,
                                "input_channels": 1, "encoder_frozen": True}


@pytest.fixture
def fixture_data(tmp_path):
    root = tmp_path / "kits_project"
    data = root / "data2d"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir()
    rows, cases = [], ["case_%05d" % index for index in range(45)]
    y, x = np.indices((256, 256))
    for index, case in enumerate(cases):
        # Case 0 is forced into train; its extra slice makes 41/2 a singleton
        # final batch, whose existence must not make an exported slice disappear.
        count = 3 if index == 0 else 2
        for z in range(count):
            ct = ((x + 3 * y + index * 5 + z) % 256).astype(np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if index % 5:
                side = 2 + index % 6
                mask[90:90 + side, 100:100 + side] = 1
                if index % 2:
                    mask[90:92, 100:102] = 2
            image, reference = "images/%s_z%03d.png" % (case, z), "masks/%s_z%03d.png" % (case, z)
            Image.fromarray(ct).save(data / image)
            Image.fromarray(mask).save(data / reference)
            rows.append({"case": case, "z": z, "image": image, "mask": reference, "n_slices": count,
                         "kidney_px": int((mask == 1).sum()), "tumor_px": int((mask == 2).sum()),
                         "has_kidney": int((mask == 1).any()), "has_tumor": int((mask == 2).any())})
    with (data / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (data / "splits.json").write_text(json.dumps({"train": cases, "val": [], "test": []}))
    audit = audit_png_dataset(root)
    assert audit["status"] == "passed" and not audit["duplicate_patient_components"]
    audit["data_origin"] = "synthetic_fixture"
    audit["audit_fingerprint_sha256"] = canonical_fingerprint(audit, "audit_fingerprint_sha256")
    cohort = make_cohort(audit)
    audit_path, cohort_path = tmp_path / "audit.json", tmp_path / "cohort.json"
    audit_path.write_text(json.dumps(audit))
    cohort_path.write_text(json.dumps(cohort))
    return {"root": root, "data": data, "rows": rows, "audit": audit, "cohort": cohort,
            "audit_path": audit_path, "cohort_path": cohort_path, "tmp": tmp_path}


def context(data):
    return training.load_training_context(data["root"], data["cohort_path"], data["audit_path"])


def run(data, **kwargs):
    return training.train_png(data["root"], data["cohort_path"], data["audit_path"],
                              data_mode="synthetic_fixture", model_builder=tiny_builder, **kwargs)


def test_metadata_load_and_role_selection_never_decode_heldout(fixture_data, monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("Metadata operations must not decode PNGs.")
    monkeypatch.setattr(png_dataset, "_read_png", prohibited)
    loaded = context(fixture_data)
    assert loaded["config"].input_channels == 1
    train = training.AuditedPNGDataset(loaded, "train")
    validation = training.AuditedPNGDataset(loaded, "validation")
    assert len(train) == 41 and len(validation) == 20
    assert set(train.patient_ids).isdisjoint(validation.patient_ids)
    for role in ("test", "prediction"):
        with pytest.raises(ValueError, match="never constructs"):
            training.AuditedPNGDataset(loaded, role)


@pytest.mark.parametrize("changed", ["manifest", "original_split", "cohort", "audit"])
def test_stale_bound_input_is_rejected_before_pixels(fixture_data, monkeypatch, changed):
    loaded = context(fixture_data)
    path = loaded["paths"][changed]
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(png_dataset, "_read_png", lambda *args: pytest.fail("Pixels read after binding drift"))
    with pytest.raises(ValueError, match="changed during training"):
        training.AuditedPNGDataset(loaded, "train")
    if changed in ("manifest", "original_split"):
        with pytest.raises(ValueError, match="differs from the locked"):
            context(fixture_data)


def test_audited_item_full_uint8_range_target_separation_and_pixel_hash(fixture_data):
    dataset = training.AuditedPNGDataset(context(fixture_data), "train")
    item = dataset[0]
    assert item["input"].shape == (1, 256, 256) and item["input"].dtype == torch.float32
    assert item["target"].shape == (256, 256) and item["target"].dtype == torch.int64
    assert item["input"].min().item() == pytest.approx(-0.456 / 0.224)
    assert item["input"].max().item() == pytest.approx((1 - 0.456) / 0.224)
    assert item["pool"] == "train" and item["contains_model_prediction"] is False
    assert item["source_paths"]["image"] != item["source_paths"]["mask"]
    row = dataset.rows[0]
    dataset.context["audit"]["assets"][row["image"]]["decoded_pixel_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="decoded-pixel hashes"):
        dataset[0]


def test_changed_png_bytes_fail_before_decoder(fixture_data, monkeypatch):
    dataset = training.AuditedPNGDataset(context(fixture_data), "train")
    source = dataset.data_root / dataset.rows[0]["image"]
    source.write_bytes(source.read_bytes() + b"changed")
    monkeypatch.setattr(png_dataset, "_read_png", lambda *args: pytest.fail("Changed bytes were decoded"))
    with pytest.raises(ValueError, match="bytes changed"):
        dataset[0]


def test_training_symlink_to_heldout_is_rejected_before_any_decode(fixture_data, monkeypatch):
    loaded = context(fixture_data)
    dataset = training.AuditedPNGDataset(loaded, "train")
    row = dataset.rows[0]
    target_case = loaded["cohort"]["pools"]["test"][0]["patient_id"]
    source = dataset.data_root / row["image"]
    source.unlink()
    source.symlink_to(dataset.data_root / ("images/%s_z000.png" % target_case))
    monkeypatch.setattr(png_dataset, "_read_png", lambda *args: pytest.fail("Heldout alias was decoded"))
    with pytest.raises(ValueError, match="symlinks"):
        dataset[0]
    with pytest.raises(ValueError, match="symlinks"):
        training.AuditedPNGDataset(loaded, "train")


def test_real_mode_cannot_inject_unpretrained_or_use_fixture_audit(fixture_data):
    with pytest.raises(ValueError, match="only for synthetic"):
        training.train_png(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"],
                           model_builder=tiny_builder)
    with pytest.raises(ValueError, match="real mode rejects"):
        training.train_png(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"])
    with pytest.raises(ValueError, match="explicit test model builder"):
        training.train_png(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"],
                           data_mode="synthetic_fixture")


def test_rehashed_wrong_preprocessing_is_rejected(fixture_data):
    audit = fixture_data["audit"]
    audit["preprocessing"]["mean"] = 0.5
    audit["preprocessing_sha256"] = hashlib.sha256(json.dumps(audit["preprocessing"], sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
    audit["audit_fingerprint_sha256"] = canonical_fingerprint(audit, "audit_fingerprint_sha256")
    fixture_data["audit_path"].write_text(json.dumps(audit))
    fixture_data["cohort_path"].write_text(json.dumps(make_cohort(audit)))
    with pytest.raises(ValueError, match="audit preprocessing differs"):
        context(fixture_data)


def test_default_real_builder_proof_is_checked_before_any_pixels(fixture_data, monkeypatch):
    # These remain synthetic files. Removing the fixture marker exercises the
    # real initialization gate without using an actual pretrained model or CT.
    audit = fixture_data["audit"]
    del audit["data_origin"]
    audit["audit_fingerprint_sha256"] = canonical_fingerprint(audit, "audit_fingerprint_sha256")
    fixture_data["audit_path"].write_text(json.dumps(audit))
    fixture_data["cohort_path"].write_text(json.dumps(make_cohort(audit)))
    monkeypatch.setattr(training, "build_verified_model", tiny_builder)
    monkeypatch.setattr(png_dataset, "_read_png", lambda *args: pytest.fail("Unverified encoder read pixels"))
    with pytest.raises(ValueError, match="pinned offline pretrained encoder proof"):
        training.train_png(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"])


def test_all_training_slices_singleton_batch_frozen_encoder_and_checkpoint_parity(fixture_data, monkeypatch):
    allowed = {r["patient_id"] for role in ("train", "validation") for r in fixture_data["cohort"]["pools"][role]}
    decoded = Counter()
    original = png_dataset._read_png
    def guarded(path, size):
        case = Path(path).stem.split("_z")[0]
        assert case in allowed, "Test/prediction pixels must stay unread"
        decoded[(case, Path(path).parent.name)] += 1
        return original(path, size)
    monkeypatch.setattr(png_dataset, "_read_png", guarded)
    models, initial_states = [], []
    def captured_builder(**kwargs):
        model, proof = tiny_builder(**kwargs)
        models.append(model)
        initial_states.append(copy.deepcopy(model.state_dict()))
        return model, proof
    events = []
    result = training.train_png(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"],
                                data_mode="synthetic_fixture", model_builder=captured_builder, progress=events.append,
                                visual_review_note="Fixture note only; no clinical alignment claim.")
    model, metadata = result["model"], result["metadata"]
    assert metadata["optimizer_steps_completed"] == 21
    assert metadata["history"][0]["train_slices_seen"] == 41
    assert metadata["drop_last"] is False
    for patient in allowed:
        expected = next(r["selected_slice_count"] for r in fixture_data["audit"]["patients"] if r["patient_id"] == patient)
        assert decoded[(patient, "images")] == decoded[(patient, "masks")] == expected
    before, after = initial_states[0], model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before if k.startswith("network.encoder."))
    for prefix in ("network.decoder.", "network.segmentation_head."):
        assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith(prefix))
    assert metadata["data_mode"] == "synthetic_fixture" and metadata["pretrained_proof"]["pretrained_encoder"] is False
    assert metadata["anatomical_alignment_verified"] is False and metadata["test_evaluated"] is False
    assert result["validation_report"]["metric_scope"] == training.METRIC_SCOPE
    assert result["validation_report"]["patient_count"] == 10
    assert all(sum(sum(row) for row in p["confusion_matrix"]) == 2 * 256 * 256
               for p in result["validation_report"]["patients"])
    assert events[0]["event"] == "initial" and events[-1]["event"] == "completed"
    assert any(e.get("phase") == "training" and e.get("batch_index") == 21 for e in events)
    assert all("timestamp_utc" in e and "metric_scope" in e for e in events)
    assert all(len(json.dumps(e)) < 5000 for e in events)
    path = fixture_data["tmp"] / "run" / "best_model.pt"
    evidence = training.save_png_checkpoint(path, model, metadata)
    assert evidence["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    restored, config, restored_metadata = training.load_png_checkpoint(path, model_builder=tiny_builder)
    assert config.input_channels == 1 and restored_metadata == metadata
    inputs = training.AuditedPNGDataset(context(fixture_data), "train")[0]["input"].unsqueeze(0)
    with torch.inference_mode():
        assert torch.equal(model(inputs), restored(inputs))
    saved = path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        training.save_png_checkpoint(path, model, metadata)
    assert path.read_bytes() == saved
    with pytest.raises(ValueError, match="Unsupported Kidney segmentation checkpoint"):
        load_legacy_checkpoint(path)
    with pytest.raises(ValueError, match="test model builder"):
        training.load_png_checkpoint(path)


def test_patient_metrics_absent_classes_and_eval_mode(fixture_data):
    loaded = context(fixture_data)
    validation = training.AuditedPNGDataset(loaded, "validation")
    class BackgroundOnly(nn.Module):
        png_cohort_fingerprint_sha256 = loaded["cohort"]["cohort_fingerprint_sha256"]
        def forward(self, inputs):
            assert not self.training and not torch.is_grad_enabled()
            logits = torch.zeros((len(inputs), 3, 256, 256))
            logits[:, 0] = 1
            return logits
    report = training.evaluate_png_validation(BackgroundOnly(), validation)
    for name in ("background", "kidney", "tumour"):
        values = [p["per_class"][name]["dice"] for p in report["patients"] if p["per_class"][name]["dice"] is not None]
        assert report["contributing_patient_counts"][name]["dice"] == len(values)
        assert report["patient_mean"][name]["dice"] == (pytest.approx(float(np.mean(values))) if values else None)
    absent = [p for p in report["patients"] if p["per_class"]["tumour"]["support_pixels"] == 0]
    assert absent and all(p["per_class"]["tumour"]["dice"] is None for p in absent)
    assert report["clinical_performance_evidence"] is False
    with pytest.raises(ValueError, match="Only a locked"):
        training.evaluate_png_validation(BackgroundOnly(), training.AuditedPNGDataset(loaded, "train"))


def test_best_validation_loss_restores_best_not_last_epoch(fixture_data, monkeypatch):
    actual_evaluation = training.evaluate_png_validation
    model_holder, states, events = [], [], []
    def builder(**kwargs):
        model, proof = tiny_builder(**kwargs)
        model_holder.append(model)
        return model, proof
    def controlled_validation(model, dataset, **kwargs):
        report = actual_evaluation(model, dataset, **kwargs)
        report["loss"] = 1.0 if kwargs["epoch"] == 1 else 2.0
        states.append(copy.deepcopy(model.state_dict()))
        return report
    monkeypatch.setattr(training, "evaluate_png_validation", controlled_validation)
    result = training.train_png(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"],
                                data_mode="synthetic_fixture", model_builder=builder, epochs=2, progress=events.append)
    assert result["metadata"]["best_epoch"] == 1
    assert result["best_validation_report"]["loss"] == 1.0
    assert result["last_validation_report"]["loss"] == 2.0
    assert all(torch.equal(result["model"].state_dict()[k], value) for k, value in states[0].items())
    assert any(not torch.equal(states[0][k], states[1][k]) for k in states[0])


def test_checkpoint_rejects_changed_metadata_and_tensors(fixture_data):
    result = run(fixture_data)
    path = fixture_data["tmp"] / "good.pt"
    training.save_png_checkpoint(path, result["model"], result["metadata"])
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["seed"] += 1
    changed = fixture_data["tmp"] / "metadata_changed.pt"
    torch.save(payload, changed)
    with pytest.raises(ValueError, match="metadata fingerprint is stale"):
        training.load_png_checkpoint(changed, model_builder=tiny_builder)
    payload = torch.load(path, weights_only=True)
    key = next(k for k in payload["state_dict"] if k.startswith("network.segmentation_head."))
    payload["state_dict"][key].reshape(-1)[0] += 1
    changed = fixture_data["tmp"] / "tensor_changed.pt"
    torch.save(payload, changed)
    with pytest.raises(ValueError, match="state fingerprint"):
        training.load_png_checkpoint(changed, model_builder=tiny_builder)
    with pytest.raises(ValueError, match="outside the original data"):
        training.save_png_checkpoint(fixture_data["root"] / "model.pt", result["model"], result["metadata"])


def test_encoder_drift_and_nonfinite_gradients_stop_training(fixture_data):
    def broken_encoder(**kwargs):
        model, proof = tiny_builder(**kwargs)
        model.network.encoder[0].weight.requires_grad_(True)
        return model, proof
    with pytest.raises(ValueError, match="Only decoder"):
        training.train_png(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"],
                           data_mode="synthetic_fixture", model_builder=broken_encoder)
    def nonfinite_gradient(**kwargs):
        model, proof = tiny_builder(**kwargs)
        model.network.segmentation_head.weight.register_hook(lambda gradient: gradient * float("nan"))
        return model, proof
    with pytest.raises(ValueError, match="non-finite head gradients"):
        training.train_png(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"],
                           data_mode="synthetic_fixture", model_builder=nonfinite_gradient)
