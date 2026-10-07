"""Fixed tumour CE weighting: independent arithmetic and synthetic PNG checks."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from kidney_segmentation import png_training as training
from kidney_segmentation.cohort import canonical_fingerprint
from kidney_segmentation.model import segmentation_loss, validate_tumour_ce_weight
from test_png_training import fixture_data, tiny_builder


def original_loss(logits, target):
    ce = F.cross_entropy(logits, target)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, 3).permute(0, 3, 1, 2).to(probabilities.dtype)
    intersection = (probabilities * one_hot).sum((0, 2, 3))[1:]
    denominator = (probabilities + one_hot).sum((0, 2, 3))[1:]
    dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return ce + dice_loss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_default_one_retains_exact_original_loss_and_gradients(dtype):
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn((2, 3, 4, 5), generator=generator, dtype=dtype, requires_grad=True)
    baseline_logits = logits.detach().clone().requires_grad_(True)
    target = torch.randint(0, 3, (2, 4, 5), generator=generator)
    observed = segmentation_loss(logits, target)
    expected = original_loss(baseline_logits, target)
    observed.backward()
    expected.backward()
    assert torch.equal(observed, expected)
    assert torch.equal(logits.grad, baseline_logits.grad)


def test_weighted_mean_and_foreground_dice_match_independent_three_pixel_hand_calculation():
    probabilities = torch.tensor([[[[0.6, 0.2, 0.1]], [[0.2, 0.5, 0.2]], [[0.2, 0.3, 0.7]]]], dtype=torch.float64)
    target = torch.tensor([[[0, 1, 2]]], dtype=torch.int64)
    weighted_numerator = -math.log(0.6) - math.log(0.5) - 2 * math.log(0.7)
    weighted_ce = weighted_numerator / 4
    kidney_dice = (2 * 0.5 + 1) / (0.2 + 0.5 + 0.2 + 1 + 1)
    tumour_dice = (2 * 0.7 + 1) / (0.2 + 0.3 + 0.7 + 1 + 1)
    expected_dice_loss = 1 - (kidney_dice + tumour_dice) / 2
    observed = segmentation_loss(probabilities.log(), target, tumour_ce_weight=2)
    assert observed.item() == pytest.approx(weighted_ce + expected_dice_loss, abs=1e-12)
    assert observed.item() != pytest.approx(weighted_numerator / 3 + expected_dice_loss)


def test_default_skips_weight_tensor_and_other_weight_uses_logits_dtype_device(monkeypatch):
    original_ce, observed = F.cross_entropy, []

    def spy(logits, target, **kwargs):
        observed.append(kwargs)
        return original_ce(logits, target, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", spy)
    logits = torch.zeros((1, 3, 1, 3), dtype=torch.float64)
    target = torch.tensor([[[0, 1, 2]]])
    segmentation_loss(logits, target, tumour_ce_weight=1)
    segmentation_loss(logits, target, tumour_ce_weight=2)
    assert observed[0] == {}
    assert observed[1]["reduction"] == "mean"
    assert observed[1]["weight"].tolist() == [1.0, 1.0, 2.0]
    assert observed[1]["weight"].dtype == logits.dtype and observed[1]["weight"].device == logits.device


@pytest.mark.parametrize("weight", [None, True, False, np.bool_(True), 0, -1, float("nan"),
                                     float("inf"), -float("inf"), "2"])
def test_invalid_weight_fails_before_tensors_or_data_are_used(monkeypatch, weight):
    monkeypatch.setattr(training, "load_training_context", lambda *args: pytest.fail("Invalid weight read data"))
    with pytest.raises(ValueError, match="positive finite"):
        segmentation_loss(None, None, tumour_ce_weight=weight)
    with pytest.raises(ValueError, match="positive finite"):
        training.train_png(Path("unused"), Path("unused"), Path("unused"), tumour_ce_weight=weight)
    with pytest.raises(ValueError, match="positive finite"):
        training.evaluate_png_validation(None, None, tumour_ce_weight=weight)


@pytest.mark.parametrize("weight", [0.25, 1, 2.0, np.float64(3.5), 10000])
def test_any_positive_finite_relative_weight_is_supported_without_an_arbitrary_upper_limit(weight):
    assert validate_tumour_ce_weight(weight) == float(weight)
    assert training.png_loss_configuration(weight)["ce_class_weights"] == [1.0, 1.0, float(weight)]


def loss_metadata(weight=2.0):
    return {"epochs": 1, "tumour_ce_weight": weight, "loss_configuration": training.png_loss_configuration(weight),
            "best_validation_report": {"tumour_ce_weight": weight,
                                       "loss_configuration": training.png_loss_configuration(weight)},
            "history": [{"epoch": 1, "tumour_ce_weight": weight}]}


@pytest.mark.parametrize("changed", ["top_weight", "configuration", "report_weight", "report_configuration",
                                     "history_weight", "missing_weight", "missing_configuration",
                                     "both_top_missing", "missing_report_configuration", "missing_history_weight"])
def test_rehashed_loss_provenance_drift_or_partial_evidence_is_rejected(changed):
    metadata = loss_metadata()
    if changed == "top_weight":
        metadata["tumour_ce_weight"] = 3.0
    elif changed == "configuration":
        metadata["loss_configuration"]["ce_class_weights"][2] = 3.0
    elif changed == "report_weight":
        metadata["best_validation_report"]["tumour_ce_weight"] = 3.0
    elif changed == "report_configuration":
        metadata["best_validation_report"]["loss_configuration"]["ce_reduction"] = "sum"
    elif changed == "history_weight":
        metadata["history"][0]["tumour_ce_weight"] = 3.0
    elif changed == "missing_weight":
        del metadata["tumour_ce_weight"]
    elif changed == "missing_configuration":
        del metadata["loss_configuration"]
    elif changed == "both_top_missing":
        del metadata["tumour_ce_weight"], metadata["loss_configuration"]
    elif changed == "missing_report_configuration":
        del metadata["best_validation_report"]["loss_configuration"]
    else:
        del metadata["history"][0]["tumour_ce_weight"]
    metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
    with pytest.raises(ValueError, match="loss configuration|CE weight"):
        training._validate_loss_configuration(metadata)


def fitted(data, weight=2.0, **kwargs):
    return training.train_png(data["root"], data["cohort_path"], data["audit_path"],
                              model_builder=tiny_builder, data_mode="synthetic_fixture",
                              selection_metric="tumour_dice", tumour_ce_weight=weight, **kwargs)


def test_tiny_png_training_validation_weight_propagation_and_checkpoint_metadata(fixture_data, monkeypatch):
    actual, calls, events = training.segmentation_loss, [], []

    def spy(logits, target, *, tumour_ce_weight=1.0):
        calls.append(("train" if torch.is_grad_enabled() else "validation", tumour_ce_weight))
        return actual(logits, target, tumour_ce_weight=tumour_ce_weight)

    monkeypatch.setattr(training, "segmentation_loss", spy)
    result = fitted(fixture_data, progress=events.append)
    assert calls.count(("train", 2.0)) == 21 and calls.count(("validation", 2.0)) == 10
    metadata, report = result["metadata"], result["best_validation_report"]
    expected = training.png_loss_configuration(2.0)
    assert metadata["tumour_ce_weight"] == report["tumour_ce_weight"] == 2.0
    assert metadata["loss_configuration"] == report["loss_configuration"] == expected
    assert metadata["history"][0]["tumour_ce_weight"] == 2.0
    assert metadata["encoder_state_unchanged"] is True
    assert metadata["optimizer_steps_completed"] == 21
    assert all(event["tumour_ce_weight"] == 2.0 for event in events)
    assert events[0]["loss_configuration"] == expected
    assert "reference-class weights" in report["loss_scope"]
    path = fixture_data["tmp"] / "weighted.pt"
    training.save_png_checkpoint(path, result["model"], metadata)
    _, _, restored = training.load_png_checkpoint(path, model_builder=tiny_builder)
    assert restored == metadata


def test_validation_objective_weight_does_not_change_fixed_model_hard_patient_metrics(fixture_data):
    result = fitted(fixture_data)
    context = training.load_training_context(fixture_data["root"], fixture_data["cohort_path"], fixture_data["audit_path"])
    dataset = training.AuditedPNGDataset(context, "validation")
    model = result["model"]
    before = training._state_hash(model.state_dict())
    unweighted = training.evaluate_png_validation(model, dataset, tumour_ce_weight=1.0)
    weighted = training.evaluate_png_validation(model, dataset, tumour_ce_weight=2.0)
    for key in ("patient_mean", "contributing_patient_counts", "patients"):
        assert unweighted[key] == weighted[key]
    assert before == training._state_hash(model.state_dict())
    assert unweighted["loss_configuration"] != weighted["loss_configuration"]


def test_legacy_v1_without_any_loss_weight_fields_loads_unchanged(fixture_data):
    result = fitted(fixture_data, weight=1.0)
    metadata = copy.deepcopy(result["metadata"])
    for target in (metadata, metadata["best_validation_report"], *metadata["history"]):
        target.pop("tumour_ce_weight", None)
        target.pop("loss_configuration", None)
    metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
    before = copy.deepcopy(metadata)
    path = fixture_data["tmp"] / "legacy_unweighted.pt"
    torch.save({"schema_version": training.SCHEMA, "metadata": metadata,
                "state_dict": result["model"].state_dict()}, path)
    _, _, restored = training.load_png_checkpoint(path, model_builder=tiny_builder)
    assert restored == before
    assert "tumour_ce_weight" not in restored and "loss_configuration" not in restored


@pytest.mark.parametrize("changed", ["weight", "partial", "history"])
def test_checkpoint_loader_rejects_recomputed_loss_metadata_drift(fixture_data, changed):
    result = fitted(fixture_data)
    metadata = copy.deepcopy(result["metadata"])
    if changed == "weight":
        metadata["tumour_ce_weight"] = 3.0
    elif changed == "partial":
        del metadata["loss_configuration"]
    else:
        metadata["history"][0]["tumour_ce_weight"] = 3.0
    metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
    path = fixture_data["tmp"] / "drift.pt"
    torch.save({"schema_version": training.SCHEMA, "metadata": metadata,
                "state_dict": result["model"].state_dict()}, path)
    with pytest.raises(ValueError, match="loss configuration|CE weight"):
        training.load_png_checkpoint(path, model_builder=tiny_builder)
