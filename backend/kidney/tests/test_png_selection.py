"""Checkpoint-selection protocol checks using synthetic PNGs and a tiny model."""
import copy
from pathlib import Path

import pytest
import torch

from kidney_segmentation import png_training as training
from kidney_segmentation.cohort import canonical_fingerprint
from test_png_training import fixture_data, tiny_builder


def report(loss=0.2, dice=0.4, count=1):
    return {"loss": loss, "patient_count": 10,
            "patient_mean": {"tumour": {"dice": dice}},
            "contributing_patient_counts": {"tumour": {"dice": count}}}


def selection_metadata(metric="tumour_dice", losses=(0.4, 0.1, 0.2), scores=(0.1, 0.2, 0.9)):
    reports = [report(loss, dice) for loss, dice in zip(losses, scores)]
    values = [training._selection_values(item, metric) for item in reports]
    best = max(range(len(values)), key=lambda index: values[index][0])
    history = [{"epoch": index + 1, "validation_loss": item["loss"],
                "validation_patient_mean": item["patient_mean"],
                "validation_contributing_patient_counts": item["contributing_patient_counts"],
                "selection_metric": metric, "selection_score": values[index][1]}
               for index, item in enumerate(reports)]
    return {"epochs": len(history), "best_epoch": best + 1, "history": history,
            "best_validation_report": reports[best], "best_validation_loss": reports[best]["loss"],
            "lowest_validation_loss_observed": min(losses),
            "checkpoint_selection": {**training._selection_rules(metric),
                                     "best_score": values[best][1],
                                     "selected_validation_loss": reports[best]["loss"],
                                     "lowest_validation_loss_observed": min(losses)}}


def test_zero_dice_is_valid_and_loss_is_an_explicit_secondary_rank():
    rank, score, loss = training._selection_values(report(dice=0), "tumour_dice")
    assert (rank, score, loss) == ((0.0, -0.2), 0.0, 0.2)
    metadata = selection_metadata(losses=(0.4, 0.2, 0.2), scores=(0.0, 0.0, 0.0))
    assert metadata["best_epoch"] == 2
    training._validate_checkpoint_selection(metadata)


def test_exact_dice_loss_ties_keep_the_earliest_epoch():
    metadata = selection_metadata(losses=(0.4, 0.2, 0.2), scores=(0.5, 0.5, 0.5))
    assert metadata["best_epoch"] == 2
    training._validate_checkpoint_selection(metadata)
    loss_selected = selection_metadata("validation_loss", losses=(0.2, 0.2, 0.4))
    assert loss_selected["best_epoch"] == 1
    training._validate_checkpoint_selection(loss_selected)


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_undefined_or_invalid_tumour_dice_never_falls_back_to_loss(score):
    with pytest.raises(ValueError, match="no fallback"):
        training._selection_values(report(dice=score), "tumour_dice")


@pytest.mark.parametrize("count", [None, 0, -1, True, 1.5, 11])
def test_dice_requires_consistent_positive_integer_contributors(count):
    with pytest.raises(ValueError):
        training._selection_values(report(count=count), "tumour_dice")


@pytest.mark.parametrize("field,value", [("patient_mean", None), ("patient_mean", {"tumour": None}),
                                         ("contributing_patient_counts", None),
                                         ("contributing_patient_counts", {"tumour": None})])
def test_malformed_metric_mapping_raises_a_clear_selection_error(field, value):
    invalid = report()
    invalid[field] = value
    with pytest.raises(ValueError, match="no fallback"):
        training._selection_values(invalid, "tumour_dice")


@pytest.mark.parametrize("loss", [None, float("nan"), float("inf"), -1, True, "0.2"])
@pytest.mark.parametrize("metric", training.SELECTION_METRICS)
def test_invalid_validation_loss_rejected_for_both_ranks(loss, metric):
    with pytest.raises(ValueError, match="finite non-negative"):
        training._selection_values(report(loss=loss), metric)


@pytest.mark.parametrize("metric", [None, "dice", "accuracy", True])
def test_unknown_selection_metric_fails_before_data_access(monkeypatch, metric):
    monkeypatch.setattr(training, "load_training_context", lambda *args: pytest.fail("Invalid criterion read data"))
    with pytest.raises(ValueError, match="selection metric"):
        training.train_png(Path("unused"), Path("unused"), Path("unused"), selection_metric=metric)


@pytest.mark.parametrize("field", ["best_epoch", "score", "selected_loss", "lowest_loss", "direction", "tie_break",
                                  "history_score", "history_loss", "history_dice", "report_dice"])
def test_recomputed_metadata_cannot_claim_a_different_selected_epoch_or_score(field):
    metadata = selection_metadata()
    if field == "best_epoch":
        metadata["best_epoch"] = 2
    elif field == "score":
        metadata["checkpoint_selection"]["best_score"] = 0.8
    elif field == "selected_loss":
        metadata["checkpoint_selection"]["selected_validation_loss"] = 0.1
    elif field == "lowest_loss":
        metadata["lowest_validation_loss_observed"] = 0.2
    elif field == "direction":
        metadata["checkpoint_selection"]["direction"] = "min"
    elif field == "tie_break":
        metadata["checkpoint_selection"]["tie_break"] = "latest_epoch"
    elif field == "history_score":
        metadata["history"][2]["selection_score"] = 0.8
    elif field == "history_loss":
        metadata["history"][2]["validation_loss"] = 0.05
    elif field == "history_dice":
        metadata["history"][2]["validation_patient_mean"] = {"tumour": {"dice": 0.8}}
        metadata["history"][2]["selection_score"] = 0.8
    else:
        metadata["best_validation_report"]["patient_mean"] = {"tumour": {"dice": 0.8}}
    metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
    with pytest.raises(ValueError):
        training._validate_checkpoint_selection(metadata)


@pytest.mark.parametrize("field", ["best_score", "selected_validation_loss", "lowest_validation_loss_observed"])
def test_numeric_selection_evidence_cannot_be_a_boolean_even_when_equal_to_one(field):
    metadata = selection_metadata(losses=(1.0, 1.0, 1.0), scores=(1.0, 1.0, 1.0))
    metadata["checkpoint_selection"][field] = True
    with pytest.raises(ValueError, match="finite numbers"):
        training._validate_checkpoint_selection(metadata)


def test_selected_report_contributors_must_match_its_history():
    metadata = selection_metadata()
    metadata["best_validation_report"]["contributing_patient_counts"] = {"tumour": {"dice": 2}}
    with pytest.raises(ValueError, match="selected epoch/report"):
        training._validate_checkpoint_selection(metadata)


@pytest.mark.parametrize("marker,value", [("selection_metric", "tumour_dice"),
                                          ("selection_score", 0.9),
                                          ("checkpoint_selected", True),
                                          ("validation_contributing_patient_counts", {"tumour": {"dice": 1}})])
def test_each_new_history_marker_prevents_partial_evidence_from_loading_as_legacy(marker, value):
    metadata = selection_metadata()
    del metadata["checkpoint_selection"]
    del metadata["lowest_validation_loss_observed"]
    for record in metadata["history"]:
        for key in ("selection_metric", "selection_score", "checkpoint_selected", "validation_contributing_patient_counts"):
            record.pop(key, None)
    metadata["history"][-1][marker] = value
    with pytest.raises(ValueError, match="selection evidence is incomplete"):
        training._validate_checkpoint_selection(metadata)


def run_controlled(data, monkeypatch, *, metric=None, losses=(0.4, 0.1, 0.2), scores=(0.1, 0.2, 0.9)):
    states, events = [], []
    actual_evaluation = training.evaluate_png_validation

    def controlled(model, dataset, **kwargs):
        result = actual_evaluation(model, dataset, **kwargs)
        index = kwargs["epoch"] - 1
        # These are declared fixture selection values, not measured model quality.
        result["loss"] = losses[index]
        result["patient_mean"]["tumour"]["dice"] = scores[index]
        result["contributing_patient_counts"]["tumour"]["dice"] = 1
        states.append(copy.deepcopy(model.state_dict()))
        return result

    monkeypatch.setattr(training, "evaluate_png_validation", controlled)
    options = {} if metric is None else {"selection_metric": metric}
    result = training.train_png(data["root"], data["cohort_path"], data["audit_path"],
                               epochs=len(losses), model_builder=tiny_builder, data_mode="synthetic_fixture",
                               progress=events.append, **options)
    return result, states, events


@pytest.mark.parametrize("metric,expected_epoch", [(None, 2), ("tumour_dice", 3)])
def test_selected_checkpoint_restores_exact_epoch_weights_and_records_protocol(fixture_data, monkeypatch, metric, expected_epoch):
    result, states, events = run_controlled(fixture_data, monkeypatch, metric=metric)
    metadata = result["metadata"]
    assert metadata["best_epoch"] == expected_epoch
    assert all(torch.equal(result["model"].state_dict()[key], value)
               for key, value in states[expected_epoch - 1].items())
    if metric is None:
        assert metadata["checkpoint_selection"]["metric"] == "validation_loss"
        assert metadata["best_validation_loss"] == metadata["lowest_validation_loss_observed"] == 0.1
    else:
        assert metadata["best_validation_loss"] == 0.2
        assert metadata["lowest_validation_loss_observed"] == 0.1
        assert metadata["checkpoint_selection"]["best_score"] == 0.9
    assert events[-1]["selection_score"] == metadata["checkpoint_selection"]["best_score"]
    assert events[0]["selection_metric"] == metadata["checkpoint_selection"]["metric"]
    assert [row["checkpoint_selected"] for row in metadata["history"]] == [True, True, metric is not None]
    path = fixture_data["tmp"] / "selected.pt"
    training.save_png_checkpoint(path, result["model"], metadata)
    restored, _, actual = training.load_png_checkpoint(path, model_builder=tiny_builder)
    assert actual == metadata
    assert all(torch.equal(restored.state_dict()[key], value)
               for key, value in states[expected_epoch - 1].items())


def test_zero_score_is_saved_in_real_training_orchestration(fixture_data, monkeypatch):
    result, states, _ = run_controlled(fixture_data, monkeypatch, metric="tumour_dice",
                                      losses=(0.4, 0.2, 0.2), scores=(0.0, 0.0, 0.0))
    assert result["metadata"]["best_epoch"] == 2
    assert result["metadata"]["checkpoint_selection"]["best_score"] == 0.0
    assert all(torch.equal(result["model"].state_dict()[key], value) for key, value in states[1].items())


def test_selection_does_not_change_any_optimization_trajectory(fixture_data, monkeypatch):
    actual_evaluation = training.evaluate_png_validation
    loss_result, loss_states, _ = run_controlled(fixture_data, monkeypatch)
    monkeypatch.setattr(training, "evaluate_png_validation", actual_evaluation)
    dice_result, dice_states, _ = run_controlled(fixture_data, monkeypatch, metric="tumour_dice")
    assert loss_result["metadata"]["best_epoch"] != dice_result["metadata"]["best_epoch"]
    for left, right in zip(loss_states, dice_states):
        assert set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)
    assert [r["train_loss"] for r in loss_result["history"]] == [r["train_loss"] for r in dice_result["history"]]
    assert loss_result["metadata"]["optimizer_steps_completed"] == dice_result["metadata"]["optimizer_steps_completed"] == 63


def test_legacy_v1_checkpoint_without_selection_fields_loads_without_metadata_mutation(fixture_data, monkeypatch):
    result, _, _ = run_controlled(fixture_data, monkeypatch)
    metadata = copy.deepcopy(result["metadata"])
    del metadata["checkpoint_selection"]
    del metadata["lowest_validation_loss_observed"]
    for key in ("tumour_ce_weight", "loss_configuration"):
        del metadata[key]
        del metadata["best_validation_report"][key]
    for record in metadata["history"]:
        for key in ("selection_metric", "selection_score", "checkpoint_selected",
                    "validation_contributing_patient_counts", "tumour_ce_weight"):
            del record[key]
    metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
    before = copy.deepcopy(metadata)
    path = fixture_data["tmp"] / "legacy_v1.pt"
    torch.save({"schema_version": training.SCHEMA, "metadata": metadata,
                "state_dict": result["model"].state_dict()}, path)
    _, _, loaded = training.load_png_checkpoint(path, model_builder=tiny_builder)
    assert loaded == before and "checkpoint_selection" not in loaded


def test_checkpoint_loader_rejects_rehashed_partial_selection_before_model_construction(fixture_data, monkeypatch):
    result, _, _ = run_controlled(fixture_data, monkeypatch, metric="tumour_dice")
    metadata = copy.deepcopy(result["metadata"])
    del metadata["checkpoint_selection"]
    del metadata["lowest_validation_loss_observed"]
    metadata["best_epoch"] = 99
    metadata["best_validation_loss"] = -1.0
    metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
    path = fixture_data["tmp"] / "partial_selection.pt"
    torch.save({"schema_version": training.SCHEMA, "metadata": metadata,
                "state_dict": result["model"].state_dict()}, path)

    def forbidden_builder(**kwargs):
        pytest.fail("Partial selection evidence constructed a model")

    with pytest.raises(ValueError, match="selection evidence is incomplete"):
        training.load_png_checkpoint(path, model_builder=forbidden_builder)


@pytest.mark.parametrize("field", ["best_score", "selected_epoch", "history_score"])
def test_checkpoint_loader_rejects_rehashed_selection_tampering(fixture_data, monkeypatch, field):
    result, _, _ = run_controlled(fixture_data, monkeypatch, metric="tumour_dice")
    metadata = copy.deepcopy(result["metadata"])
    if field == "best_score":
        metadata["checkpoint_selection"]["best_score"] = 0.8
    elif field == "selected_epoch":
        metadata["best_epoch"] = 2
    else:
        metadata["history"][2]["selection_score"] = 0.8
    metadata["metadata_fingerprint_sha256"] = canonical_fingerprint(metadata, "metadata_fingerprint_sha256")
    path = fixture_data["tmp"] / (field + ".pt")
    torch.save({"schema_version": training.SCHEMA, "metadata": metadata,
                "state_dict": result["model"].state_dict()}, path)
    with pytest.raises(ValueError, match="selection|selected"):
        training.load_png_checkpoint(path, model_builder=tiny_builder)
