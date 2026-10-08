"""Step 6C fixture checks: no real patient pixels, trained weights or downloads."""
from argparse import Namespace
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "step6c_cli_under_test", WORKSPACE / "learning" / "06c_review_predictions.py")
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def snapshot(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


@dataclass(frozen=True)
class FixtureConfig:
    input_channels: int = 1
    size: int = 256
    mean: float = 0.456
    std: float = 0.224
    version: str = "png-single-slice-gray8-normalized-v1"

    def to_dict(self):
        return asdict(self)


def state_of(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_hash(state):
    return digest(b"".join(key.encode() + value.numpy().tobytes() for key, value in sorted(state.items())))


class FixtureModel(torch.nn.Module):
    """Small deterministic inference stub, deliberately different from references."""
    def __init__(self, state):
        super().__init__()
        self.register_buffer("fixed", torch.tensor(1.0))
        self.observed = state

    def forward(self, inputs):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert inputs.shape == (1, 1, 256, 256)
        self.observed["forward_inputs"].append(inputs.detach().cpu().clone())
        logits = torch.zeros((1, 3, 256, 256))
        logits[:, 0] = 2
        logits[:, 1, 80:84, 70:74] = 4
        logits[:, 2, 80:82, 70:72] = 6
        if self.observed["logit_error"] == "shape":
            return logits[:, :2]
        if self.observed["logit_error"] == "nan":
            logits[0, 0, 0, 0] = float("nan")
        return logits


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root, run_dir = tmp_path / "original_data", tmp_path / "run01"
    data = root / "data2d"
    data.mkdir(parents=True)
    run_dir.mkdir()
    args = Namespace(root=root, cohort=tmp_path / "cohort.json", audit=tmp_path / "audit.json",
                     run_dir=run_dir, output_dir=tmp_path / "reviews" / "run01",
                     check_only=False, cpu_threads=2)
    paths = {"cohort": args.cohort, "audit": args.audit,
             "manifest": data / "manifest.csv", "original_split": data / "splits.json"}
    for key, path in paths.items():
        path.write_text("fixture %s\n" % key)
    config = FixtureConfig()
    pools, offset = {}, 0
    for role, count in (("train", 20), ("validation", 10), ("test", 10), ("prediction", 5)):
        pools[role] = [{"patient_id": "case_%05d" % index, "selected_slice_count": 3}
                       for index in range(offset, offset + count)]
        offset += count
    context = {
        "root": root.resolve(), "data_root": data.resolve(), "paths": paths,
        "file_sha256": {key: digest(path.read_bytes()) for key, path in paths.items()},
        "cohort": {"pools": pools, "cohort_fingerprint_sha256": digest(b"cohort"),
                   "preprocessing": {"config": config.to_dict(), "config_sha256": digest(b"config")}},
        "audit": {"audit_fingerprint_sha256": digest(b"audit")}, "config": config,
    }
    metadata = {
        "schema_version": "png_segmentation_v1", "data_mode": "real",
        "architecture": "smp_unet_resnet18", "input_channels": 1, "input_mode": "single_2d_slice",
        "classes": {0: "background", 1: "kidney", 2: "tumour"},
        "bound_file_sha256": context["file_sha256"].copy(), "preprocessing": config.to_dict(),
        "preprocessing_sha256": digest(b"config"), "cohort": copy.deepcopy(context["cohort"]),
        "audit": copy.deepcopy(context["audit"]), "best_epoch": 1, "epochs": 1,
        "state_fingerprint_sha256": "fixture-state", "source_root": str(root.resolve()),
        "train_slice_count": 60, "validation_slice_count": 30,
    }
    checkpoint = run_dir / "best_model.pt"
    checkpoint.write_bytes(b"fixture checkpoint placeholder; never deserialized as real weights\n")
    checkpoint_hash = digest(checkpoint.read_bytes())
    summary = {
        "schema_version": "png_training_run_v1", "status": "completed",
        "formal_training_completed": True, "epochs_completed": 1, "best_epoch": 1,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash,
                       "schema_version": "png_segmentation_v1"},
        "artifact_sha256": {"best_model.pt": checkpoint_hash},
        "cohort_fingerprint_sha256": context["cohort"]["cohort_fingerprint_sha256"],
        "audit_fingerprint_sha256": context["audit"]["audit_fingerprint_sha256"],
    }
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary))
    observed = {"context": context, "metadata": metadata, "summary": summary, "summary_path": summary_path,
                "dataset_roles": [], "samples": [], "forward_inputs": [], "checkpoint_calls": [],
                "logit_error": None, "config": config, "checkpoint_config": config}

    class FixtureDataset:
        def __init__(self, loaded, role):
            assert loaded is context
            assert role in ("train", "validation"), "Held-out dataset construction is prohibited"
            observed["dataset_roles"].append(role)
            self.role, self.context, self.config = role, context, config
            self.patient_ids = tuple(record["patient_id"] for record in pools[role])
            self.rows = tuple({"case": patient, "z": z, "tumor_px": tumour,
                               "kidney_px": kidney, "image": "images/%s_z%03d.png" % (patient, z),
                               "mask": "masks/%s_z%03d.png" % (patient, z)}
                              for patient in self.patient_ids
                              for z, tumour, kidney in ((2, 4, 20), (10, 10, 20), (30, 2, 30)))

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            patient, z = row["case"], row["z"]
            assert patient in self.patient_ids
            assert patient not in {record["patient_id"] for role in ("test", "prediction") for record in pools[role]}
            observed["samples"].append((self.role, patient, z))
            gray = np.full((256, 256), 80 + z, dtype=np.uint8)
            target = np.zeros((256, 256), dtype=np.uint8)
            target[100:105, 110:115] = 1
            target[100:102, 110:112] = 2
            normalized = (gray.astype(np.float32) / 255 - config.mean) / config.std
            return {"input": torch.from_numpy(normalized[None]), "target": torch.from_numpy(target.astype(np.int64)),
                    "patient_id": patient, "slice_index": z, "sample_id": "%s:z%d" % (patient, z),
                    "source_sha256": {"image": digest(gray.tobytes()), "mask": digest(target.tobytes())},
                    "source_paths": {"image": str(data / row["image"]), "mask": str(data / row["mask"])},
                    "mask_label_counts": {str(label): int((target == label).sum()) for label in (0, 1, 2)}}

    def check_bindings(loaded):
        assert loaded is context
        if any(digest(path.read_bytes()) != context["file_sha256"][key] for key, path in paths.items()):
            raise ValueError("A bound file changed during review")

    runtime = SimpleNamespace(load_training_context=lambda *unused: context,
                              AuditedPNGDataset=FixtureDataset, _check_bindings=check_bindings,
                              _state=state_of, _state_hash=state_hash)
    model = FixtureModel(observed)

    def load_stub(actual_runtime, path, check_only=False):
        assert actual_runtime is runtime and path == checkpoint
        observed["checkpoint_calls"].append(check_only)
        return (None if check_only else model), observed["checkpoint_config"], metadata

    monkeypatch.setattr(CLI, "load_runtime", lambda: runtime)
    monkeypatch.setattr(CLI, "load_checked_checkpoint", load_stub)
    source = WORKSPACE / "kidney-segmentation" if (WORKSPACE / "kidney-segmentation").is_dir() else WORKSPACE
    monkeypatch.syspath_prepend(str(source))
    observed["model"] = model
    observed["runtime"] = runtime
    return args, observed


def selection_dataset(rows, patients=("case_00000",)):
    return SimpleNamespace(rows=rows, patient_ids=patients)


def test_selection_is_fixed_distinct_and_uses_original_axial_positions():
    rows = [{"case": "case_00000", "z": z, "tumor_px": tumour, "kidney_px": 4}
            for z, tumour in ((2, 4), (10, 10), (30, 2))]
    selected = CLI.select_indices(selection_dataset(rows))
    assert [rows[index]["z"] for index, _ in selected] == [10, 2]
    assert len({index for index, _ in selected}) == 2
    assert all(isinstance(reason, dict) and reason["reason"] for _, reason in selected)
    assert CLI.select_indices(selection_dataset(rows)) == selected


def test_selection_tumour_area_tie_uses_smaller_z():
    rows = [{"case": "case_00000", "z": z, "tumor_px": tumour, "kidney_px": 4}
            for z, tumour in ((20, 9), (2, 9), (50, 1))]
    selected = CLI.select_indices(selection_dataset(rows))
    assert rows[selected[0][0]]["z"] == 2
    assert len({index for index, _ in selected}) == 2


@pytest.mark.parametrize("kidneys", [(20, 10, 0), (0, 0, 0)])
def test_selection_no_tumour_fallback_is_distinct_and_deterministic(kidneys):
    rows = [{"case": "case_00000", "z": z, "tumor_px": 0, "kidney_px": kidney}
            for z, kidney in zip((2, 10, 50), kidneys)]
    selected = CLI.select_indices(selection_dataset(rows))
    assert len(selected) == 2 and len({index for index, _ in selected}) == 2
    assert CLI.select_indices(selection_dataset(rows)) == selected


def test_selection_requires_two_distinct_slices():
    row = {"case": "case_00000", "z": 2, "tumor_px": 3, "kidney_px": 4}
    with pytest.raises(ValueError):
        CLI.select_indices(selection_dataset([row]))


def test_kidney_median_excludes_tumour_only_rows_and_is_based_on_original_z():
    rows = [{"case": "case_00000", "z": z, "tumor_px": tumour, "kidney_px": kidney}
            for z, tumour, kidney in ((1, 99, 0), (2, 1, 4), (100, 2, 4), (101, 3, 4))]
    selected = CLI.select_indices(selection_dataset(rows))
    assert [rows[index]["z"] for index, _ in selected] == [1, 100]
    assert selected[1][1]["requested_original_slice_index"] == 100


def test_checkpoint_preflight_checks_tensors_without_initializing_model(tmp_path):
    state = {"fixture": torch.tensor([1.0, 2.0])}
    metadata = {"state_fingerprint_sha256": state_hash(state)}
    checkpoint = tmp_path / "fixture.pt"
    torch.save({"schema_version": "png_segmentation_v1", "metadata": metadata,
                "state_dict": state}, checkpoint)
    calls = []
    config = FixtureConfig()

    def check_metadata(actual):
        calls.append(actual)
        assert actual == metadata
        return config

    runtime = SimpleNamespace(SCHEMA="png_segmentation_v1", _validate_metadata=check_metadata,
                              _state_hash=state_hash,
                              load_png_checkpoint=lambda *args, **kwargs: pytest.fail("Preflight initialized a model"))
    assert CLI.load_checked_checkpoint(runtime, checkpoint, check_only=True) == (None, config, metadata)
    assert calls == [metadata]


@pytest.mark.parametrize("wrong", ["schema", "state_fingerprint"])
def test_checkpoint_preflight_rejects_wrong_schema_or_tensor_fingerprint(tmp_path, wrong):
    metadata = {"state_fingerprint_sha256": "wrong fingerprint"}
    checkpoint = tmp_path / "fixture.pt"
    torch.save({"schema_version": "old_volume_schema" if wrong == "schema" else "png_segmentation_v1",
                "metadata": metadata, "state_dict": {"fixture": torch.tensor([1.0])}}, checkpoint)
    runtime = SimpleNamespace(SCHEMA="png_segmentation_v1", _validate_metadata=lambda unused: FixtureConfig(),
                              _state_hash=state_hash,
                              load_png_checkpoint=lambda *args, **kwargs: pytest.fail("Preflight initialized a model"))
    with pytest.raises(ValueError):
        CLI.load_checked_checkpoint(runtime, checkpoint, check_only=True)


def test_checkpoint_inference_loader_uses_cpu_complete_checkpoint_not_pretrained_initializer(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    expected = object(), FixtureConfig(), {"fixture": True}
    calls = []

    def load(path, device):
        calls.append((path, device))
        return expected

    assert CLI.load_checked_checkpoint(SimpleNamespace(load_png_checkpoint=load), checkpoint) == expected
    assert calls == [(checkpoint, "cpu")]


def test_check_only_never_constructs_pixel_dataset_or_writes_output(environment):
    args, observed = environment
    args.check_only = True
    original = snapshot(args.root), snapshot(args.run_dir)
    report = CLI.run(args)
    assert report["status"] == "preflight_completed"
    assert not observed["dataset_roles"] and not observed["samples"] and not observed["forward_inputs"]
    assert observed["checkpoint_calls"] == [True]
    assert not args.output_dir.exists() and not args.output_dir.parent.exists()
    assert (snapshot(args.root), snapshot(args.run_dir)) == original


@pytest.mark.parametrize("field", ["cohort", "audit", "manifest", "original_split"])
def test_external_checkpoint_binding_mismatch_stops_before_pixels_or_output(environment, field):
    args, observed = environment
    observed["metadata"]["bound_file_sha256"][field] = digest(b"wrong binding")
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not observed["dataset_roles"] and not observed["samples"] and not observed["forward_inputs"]
    assert not args.output_dir.exists()


@pytest.mark.parametrize("subject", ["cohort", "audit"])
def test_checkpoint_fingerprint_mismatch_stops_before_pixels(environment, subject):
    args, observed = environment
    observed["metadata"][subject][subject + "_fingerprint_sha256"] = digest(b"wrong fingerprint")
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not observed["samples"] and not args.output_dir.exists()


def test_checkpoint_config_mismatch_stops_before_pixels(environment):
    args, observed = environment
    observed["checkpoint_config"] = FixtureConfig(mean=0.5)
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not observed["samples"] and not args.output_dir.exists()


@pytest.mark.parametrize("subject", ["checkpoint_bytes", "checkpoint_summary", "artifact_summary"])
def test_checkpoint_hash_rejection_preserves_source_run(environment, subject):
    args, observed = environment
    if subject == "checkpoint_bytes":
        (args.run_dir / "best_model.pt").write_bytes(b"changed fixture checkpoint")
    else:
        if subject == "checkpoint_summary":
            observed["summary"]["checkpoint"]["sha256"] = digest(b"wrong hash")
        else:
            observed["summary"]["artifact_sha256"]["best_model.pt"] = digest(b"wrong hash")
        observed["summary_path"].write_text(json.dumps(observed["summary"]))
    before = snapshot(args.root), snapshot(args.run_dir)
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not observed["samples"] and not args.output_dir.exists()
    assert (snapshot(args.root), snapshot(args.run_dir)) == before


@pytest.mark.parametrize("field,value", [("status", "running"), ("formal_training_completed", False),
                                        ("schema_version", "old_run")])
def test_uncompleted_or_unknown_run_rejected(environment, field, value):
    args, observed = environment
    observed["summary"][field] = value
    observed["summary_path"].write_text(json.dumps(observed["summary"]))
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not observed["samples"] and not args.output_dir.exists()


def test_fixture_checkpoint_is_not_accepted_by_real_cli(environment):
    args, observed = environment
    observed["metadata"]["data_mode"] = "synthetic_fixture"
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not observed["samples"] and not args.output_dir.exists()


@pytest.mark.parametrize("location", ["original_data", "source_run"])
def test_outputs_inside_original_data_or_source_run_are_rejected(environment, location):
    args, observed = environment
    args.output_dir = (args.root if location == "original_data" else args.run_dir) / "unsafe_review"
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not observed["samples"] and not args.output_dir.exists()


def test_existing_output_is_preserved_before_pixels(environment):
    args, observed = environment
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "human_notes.txt").write_text("Keep these existing review notes")
    before = snapshot(args.output_dir)
    with pytest.raises(ValueError):
        CLI.run(args)
    assert snapshot(args.output_dir) == before and not observed["samples"]


def test_symlink_output_is_rejected_before_pixels(environment):
    args, observed = environment
    args.output_dir.parent.mkdir(parents=True)
    target = args.output_dir.parent / "other_directory"
    target.mkdir()
    args.output_dir.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        CLI.run(args)
    assert not observed["samples"] and not list(target.iterdir())


@pytest.mark.parametrize("threads", [0, 17])
def test_invalid_cpu_threads_fail_before_context_or_pixels(environment, threads):
    args, observed = environment
    args.cpu_threads = threads
    with pytest.raises(ValueError):
        CLI.run(args)
    assert not observed["checkpoint_calls"] and not observed["samples"] and not args.output_dir.exists()


def test_model_state_drift_is_reported_as_failed_and_threads_restored(environment, monkeypatch):
    args, observed = environment
    previous_threads = torch.get_num_threads()
    model, original_forward = observed["model"], observed["model"].forward

    def changing_forward(inputs):
        result = original_forward(inputs)
        model.fixed.add_(1)
        return result

    monkeypatch.setattr(model, "forward", changing_forward)
    with pytest.raises(ValueError, match="state changed"):
        CLI.run(args)
    report = json.loads((args.output_dir / "predictions_review.json").read_bytes())
    assert report["status"] == "failed" and "model_state_unchanged" not in report
    assert torch.get_num_threads() == previous_threads


def test_summary_drift_during_inference_prevents_completed_evidence(environment, monkeypatch):
    args, observed = environment
    original_forward = observed["model"].forward

    def changing_forward(inputs):
        result = original_forward(inputs)
        if len(observed["forward_inputs"]) == 1:
            observed["summary_path"].write_bytes(observed["summary_path"].read_bytes() + b" ")
        return result

    monkeypatch.setattr(observed["model"], "forward", changing_forward)
    with pytest.raises(ValueError, match="changed during"):
        CLI.run(args)
    report = json.loads((args.output_dir / "predictions_review.json").read_bytes())
    assert report["status"] == "failed" and "finished_at_utc" not in report


@pytest.mark.parametrize("error", ["shape", "nan"])
def test_invalid_logits_are_rejected_without_modifying_source_run(environment, error):
    args, observed = environment
    observed["logit_error"] = error
    before = snapshot(args.root), snapshot(args.run_dir)
    with pytest.raises(ValueError):
        CLI.run(args)
    assert (snapshot(args.root), snapshot(args.run_dir)) == before
    report = json.loads((args.output_dir / "predictions_review.json").read_bytes())
    assert report["status"] == "failed" and "logits" in report["error_message"]


def test_review_reads_exactly_two_train_validation_samples_and_no_heldout(environment):
    args, observed = environment
    before = snapshot(args.root), snapshot(args.run_dir)
    state_before = state_hash(state_of(observed["model"]))
    report = CLI.run(args)
    assert (args.output_dir / "predictions_review.json").is_file()
    assert json.loads((args.output_dir / "predictions_review.json").read_bytes()) == report
    assert observed["dataset_roles"] == ["train", "validation"]
    assert len(observed["samples"]) == len(observed["forward_inputs"]) == 60
    assert sum(role == "train" for role, _, _ in observed["samples"]) == 40
    assert sum(role == "validation" for role, _, _ in observed["samples"]) == 20
    assert len(set(observed["samples"])) == 60
    for (_, _, z), inputs in zip(observed["samples"], observed["forward_inputs"]):
        expected = ((80 + z) / 255 - observed["config"].mean) / observed["config"].std
        assert torch.allclose(inputs, torch.full_like(inputs, expected))
    assert state_hash(state_of(observed["model"])) == state_before
    assert (snapshot(args.root), snapshot(args.run_dir)) == before
    assert report["aggregate_performance_evaluation"] is False
    assert report["clinical_alignment_verified"] is False
    assert report.get("selection_uses_reference_labels") is True
    assert report.get("manual_review_status") == "pending"
    assert report["decoded_pairs"] == {"train": 40, "validation": 20, "test": 0, "prediction": 0}
    assert report["training_performed"] is False and report["optimizer_steps"] == 0
    assert report["model_state_unchanged"] is True
    assert report["grad_cam_performed"] is False
    first = report["patients"][0]
    assert len(first["samples"]) == 2
    preview = Image.open(args.output_dir / first["preview_path"])
    for index, sample in enumerate(first["samples"]):
        z = sample["original_slice_index"]
        y = 110 + index * 304
        ct = np.asarray(preview.crop((12, y, 268, y + 256)).convert("L"))
        reference = np.asarray(preview.crop((288, y, 544, y + 256)))
        prediction = np.asarray(preview.crop((564, y, 820, y + 256)))
        assert np.all(ct == 80 + z)
        assert not np.array_equal(reference, prediction)
        with Image.open(args.output_dir / sample["predicted_mask_path"]) as label_image:
            labels = np.asarray(label_image)
            assert label_image.mode == "L" and label_image.size == (256, 256)
            assert set(np.unique(labels)) == {0, 1, 2}
            assert labels[80, 70] == 2 and labels[100, 110] == 0
        assert sample["predicted_mask_sha256"] == digest((args.output_dir / sample["predicted_mask_path"]).read_bytes())
