"""Minimal training, audited evaluation, checkpointing and mask-free inference."""
import copy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from . import CLASSES
from .data import file_hash, volume_fingerprint
from .metrics import confusion_matrix, patient_report, summarize_patients
from .model import KidneyUNet, segmentation_loss
from .preprocessing import PreprocessConfig, preprocess_slice, validate_volume


def _identity_map(dataset):
    return {patient_id: source_id for patient_id, source_id, _ in dataset.identities}


def _validate_lineage(lineage, require_complete=False):
    patient_ids, source_ids = [], []
    for split in ("train", "validation", "test", "prediction"):
        group = lineage.get(split, {})
        if not isinstance(group, dict) or (require_complete and not group):
            raise ValueError("Complete canonical train/validation/test/prediction lineage is required")
        patient_ids.extend(group)
        source_ids.extend(group.values())
    for values in (patient_ids, source_ids):
        if any(not isinstance(value, str) or not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("Patient/source leakage across canonical lineage splits")


class PatientBalancedSampler(Sampler):
    """Group a uniform slice sample by patient so full volumes are cached once."""
    def __init__(self, ends, seed=42):
        self.ends = ends
        self.lengths = np.diff([0] + ends)
        self.slices_per_patient = int(np.median(self.lengths))
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        for patient_index in torch.randperm(len(self.ends), generator=self.generator).tolist():
            offset = self.ends[patient_index - 1] if patient_index else 0
            for index in torch.randint(int(self.lengths[patient_index]), (self.slices_per_patient,), generator=self.generator).tolist():
                yield offset + index

    def __len__(self):
        return len(self.ends) * self.slices_per_patient


def train_model(training, validation, epochs=1, batch_size=4, lr=1e-3, pretrained=True,
                seed=42, device="cpu", progress=None, lineage=None, data_mode="real"):
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Training settings must be positive")
    if training.config != validation.config:
        raise ValueError("Training and validation preprocessing must match")
    if any(split != "train" for _, _, split in training.identities):
        raise ValueError("Training dataset contains a non-training patient")
    if any(split != "validation" for _, _, split in validation.identities):
        raise ValueError("Validation dataset contains a non-validation patient")
    train_ids, val_ids = _identity_map(training), _identity_map(validation)
    if set(train_ids) & set(val_ids) or set(train_ids.values()) & set(val_ids.values()):
        raise ValueError("Training/validation patient leakage")
    if data_mode not in ("real", "synthetic_demo"):
        raise ValueError("Unknown data mode")
    if data_mode == "real":
        if not pretrained or not lineage or not lineage.get("audit_sha256") or not lineage.get("manifest_sha256"):
            raise ValueError("Real training requires pretrained weights and completed canonical audit lineage")
        if (len(train_ids), len(val_ids), len(lineage.get("test", {}))) != (20, 10, 10):
            raise ValueError("Real training requires canonical 20/10/10 patients")
        if lineage.get("train") != train_ids or lineage.get("validation") != val_ids:
            raise ValueError("Training datasets differ from canonical lineage")
    else:
        if any(not patient_id.startswith("synthetic-") for patient_id in [*train_ids, *val_ids]):
            raise ValueError("Synthetic engineering mode only accepts explicitly synthetic patient identities")
    _validate_lineage({**(lineage or {}), "train": train_ids, "validation": val_ids}, require_complete=data_mode == "real")
    torch.manual_seed(seed)
    model = KidneyUNet(pretrained=pretrained).to(device)
    model.lineage = copy.deepcopy(lineage or {})
    model.lineage.update(train=train_ids, validation=val_ids)
    model.preprocessing = training.config
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    # Equal sampled slice count per patient; selection consults no mask/ROI labels.
    sampler = PatientBalancedSampler(training.ends, seed)
    loader = DataLoader(training, batch_size=batch_size, sampler=sampler, num_workers=0)
    best_state, best_loss, best_epoch = None, float("inf"), None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch in loader:
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()
            loss = segmentation_loss(model(images), masks)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became non-finite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
            seen += len(images)
        evaluation = evaluate_dataset(model, validation, batch_size=batch_size, device=device,
                                      expected_split="validation", expected_patient_ids=set(val_ids))
        record = {"epoch": epoch, "epochs": epochs, "train_loss": loss_sum / seen,
                  "validation_loss": evaluation["loss"],
                  "validation_patient_mean": evaluation["patient_mean"],
                  "progress": epoch / epochs}
        history.append(record)
        if record["validation_loss"] < best_loss:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_loss, best_epoch = record["validation_loss"], epoch
        if progress:
            progress(copy.deepcopy(record))
    model.load_state_dict(best_state)
    model.eval()
    metadata = {"architecture": "smp_unet_resnet18", "classes": CLASSES,
                "data_mode": data_mode, "clinical_performance_evidence": False,
                "evaluation_evidence_kind": "real_cohort_segmentation" if data_mode == "real" else "engineering_fixture",
                "pretrained_encoder": pretrained, "encoder_frozen": True,
                "library_versions": {name: version(name) for name in
                                     ("torch", "torchvision", "segmentation-models-pytorch", "opencv-python-headless", "numpy", "nibabel")},
                "preprocessing": training.config.to_dict(), "epochs": epochs, "best_epoch": best_epoch,
                "seed": seed, "learning_rate": lr, "batch_size": batch_size,
                "sampling": "equal slice count per patient; uniform replacement from all slices; grouped patient IO",
                "training_slices_per_patient_per_epoch": sampler.slices_per_patient,
                "lineage": model.lineage, "history": history,
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
    return model, metadata


def evaluate_dataset(model, dataset, batch_size=4, device="cpu", expected_split=None, expected_patient_ids=None):
    if getattr(model, "preprocessing", dataset.config) != dataset.config:
        raise ValueError("Evaluation preprocessing differs from the fitted model")
    ids = _identity_map(dataset)
    if expected_split is None:
        splits = {split for _, _, split in dataset.identities}
        if len(splits) != 1 or not splits <= {"validation", "test"}:
            raise ValueError("Evaluation requires one canonical validation or test split")
        expected_split = splits.pop()
    if expected_split is not None and any(split != expected_split for _, _, split in dataset.identities):
        raise ValueError("Evaluation dataset contains a patient from the wrong split")
    if expected_patient_ids is not None and set(ids) != set(expected_patient_ids):
        raise ValueError("Evaluation membership differs from the canonical split")
    lineage = getattr(model, "lineage", {})
    trained = lineage.get("train", {})
    if set(ids) & set(trained) or set(ids.values()) & set(trained.values()):
        raise ValueError("Training patients cannot be evaluated as held-out patients")
    if expected_split not in ("validation", "test") or not lineage.get(expected_split):
        raise ValueError("Canonical evaluation lineage is required")
    if ids != lineage[expected_split]:
        raise ValueError("Evaluation patient/source identities differ from fitted model lineage")
    model.eval()
    matrices = {patient_id: np.zeros((3, 3), dtype=np.int64) for patient_id in ids}
    loss_sum, seen = 0.0, 0
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            images, target = batch["image"].to(device), batch["mask"].to(device)
            logits = model(images)
            loss_sum += float(segmentation_loss(logits, target)) * len(images)
            seen += len(images)
            predicted = logits.argmax(1).cpu().numpy()
            for patient_id, actual, prediction in zip(batch["patient_id"], target.cpu().numpy(), predicted):
                matrices[patient_id] += confusion_matrix(actual, prediction)
    summary = summarize_patients([patient_report(patient_id, matrix) for patient_id, matrix in matrices.items()])
    summary.update(loss=loss_sum / seen, slice_count=seen, split=expected_split)
    return summary


def save_checkpoint(path, model, metadata):
    if metadata.get("lineage") != getattr(model, "lineage", None):
        raise ValueError("Checkpoint metadata lineage differs from the fitted model")
    if PreprocessConfig(**metadata["preprocessing"]) != getattr(model, "preprocessing", None):
        raise ValueError("Checkpoint preprocessing metadata differs from the fitted model")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload = {"schema_version": 1, "metadata": metadata, "state_dict": state}
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
        model.checkpoint_sha256 = file_hash(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload.get("metadata", {})
    if payload.get("schema_version") != 1 or metadata.get("architecture") != "smp_unet_resnet18":
        raise ValueError("Unsupported Kidney segmentation checkpoint")
    if metadata.get("classes") != CLASSES:
        raise ValueError("Checkpoint class labels do not match the segmentation task")
    if metadata.get("data_mode") not in ("real", "synthetic_demo"):
        raise ValueError("Checkpoint data mode is invalid")
    _validate_lineage(metadata.get("lineage", {}), require_complete=metadata["data_mode"] == "real")
    config = PreprocessConfig(**metadata["preprocessing"])
    # A full checkpoint already contains the encoder: never download weights on load.
    model = KidneyUNet(pretrained=False).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.pretrained = metadata["pretrained_encoder"]
    model.lineage = metadata.get("lineage", {})
    model.preprocessing = config
    model.checkpoint_sha256 = file_hash(path)
    model.eval()
    return model, config, metadata


def predict_volume(model, volume, patient_id, source_patient_id, config=None,
                   batch_size=4, device="cpu", require_new=True):
    """Every slice is inferred; neither a mask nor a label is accepted."""
    volume = validate_volume(volume)
    config = config or getattr(model, "preprocessing", PreprocessConfig())
    if getattr(model, "preprocessing", config) != config:
        raise ValueError("Prediction preprocessing differs from the fitted model")
    if not patient_id or not source_patient_id or batch_size < 1:
        raise ValueError("Prediction needs valid patient identity and batch size")
    lineage = getattr(model, "lineage", {})
    if require_new:
        _validate_lineage(lineage, require_complete=True)
        known = {key: source for split in ("train", "validation", "test")
                 for key, source in lineage.get(split, {}).items()}
        fingerprints = lineage.get("canonical_volume_fingerprints", [])
        if len(set(fingerprints)) != len(known):
            raise ValueError("Complete canonical volume content fingerprints are required for new prediction")
        if patient_id in known or source_patient_id in known.values():
            raise ValueError("New prediction patients must be outside all training/evaluation splits")
        fingerprint = volume_fingerprint(volume)
        if fingerprint in fingerprints:
            raise ValueError("Prediction volume duplicates a canonical training/evaluation patient")
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(volume), batch_size):
            images = torch.stack([preprocess_slice(volume, z, config)[0]
                                  for z in range(start, min(start + batch_size, len(volume)))])
            predictions.append(model(images.to(device)).argmax(1).cpu().numpy().astype(np.uint8))
    mask = np.concatenate(predictions)
    return {"patient_id": patient_id, "source_patient_id": source_patient_id,
            "prediction_context": "new_patient" if require_new else "held_out_case_review",
            "task": "kidney_tumour_segmentation", "predicted_mask": mask,
            "slice_indices": list(range(len(volume))), "coordinate_space": "resized_axial_%dx%d" % (config.size, config.size),
            "tumour_pixels": int((mask == 2).sum()),
            "educational_message": "Predicted image regions are an educational model output, not a patient diagnosis."}


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False))
