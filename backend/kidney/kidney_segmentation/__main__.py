"""Run from the package directory: python -m kidney_segmentation --help."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .data import (FullSliceDataset, file_hash, load_patient, manifest_hash,
                   synthetic_patients, validate_real_training, volume_fingerprint)
from .engine import (evaluate_dataset, load_checkpoint, predict_volume, save_checkpoint,
                     train_model, write_json)
from .overlays import save_overlays
from .preprocessing import PreprocessConfig


def canonical_lineage(manifest, manifest_path, audit_path):
    return {**{split: {row["patient_id"]: row["source_patient_id"] for row in manifest["patients"] if row["split"] == split}
               for split in ("train", "validation", "test", "prediction")},
            "manifest_sha256": manifest_hash(manifest_path), "audit_sha256": file_hash(audit_path),
            "canonical_volume_fingerprints": [row["volume_fingerprint"] for row in manifest["patients"]
                                               if row["split"] != "prediction"]}


def real_context(args, metadata=None):
    manifest, root = validate_real_training(args.manifest, args.audit, args.data_root)
    lineage = canonical_lineage(manifest, args.manifest, args.audit)
    if metadata is not None and metadata["lineage"] != lineage:
        raise ValueError("Checkpoint does not match this audited canonical cohort")
    return manifest, root, lineage


def write_prediction(out, patient, model, config, metadata, include_reference=False):
    prediction = predict_volume(model, patient.volume, patient.patient_id, patient.source_patient_id, config)
    mask = prediction.pop("predicted_mask")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "predicted_segmentation.npy", mask, allow_pickle=False)
    assets = save_overlays(out / "images", patient.volume, mask, patient.patient_id, config,
                           ground_truth=patient.mask if include_reference else None)
    prediction.update(predicted_mask_file="predicted_segmentation.npy", data_mode=metadata["data_mode"],
                      clinical_performance_evidence=metadata["clinical_performance_evidence"],
                      model_architecture=metadata["architecture"], visualization=assets,
                      checkpoint_sha256=model.checkpoint_sha256, model_version=model.checkpoint_sha256)
    write_json(out / "prediction.json", prediction)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kidney Tumour 2.5D segmentation baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("synthetic-smoke", help="Random encoder, synthetic fixtures; engineering evidence only")
    smoke.add_argument("--out", type=Path, required=True)
    smoke.add_argument("--epochs", type=int, default=2)
    for command in ("train", "evaluate", "infer"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", type=Path, required=True)
        item.add_argument("--audit", type=Path, required=True)
        item.add_argument("--data-root", type=Path)
        item.add_argument("--out", type=Path, required=True)
        if command == "train":
            item.add_argument("--epochs", type=int, default=5)
            item.add_argument("--batch-size", type=int, default=4)
        else:
            item.add_argument("--checkpoint", type=Path, required=True)
        if command == "evaluate":
            item.add_argument("--split", choices=("validation", "test"), required=True)
        if command == "infer":
            item.add_argument("--patient-id", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.command == "synthetic-smoke":
        patients = synthetic_patients()
        config = PreprocessConfig(size=64)
        groups = {split: [patient for patient in patients if patient.split == split]
                  for split in ("train", "validation", "test", "prediction")}
        lineage = {split: {patient.patient_id: patient.source_patient_id for patient in group}
                   for split, group in groups.items()}
        lineage["canonical_volume_fingerprints"] = [volume_fingerprint(patient.volume) for patient in patients
                                                     if patient.split != "prediction"]
        train = FullSliceDataset(groups["train"], config=config)
        validation = FullSliceDataset(groups["validation"], config=config)
        model, metadata = train_model(train, validation, epochs=args.epochs, batch_size=4, pretrained=False,
                                      lineage=lineage, data_mode="synthetic_demo")
        save_checkpoint(args.out / "synthetic_model.pt", model, metadata)
        loaded, config, metadata = load_checkpoint(args.out / "synthetic_model.pt")
        test = evaluate_dataset(loaded, FullSliceDataset(groups["test"], config=config), expected_split="test",
                                expected_patient_ids=set(lineage["test"]))
        write_json(args.out / "smoke_report.json", {"data_mode": metadata["data_mode"], "clinical_performance_evidence": False,
                                                    "encoder_weights": "random_engineering_test_only",
                                                    "history": metadata["history"], "test": test})
        write_prediction(args.out / "prediction", groups["prediction"][0], loaded, config, metadata)
        print("Synthetic segmentation smoke completed; not medical performance evidence.")
        return
    if args.command == "train":
        manifest, root, lineage = real_context(args)
        config = PreprocessConfig()
        training = FullSliceDataset([row for row in manifest["patients"] if row["split"] == "train"], root, config)
        validation = FullSliceDataset([row for row in manifest["patients"] if row["split"] == "validation"], root, config)
        model, metadata = train_model(training, validation, epochs=args.epochs, batch_size=args.batch_size,
                                      pretrained=True, lineage=lineage,
                                      progress=lambda event: print("Epoch %d/%d: validation loss %.6f" %
                                                                   (event["epoch"], event["epochs"], event["validation_loss"])))
        save_checkpoint(args.out / "kidney_model.pt", model, metadata)
        write_json(args.out / "training.json", metadata)
        print("Saved audited Kidney segmentation baseline.")
        return
    model, config, metadata = load_checkpoint(args.checkpoint)
    if metadata["data_mode"] != "real":
        raise ValueError("Real-data commands cannot use a synthetic engineering checkpoint")
    manifest, root, lineage = real_context(args, metadata)
    if args.command == "evaluate":
        output = args.out / (args.split + "_evaluation.json")
        if output.exists():
            raise ValueError("Evaluation output exists; preserve the locked benchmark result")
        dataset = FullSliceDataset([row for row in manifest["patients"] if row["split"] == args.split], root, config)
        result = evaluate_dataset(model, dataset, expected_split=args.split, expected_patient_ids=set(lineage[args.split]))
        result.update(data_mode="real", manifest_sha256=lineage["manifest_sha256"], audit_sha256=lineage["audit_sha256"],
                      checkpoint_sha256=model.checkpoint_sha256, model_version=model.checkpoint_sha256)
        write_json(output, result)
        print("Saved patient-level segmentation evaluation.")
    else:
        row = next((row for row in manifest["patients"] if row["patient_id"] == args.patient_id), None)
        if row is None or row["split"] != "prediction":
            raise ValueError("Select a registered prediction-only patient outside the canonical 40")
        patient = load_patient(row, root, supervised=False)
        write_prediction(args.out, patient, model, config, metadata)
        print("Saved new-patient predicted regions and separate display assets.")


if __name__ == "__main__":
    main()
