#!/usr/bin/env python3
"""Step 3: inspect at most two pure-2D samples from one development patient.

Run from the extracted starter:
python learning/03_check_2d_loader.py --root ../kits_project --case case_00000

No model, optimizer, training loop, cohort split or evaluation is executed.
Only a small JSON report is written, outside the original data project.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import tempfile


# 1. Import the reusable adapter from the backend/kidney package root.
def load_adapter():
    package_root = Path(__file__).resolve().parents[1]
    adapter = package_root / "kidney_segmentation/png_dataset.py"
    if not adapter.is_file():
        raise ValueError("Cannot find kidney_segmentation/png_dataset.py under backend/kidney/.")
    sys.path.insert(0, str(package_root))
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from kidney_segmentation.png_dataset import PNGConfig, PNGPatientDataset
    except ImportError as error:
        raise ValueError("Use the starter's Python environment and install learning/requirements-step3.txt first: %s" % error) from None
    return torch, DataLoader, Subset, PNGConfig, PNGPatientDataset


# 2. Keep the report outside kits_project and reject symlinked output locations.
def checked_output_dir(requested, root):
    output = Path(os.path.abspath(str(requested.expanduser())))
    for part in [output, *output.parents]:
        if part.is_symlink():
            raise ValueError("The output directory and its parents must not be symlinks: %s" % part)
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Save --output-dir outside kits_project; original data must remain unchanged.")
    if output.exists() and not output.is_dir():
        raise ValueError("The output directory is an existing file: %s" % output)
    destination = output / "loader_report.json"
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError("The report destination must be a regular file, not a symlink or directory.")
    return output, destination


# 3. Select explicit development examples without creating or reusing a split.
def select_samples(dataset, data_root, maximum):
    present = []
    for index, row in enumerate(dataset.rows):
        image = (data_root / row["image"]).resolve()
        mask = (data_root / row["mask"]).resolve()
        if data_root not in image.parents or data_root not in mask.parents:
            raise ValueError("A selected patient asset resolves outside data2d.")
        if image.is_file() and mask.is_file():
            present.append(index)
    if not present:
        raise ValueError("No existing image/mask pair was found for the explicit development patient.")
    # z294 was inspected in Step 2. Other examples are ordered by z, not by labels.
    present.sort(key=lambda index: (dataset.rows[index]["z"] != 294, dataset.rows[index]["z"]))
    return present[:maximum], len(present)


def tensor_statistics(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "minimum": float(tensor.min().item()), "maximum": float(tensor.max().item())}


# 4. Let DataLoader stack adapter samples; never use the mask as an input channel.
def inspect(args):
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The --root kits_project directory does not exist: %s" % root)
    if not re.fullmatch(r"case_[0-9]{5}", args.case):
        raise ValueError("Use one explicit development patient ID, such as case_00000.")
    data_root = (root / "data2d").resolve()
    if root not in data_root.parents or not data_root.is_dir():
        raise ValueError("The data2d directory must exist inside kits_project.")
    output, destination = checked_output_dir(args.output_dir, root)
    torch, DataLoader, Subset, PNGConfig, PNGPatientDataset = load_adapter()
    config = PNGConfig(input_channels=1)
    dataset = PNGPatientDataset(data_root / "manifest.csv", patient_ids=[args.case],
                               data_root=data_root, config=config)
    indices, present_pairs = select_samples(dataset, data_root, args.max_samples)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    batches = []
    print("\n[PURE 2D DEVELOPMENT DATA LOADER]")
    print("Patient: %s; inspected samples: %d; patient manifest records: %d; existing pairs: %d" %
          (args.case, len(indices), len(dataset), present_pairs))
    print("Each input reads one grayscale PNG. No neighbouring slices, raw HU or reference-mask input channels are used.")
    print("Normalization: (PNG / 255.0 - %.3f) / %.3f" % (config.mean, config.std))

    for batch_number, batch in enumerate(loader, start=1):
        inputs, targets = batch["input"], batch["target"]
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 256, 256) or inputs.dtype != torch.float32:
            raise ValueError("Expected input [B,1,256,256] with dtype torch.float32.")
        if targets.ndim != 3 or tuple(targets.shape[1:]) != (256, 256) or targets.dtype != torch.int64:
            raise ValueError("Expected target [B,256,256] with dtype torch.int64.")
        if len(inputs) != len(targets) or not torch.isfinite(inputs).all():
            raise ValueError("Input/target batch sizes must match and input values must be finite.")
        unique_labels = [int(value) for value in torch.unique(targets).tolist()]
        if not set(unique_labels).issubset({0, 1, 2}):
            raise ValueError("Target class IDs must be a subset of 0/1/2.")
        samples = []
        for offset, patient_id in enumerate(batch["patient_id"]):
            if patient_id != args.case:
                raise ValueError("The development batch contains a different patient.")
            sample = {
                "patient_id": patient_id, "slice_index": int(batch["slice_index"][offset].item()),
                "source_paths": {key: values[offset] for key, values in batch["source_paths"].items()},
                "source_sha256": {key: values[offset] for key, values in batch["source_sha256"].items()},
                "manifest_sha256": batch["manifest_sha256"][offset],
                "input_statistics": tensor_statistics(inputs[offset]),
                "target_unique_class_ids": [int(value) for value in torch.unique(targets[offset]).tolist()],
            }
            samples.append(sample)
        record = {"batch_number": batch_number, "input": tensor_statistics(inputs),
                  "target": {"shape": list(targets.shape), "dtype": str(targets.dtype),
                             "unique_class_ids": unique_labels}, "samples": samples}
        batches.append(record)
        print("\n[BATCH %d]" % batch_number)
        print("input shape=%s; dtype=%s; normalized range=%.6f..%.6f" %
              (tuple(inputs.shape), inputs.dtype, inputs.min().item(), inputs.max().item()))
        print("target shape=%s; dtype=%s; unique class IDs=%s" %
              (tuple(targets.shape), targets.dtype, unique_labels))
        for sample in samples:
            print("patient_id=%s; original slice_index=%d" % (sample["patient_id"], sample["slice_index"]))

    # 5. Save metadata and checksums only: no image files or tensor arrays.
    report = {
        "step": "03_check_2d_loader", "status": "completed", "task": "pure_2d_segmentation_input_inspection",
        "model_executed": False, "model_trained": False, "contains_model_prediction": False,
        "formal_cohort_selected": False, "split_file_read": False, "clinical_performance_evidence": False,
        "patient_id": args.case, "manifest_path": str((data_root / "manifest.csv").resolve()),
        "manifest_sha256": dataset.manifest_sha256,
        "normalization_config": config.to_dict(), "config_sha256": dataset.config_sha256,
        "input_formula": "(PNG_uint8 / 255.0 - mean) / std",
        "target_policy": "Unchanged integer class IDs: 0=background, 1=kidney, 2=tumour; masks are targets only.",
        "sampling": {"patient_manifest_records": len(dataset), "existing_image_mask_pairs": present_pairs,
                     "inspected_sample_count": len(indices), "maximum_samples": args.max_samples,
                     "selection": "Prefer existing z294, then other available rows of the same patient ordered by z; no label-based selection.",
                     "patient_dataset_indices": indices, "batch_size": args.batch_size,
                     "shuffle": False, "num_workers": 0, "augmentation": False},
        "batches": batches,
        "development_exposure": {"case_inspected_for_development": args.case,
                                 "recommended_future_role": "training_patient_or_outside_formal_cohort",
                                 "must_not_be_presented_as_unseen_final_test": True,
                                 "must_not_be_presented_as_independent_prediction": True},
        "coverage_note": "At most two existing manifest-selected PNG pairs were inspected. This is not complete-patient validation or full-scan evaluation.",
        "next_step_note": "A future pure-2D model pass needs one-channel model input adaptation; do not duplicate this PNG into three channels or call it 2.5D.",
    }
    output.mkdir(parents=True, exist_ok=True)
    # Atomic replacement avoids leaving a partial report after interruption.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, suffix=".json.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        if destination.is_symlink():
            raise ValueError("The report destination became a symlink; nothing will be written to its target.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print("\nMetadata-only report: %s" % destination)
    print("No model was run or trained. No CT, mask, manifest or split file was modified.")
    print("This patient is exposed to development: later place it in training or outside the formal cohort.")
    print("Do not present it as unseen final-test or independent new-patient evidence.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 3: inspect a pure-2D tensor batch from one explicit development patient without training")
    parser.add_argument("--root", type=Path, default=Path("../kits_project"), help="Original kits_project directory (default: ../kits_project)")
    parser.add_argument("--case", default="case_00000", help="One explicit development patient (default: case_00000)")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2, help="Samples per inspected batch (1 or 2; default: 2)")
    parser.add_argument("--max-samples", type=int, choices=(1, 2), default=2, help="Maximum development samples to inspect (1 or 2; default: 2)")
    parser.add_argument("--output-dir", type=Path, default=Path("learning_outputs/step3"), help="Metadata-report directory outside kits_project")
    args = parser.parse_args(argv)
    try:
        inspect(args)
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
