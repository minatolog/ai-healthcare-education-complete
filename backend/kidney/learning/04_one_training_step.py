#!/usr/bin/env python3
"""Step 4: one verified-pretrained pure-2D forward/backward/optimizer step.

This is a development exercise on one explicitly selected patient, not formal
training, validation, test evaluation or an independent new-patient prediction.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

SCOPE = "training_patient_only_not_validated_model"
FILES = ("learning_step.pt", "training_step.json", "reference_overlay.png",
         "prediction_before_overlay.png", "prediction_after_overlay.png", "comparison_preview.png")


# 1. Reuse Step 3's patient/sample selection and import the source model package.
def load_runtime():
    starter = Path(__file__).resolve().parents[1]
    step3_path = starter / "learning/03_check_2d_loader.py"
    if not step3_path.is_file():
        raise ValueError("Keep learning/03_check_2d_loader.py from Step 3 in this starter.")
    spec = importlib.util.spec_from_file_location("step3_reusable_helpers", step3_path)
    step3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step3)
    torch, DataLoader, Subset, PNGConfig, PNGPatientDataset = step3.load_adapter()
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from kidney_segmentation.model import segmentation_loss
        from kidney_segmentation.pretrained import build_verified_model
    except ImportError as error:
        raise ValueError("Use the starter's Python environment and install learning/requirements-step4.txt: %s" % error) from None
    return step3, torch, DataLoader, Subset, PNGConfig, PNGPatientDataset, np, Image, ImageDraw, ImageFont, segmentation_loss, build_verified_model


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_output_dir(requested, root):
    output = Path(os.path.abspath(str(requested.expanduser())))
    for part in [output, *output.parents]:
        if part.is_symlink():
            raise ValueError("The output directory and its parents must not be symlinks: %s" % part)
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Save --output-dir outside kits_project; original data must remain unchanged.")
    if output.exists() and not output.is_dir():
        raise ValueError("The output directory is an existing file.")
    for name in FILES:
        path = output / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("Output %s must not be a symlink or directory." % name)
    return output


# 2. Report the actual combined loss and its two scalar components.
def loss_parts(torch, segmentation_loss, logits, targets):
    total = segmentation_loss(logits, targets)
    cross_entropy = torch.nn.functional.cross_entropy(logits, targets)
    foreground_soft_dice = total - cross_entropy
    if not all(bool(torch.isfinite(value)) for value in (total, cross_entropy, foreground_soft_dice)):
        raise ValueError("A loss component became non-finite; no optimizer update will be saved.")
    values = {"total": float(total.detach()), "cross_entropy": float(cross_entropy.detach()),
              "foreground_soft_dice": float(foreground_soft_dice.detach())}
    return total, values


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update((name + "|" + str(value.dtype) + "|" + str(tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def batch_sources(batch):
    return [{"patient_id": patient, "slice_index": int(batch["slice_index"][index].item()),
             "source_paths": {key: values[index] for key, values in batch["source_paths"].items()},
             "source_sha256": {key: values[index] for key, values in batch["source_sha256"].items()},
             "manifest_sha256": batch["manifest_sha256"][index]}
            for index, patient in enumerate(batch["patient_id"])]


# 3. Display separate reference and predicted regions without changing labels.
def overlay_image(np, Image, ct, labels, alpha=0.4):
    if ct.shape != labels.shape or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Display overlays require aligned 256x256 class IDs.")
    rgb = np.repeat(ct[..., None], 3, axis=2)
    colors = np.array([[0, 0, 0], [50, 200, 100], [255, 80, 80]], dtype=np.float32)
    active = labels > 0
    blended = rgb.astype(np.float32)
    blended[active] = (1 - alpha) * blended[active] + alpha * colors[labels[active]]
    return Image.fromarray(blended.round().astype(np.uint8))


def labelled_preview(Image, ImageDraw, ImageFont, ct_image, reference, before, after, sample):
    margin, size = 16, 256
    canvas = Image.new("RGB", (4 * size + 5 * margin, 434), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = ImageFont.load_default(size=17), ImageFont.load_default(size=13)
    draw.text((margin, 13), "Development patient only | Exactly one optimizer step | Not a validated model", fill="black", font=title_font)
    draw.text((margin, 42), "Patient: %s | Original z: %d | Green: kidney | Red: tumour" %
              (sample["patient_id"], sample["slice_index"]), fill="black", font=label_font)
    panels = [("CT input (processed PNG)", ct_image), ("REFERENCE annotation", reference),
              ("MODEL before update", before), ("MODEL after 1 update", after)]
    for index, (title, image) in enumerate(panels):
        x = margin + index * (size + margin)
        draw.text((x, 77), title, fill="black", font=label_font)
        canvas.paste(image, (x, 105))
    draw.text((margin, 379), "Reference colours are existing labels; model colours are argmax predictions on the SAME development sample.", fill="black", font=label_font)
    draw.text((margin, 406), "No validation/test metrics, disease confidence or physical tumour volume is reported.", fill="black", font=label_font)
    return canvas


def atomic_json(path, payload):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".json.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        if path.is_symlink():
            raise ValueError("The report destination became a symlink.")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(args):
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive and finite.")
    if not 0 <= args.seed < 2**63:
        raise ValueError("--seed must be a non-negative integer below 2**63.")
    root = args.root.expanduser().resolve()
    if not root.is_dir() or not re.fullmatch(r"case_[0-9]{5}", args.case):
        raise ValueError("Use an existing kits_project root and one explicit development case_XXXXX ID.")
    data_root = (root / "data2d").resolve()
    if root not in data_root.parents:
        raise ValueError("data2d must remain inside kits_project.")
    output = checked_output_dir(args.output_dir, root)
    step3, torch, DataLoader, Subset, PNGConfig, PNGPatientDataset, np, Image, ImageDraw, ImageFont, segmentation_loss, build_verified_model = load_runtime()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    config = PNGConfig(input_channels=1)
    dataset = PNGPatientDataset(data_root / "manifest.csv", patient_ids=[args.case], data_root=data_root, config=config)
    indices, _ = step3.select_samples(dataset, data_root, args.batch_size)
    if len(indices) != args.batch_size:
        raise ValueError("Not enough existing pairs for the requested single development batch.")
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    batch = next(iter(loader))
    inputs, targets = batch["input"], batch["target"]
    samples = batch_sources(batch)
    if tuple(inputs.shape) != (args.batch_size, 1, 256, 256) or inputs.dtype != torch.float32:
        raise ValueError("Expected float32 input [B,1,256,256].")
    if tuple(targets.shape) != (args.batch_size, 256, 256) or targets.dtype != torch.int64:
        raise ValueError("Expected int64 target [B,256,256].")
    if not torch.isfinite(inputs).all() or any(sample["patient_id"] != args.case for sample in samples):
        raise ValueError("Inputs must be finite and all batch samples must belong to the explicit development patient.")

    # This helper loads and verifies pinned cached ImageNet weights offline.
    # A missing or unverified cache is an error; no random encoder fallback exists.
    torch.manual_seed(args.seed)
    model, pretrained_proof = build_verified_model(input_channels=1)
    json.dumps(pretrained_proof, allow_nan=False)
    if not model.pretrained or model.input_channels != 1:
        raise ValueError("A verified ImageNet encoder with one input channel is required.")
    if any(parameter.requires_grad for parameter in model.network.encoder.parameters()):
        raise ValueError("The encoder must remain frozen for this learning step.")
    model = model.cpu()
    encoder_before = {name: value.detach().clone() for name, value in model.network.encoder.state_dict().items()}
    trainable_before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    if not trainable_before or any(not name.startswith(("network.decoder.", "network.segmentation_head.")) for name in trainable_before):
        raise ValueError("Only decoder and segmentation-head parameters may be trainable.")

    # 4. Measure before/after on the SAME batch in the SAME eval mode.
    # eval() is a module setting here, not held-out model evaluation.
    model.eval()
    with torch.no_grad():
        before_logits = model(inputs)
        _, before_losses = loss_parts(torch, segmentation_loss, before_logits, targets)
        before_labels = before_logits[0].argmax(0).cpu().numpy().astype(np.uint8)
    if tuple(before_logits.shape) != (args.batch_size, 3, 256, 256) or not torch.isfinite(before_logits).all():
        raise ValueError("Expected finite logits [B,3,256,256].")

    # 5. Exactly ONE backward pass and ONE optimizer.step().
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate)
    model.train()
    if model.network.encoder.training or any(module.training for module in model.network.encoder.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
        raise ValueError("Frozen encoder BatchNorm must remain in eval mode.")
    optimizer.zero_grad(set_to_none=True)
    training_logits = model(inputs)
    training_loss, training_losses = loss_parts(torch, segmentation_loss, training_logits, targets)
    training_loss.backward()
    gradient_names, nonzero_elements, squared_norm = [], 0, 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise ValueError("A frozen encoder parameter unexpectedly received a gradient.")
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("A trainable parameter has missing or non-finite gradients: %s" % name)
        count = int(torch.count_nonzero(parameter.grad).item())
        if count:
            gradient_names.append(name)
            nonzero_elements += count
        squared_norm += float(parameter.grad.norm().item()) ** 2
    if not gradient_names or not math.isfinite(squared_norm) or squared_norm <= 0:
        raise ValueError("Finite nonzero decoder/head gradients are required before an update.")
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Updated model state became non-finite; no checkpoint will be saved.")
    encoder_after = model.network.encoder.state_dict()
    encoder_unchanged = all(torch.equal(value, encoder_after[name]) for name, value in encoder_before.items())
    changed_parameters = [name for name, value in model.named_parameters() if value.requires_grad and not torch.equal(value.detach(), trainable_before[name])]
    if not encoder_unchanged or not changed_parameters:
        raise ValueError("Expected the entire encoder state unchanged and decoder/head parameters changed.")
    model.eval()
    with torch.no_grad():
        after_logits = model(inputs)
        _, after_losses = loss_parts(torch, segmentation_loss, after_logits, targets)
        after_labels = after_logits[0].argmax(0).cpu().numpy().astype(np.uint8)
    if not torch.isfinite(after_logits).all():
        raise ValueError("Updated logits became non-finite; no checkpoint will be saved.")

    # 6. All artifacts refer to this development batch, never unseen patients.
    first = samples[0]
    image_path = Path(first["source_paths"]["image"])
    for sample in samples:
        for asset, expected_hash in sample["source_sha256"].items():
            if sha256_file(sample["source_paths"][asset]) != expected_hash:
                raise ValueError("A source CT or mask changed after loading; learning artifacts cannot be saved.")
    if sha256_file(dataset.manifest_path) != dataset.manifest_sha256:
        raise ValueError("The manifest changed after loading.")
    with Image.open(image_path) as image:
        ct = np.array(image, dtype=np.uint8)
    reference_labels = targets[0].cpu().numpy().astype(np.uint8)
    reference = overlay_image(np, Image, ct, reference_labels)
    before_overlay = overlay_image(np, Image, ct, before_labels)
    after_overlay = overlay_image(np, Image, ct, after_labels)
    preview = labelled_preview(Image, ImageDraw, ImageFont, Image.fromarray(ct).convert("RGB"), reference, before_overlay, after_overlay, first)
    metadata = {
        "scope": SCOPE, "formal_training_completed": False, "model_validated": False,
        "clinical_performance_evidence": False, "optimizer_steps": 1,
        "architecture": model.architecture, "input_channels": 1, "input_mode": model.input_mode,
        "output_classes": {"0": "background", "1": "kidney", "2": "tumour"},
        "png_preprocessing": config.to_dict(), "config_sha256": dataset.config_sha256,
        "manifest_sha256": dataset.manifest_sha256, "development_samples": samples,
        "pretrained_encoder_proof": pretrained_proof,
        "decoder_initialization": "random_initialization_then_exactly_one_Adam_update",
        "seed": args.seed, "device": "cpu", "cpu_threads": 2, "learning_rate": args.learning_rate,
        "encoder_frozen": True, "encoder_batchnorm_eval": True,
        "development_exposure": {"patient_id": args.case, "future_role": "training_patient_or_outside_formal_cohort",
                                 "must_not_be_presented_as_unseen_final_test": True,
                                 "must_not_be_presented_as_independent_prediction": True},
        "formal_cli_compatible": False,
    }
    report = {
        "step": "04_one_training_step", "status": "completed", "scope": SCOPE,
        "formal_cohort_selected": False, "held_out_validation_or_test_performed": False,
        "performance_metrics_reported": False, "contains_development_model_predictions": True,
        "input_shape": list(inputs.shape), "input_dtype": str(inputs.dtype),
        "input_range": [float(inputs.min()), float(inputs.max())],
        "target_shape": list(targets.shape), "target_dtype": str(targets.dtype),
        "target_unique_class_ids": [int(value) for value in torch.unique(targets).tolist()],
        "logits_shape": list(before_logits.shape),
        "loss_before_same_batch_eval_mode": before_losses,
        "loss_used_for_backward_train_mode": training_losses,
        "loss_after_same_batch_eval_mode": after_losses,
        "loss_comparison_note": "Before/after use the same development batch and eval mode; decoder BatchNorm buffers also update during the training pass. A loss decrease is not guaranteed and is not generalization evidence.",
        "update_checks": {"optimizer_steps": 1, "backward_passes": 1, "all_gradients_finite": True,
                          "nonzero_gradient_elements": nonzero_elements, "gradient_l2_norm": math.sqrt(squared_norm),
                          "encoder_parameters_and_buffers_unchanged": encoder_unchanged,
                          "encoder_state_sha256_before": state_sha256(encoder_before),
                          "encoder_state_sha256_after": state_sha256(encoder_after),
                          "changed_decoder_or_head_parameters": changed_parameters},
        "checkpoint_metadata": metadata,
        "visualization_identity": {"patient_id": first["patient_id"], "slice_index": first["slice_index"],
                                   "source_ct_sha256": first["source_sha256"]["image"],
                                   "source_mask_sha256": first["source_sha256"]["mask"],
                                   "coordinate_space": "processed_png_256x256", "reference_type": "ground_truth_annotation",
                                   "before_type": "same_development_sample_prediction_before_update",
                                   "after_type": "same_development_sample_prediction_after_one_update",
                                   "model_stage": SCOPE, "overlay_alpha": 0.4},
        "output_files": list(FILES),
    }
    # Re-check all output destinations immediately before writing.
    output = checked_output_dir(args.output_dir, root)
    output.mkdir(parents=True, exist_ok=True)
    reference.save(output / "reference_overlay.png")
    before_overlay.save(output / "prediction_before_overlay.png")
    after_overlay.save(output / "prediction_after_overlay.png")
    preview.save(output / "comparison_preview.png")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, suffix=".pt.tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save({"schema_version": "learning_step4_v1", "metadata": metadata,
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}}, temporary)
        if (output / "learning_step.pt").is_symlink():
            raise ValueError("The checkpoint destination became a symlink.")
        temporary.replace(output / "learning_step.pt")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report["checkpoint_sha256"] = sha256_file(output / "learning_step.pt")
    report["visualization_identity"]["checkpoint_sha256"] = report["checkpoint_sha256"]
    report["output_sha256"] = {name: sha256_file(output / name) for name in FILES if name != "training_step.json"}
    atomic_json(output / "training_step.json", report)

    print("\n[ONE PURE-2D DEVELOPMENT UPDATE]")
    print("input shape=%s; dtype=%s" % (tuple(inputs.shape), inputs.dtype))
    print("logits shape=%s; classes=background/kidney/tumour" % (tuple(before_logits.shape),))
    for sample in samples:
        print("patient_id=%s; original slice_index=%d" % (sample["patient_id"], sample["slice_index"]))
    for label, values in [("Before (same batch, eval mode)", before_losses), ("Backward loss (train mode)", training_losses), ("After (same batch, eval mode)", after_losses)]:
        print("%s: CE=%.6f; foreground soft-Dice=%.6f; total=%.6f" %
              (label, values["cross_entropy"], values["foreground_soft_dice"], values["total"]))
    print("optimizer steps=1; finite nonzero gradients=%d elements; encoder state unchanged=%s; changed decoder/head parameter tensors=%d" %
          (nonzero_elements, encoder_unchanged, len(changed_parameters)))
    print("Learning artifacts: %s" % output)
    print("Checkpoint scope: %s; not accepted by the formal model CLI." % SCOPE)
    print("A loss decrease is not guaranteed. No formal training, validation/test evaluation or confidence score is reported.")
    print("No CT, mask, manifest or split file was modified.")
    print("Each run starts a fresh seed-controlled decoder; this command does not resume the saved checkpoint.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Step 4: exactly one pretrained pure-2D decoder/head update on a development patient's PNG batch")
    parser.add_argument("--root", type=Path, default=Path("../kits_project"), help="Original kits_project directory")
    parser.add_argument("--case", default="case_00000", help="Explicit development patient, never an unseen holdout")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2, help="One development batch of 1 or 2 samples")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Positive finite Adam learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=42, help="Decoder initialization and loader seed (default: 42)")
    parser.add_argument("--output-dir", type=Path, default=Path("learning_outputs/step4"), help="Learning artifacts outside kits_project")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
