"""Strict offline ImageNet encoder initialization from one pinned SMP snapshot.

This helper never calls HF Hub download or SMP's remote pretrained constructor.
It builds the architecture without weights, replaces the entire encoder with
verified cached weights, proves the result, and only then returns the model.
The decoder and segmentation head keep their ordinary random initialization.
"""
from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import KidneyUNet

REPOSITORY = "smp-hub/resnet18.imagenet"
REVISION = "3f2325ff978283d47aa6a1d6878ca20565622683"
WEIGHTS_SHA256 = "b1d1d36f298c29c07afa9d5be76d888b01d174824177026613bbd64a26612d19"
CONFIG_SHA256 = "01bf2cf24eb29b405c28c159f46ceda92c098ab85868be88fb100967db47166e"
EXPECTED_CONFIG = {
    "input_space": "RGB", "input_size": [3, 224, 224], "input_range": [0, 1],
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "num_classes": 1000,
}


class PretrainedVerificationError(RuntimeError):
    """Cached pretrained evidence is missing or does not match the pinned model."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_cache_root() -> Path:
    # Resolve the standard HF cache layout without importing its network client.
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "huggingface" / "hub"


def _verified_cache(cache_root=None):
    root = Path(cache_root).expanduser() if cache_root is not None else _default_cache_root()
    snapshot = root / "models--smp-hub--resnet18.imagenet" / "snapshots" / REVISION
    weights_path, config_path = snapshot / "model.safetensors", snapshot / "config.json"
    for path, expected in ((config_path, CONFIG_SHA256), (weights_path, WEIGHTS_SHA256)):
        if not path.is_file():
            raise PretrainedVerificationError(
                "Verified offline ResNet-18 cache is missing %s for pinned revision %s. "
                "No download or random encoder fallback was attempted." % (path.name, REVISION))
        if _sha256(path) != expected:
            raise PretrainedVerificationError(
                "Offline ResNet-18 %s failed the pinned SHA-256 check. "
                "No model was returned; no download or random encoder fallback was attempted." % path.name)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PretrainedVerificationError("Pinned encoder config could not be decoded.") from error
    if config != EXPECTED_CONFIG:
        raise PretrainedVerificationError("Pinned config is not the expected RGB ImageNet-1000 source.")
    return snapshot, weights_path, config


def _load_and_prove_encoder(model, source, input_channels):
    """Strictly replace every encoder tensor; return exact equality evidence."""
    if (set(source) & {"fc.weight", "fc.bias"}) != {"fc.weight", "fc.bias"}:
        raise PretrainedVerificationError("ImageNet checkpoint classification-head evidence is missing.")
    if (tuple(source["fc.weight"].shape), tuple(source["fc.bias"].shape)) != ((1000, 512), (1000,)):
        raise PretrainedVerificationError("ImageNet checkpoint is not the expected ResNet-18 source.")
    if "conv1.weight" not in source or tuple(source["conv1.weight"].shape) != (64, 3, 7, 7):
        raise PretrainedVerificationError("ImageNet checkpoint has an unexpected first convolution.")
    encoder = model.network.encoder
    live = encoder.state_dict()
    official = {key: value for key, value in source.items() if not key.startswith("fc.")}
    # The published safetensors source has no BatchNorm num_batches_tracked
    # counters. These are integer bookkeeping buffers, not learned weights.
    counters = {key for key in live if key.endswith(".num_batches_tracked")}
    if set(official) != set(live) - counters:
        raise PretrainedVerificationError("Official and instantiated encoder state keys do not match exactly.")
    expected = dict(official)
    if input_channels == 1:
        expected["conv1.weight"] = source["conv1.weight"].sum(dim=1, keepdim=True)
    for key in counters:
        if live[key].dtype != torch.int64 or live[key].numel() != 1:
            raise PretrainedVerificationError("Unexpected BatchNorm counter format.")
        expected[key] = torch.zeros_like(live[key])
    for key, value in expected.items():
        if value.shape != live[key].shape or value.dtype != live[key].dtype:
            raise PretrainedVerificationError("Encoder tensor shape or dtype mismatch: %s." % key)
    try:
        encoder.load_state_dict(expected, strict=True)
    except Exception as error:
        raise PretrainedVerificationError("Strict offline encoder loading failed; no model was returned.") from error
    loaded = encoder.state_dict()
    if any(not torch.equal(loaded[key], value) for key, value in expected.items()):
        raise PretrainedVerificationError("Loaded encoder tensors do not exactly match the verified source.")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    batch_norms = [module for module in encoder.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    if not batch_norms or any(module.training for module in batch_norms):
        raise PretrainedVerificationError("Encoder BatchNorm must remain in evaluation mode.")
    return {
        "all_official_encoder_tensors_exact": True,
        "official_encoder_tensor_count": len(official),
        "encoder_state_tensor_count": len(loaded),
        "initialized_zero_batch_norm_counter_count": len(counters),
        "batch_norm_counter_policy": "Source omits counters; initialize integer bookkeeping buffers to zero.",
        "first_conv_source_shape": [64, 3, 7, 7],
        "first_conv_loaded_shape": list(loaded["conv1.weight"].shape),
        "first_conv_adaptation": "sum_rgb_weights" if input_channels == 1 else "unchanged_rgb_weights",
        "adapted_first_conv_equals_rgb_sum": input_channels == 1,
        "encoder_parameter_tensor_count": len(list(encoder.parameters())),
        "encoder_frozen": all(not p.requires_grad for p in encoder.parameters()),
        "encoder_batch_norm_eval": True,
        "ignored_imagenet_head_keys": ["fc.bias", "fc.weight"],
    }


def build_verified_model(input_channels=1, cache_root=None):
    """Return ``(model, proof)`` using cached official weights, or fail closed.

    ``cache_root`` is the HF Hub cache root, containing the ``models--...``
    directory. Its default respects HF_HUB_CACHE/HF_HOME/XDG_CACHE_HOME.
    No environment setting or network access is required by this helper.
    """
    if type(input_channels) is not int or input_channels not in (1, 3):
        raise ValueError("Verified encoder supports integer input_channels=1 or 3.")
    if version("segmentation-models-pytorch") != "0.5.0":
        raise PretrainedVerificationError("This verified loader requires segmentation-models-pytorch 0.5.0.")
    snapshot, weights_path, config = _verified_cache(cache_root)
    try:
        source = load_file(str(weights_path), device="cpu")
    except Exception as error:
        raise PretrainedVerificationError("Verified cached safetensors could not be loaded.") from error
    # weights=None constructs architecture only and cannot invoke Hub fallback.
    # The temporary encoder is completely overwritten before any model returns.
    model = KidneyUNet(pretrained=False, input_channels=input_channels)
    equality_proof = _load_and_prove_encoder(model, source, input_channels)
    model.pretrained = True
    proof = {
        "schema_version": 1,
        "offline_only": True,
        "download_attempted": False,
        "random_encoder_fallback": False,
        "pretrained_encoder": True,
        "decoder_pretrained": False,
        "decoder_initialization": "random; not task-specific pretrained weights",
        "segmentation_head_initialization": "random; three segmentation classes",
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "source_weights_url": "https://huggingface.co/%s/resolve/%s/model.safetensors" % (REPOSITORY, REVISION),
        "weights_sha256": WEIGHTS_SHA256,
        "config_sha256": CONFIG_SHA256,
        "cache_snapshot_directory": str(snapshot.resolve()),
        "source_config": config,
        "input_channels": input_channels,
        "input_mode": model.input_mode,
        "architecture": model.architecture,
        "initialization_kind": "verified_imagenet_encoder_random_decoder",
        "library_versions": {name: version(name) for name in
                             ("torch", "torchvision", "segmentation-models-pytorch", "safetensors")},
        **equality_proof,
    }
    model.pretrained_proof = proof
    return model, proof
