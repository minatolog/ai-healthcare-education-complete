"""Strict cached-weight proof tests; no download, optimizer or patient data."""
import json
from unittest.mock import patch

import pytest
import torch

from kidney_segmentation import pretrained as verified


def test_missing_cache_fails_before_architecture_or_download(tmp_path):
    with patch.object(verified, "KidneyUNet") as factory:
        with pytest.raises(verified.PretrainedVerificationError, match="missing config.json"):
            verified.build_verified_model(cache_root=tmp_path)
        factory.assert_not_called()


def test_changed_cached_config_rejected_before_model_initialization(tmp_path):
    snapshot = tmp_path / "models--smp-hub--resnet18.imagenet" / "snapshots" / verified.REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"num_classes": 2}))
    with patch.object(verified, "KidneyUNet") as factory:
        with pytest.raises(verified.PretrainedVerificationError, match="SHA-256"):
            verified.build_verified_model(cache_root=tmp_path)
        factory.assert_not_called()


@pytest.mark.parametrize("channels", [True, 0, 2, "1"])
def test_unsupported_input_channels_rejected_without_initialization(channels):
    with patch.object(verified, "KidneyUNet") as factory:
        with pytest.raises(ValueError, match="integer input_channels"):
            verified.build_verified_model(input_channels=channels)
        factory.assert_not_called()


class TinyEncoder(torch.nn.Module):
    """Small strict-state fixture with source-shaped convolution and BatchNorm."""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 7, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)


def source_fixture():
    return {
        "conv1.weight": torch.arange(64 * 3 * 7 * 7, dtype=torch.float32).reshape(64, 3, 7, 7),
        "bn1.weight": torch.full((64,), 2.0), "bn1.bias": torch.full((64,), 3.0),
        "bn1.running_mean": torch.full((64,), 4.0), "bn1.running_var": torch.full((64,), 5.0),
        "fc.weight": torch.zeros(1000, 512), "fc.bias": torch.zeros(1000),
    }


def fixture_model():
    class Model:
        class Network:
            encoder = TinyEncoder()
        network = Network()
    return Model()


def test_strict_loader_proves_rgb_sum_and_batch_norm_buffers():
    model, source = fixture_model(), source_fixture()
    proof = verified._load_and_prove_encoder(model, source, 1)
    assert torch.equal(model.network.encoder.conv1.weight, source["conv1.weight"].sum(1, keepdim=True))
    assert torch.equal(model.network.encoder.bn1.running_mean, source["bn1.running_mean"])
    assert model.network.encoder.bn1.num_batches_tracked.item() == 0
    assert all(not p.requires_grad for p in model.network.encoder.parameters())
    assert model.network.encoder.bn1.training is False
    assert proof["all_official_encoder_tensors_exact"] is True
    assert proof["initialized_zero_batch_norm_counter_count"] == 1


def test_missing_encoder_tensor_or_wrong_shape_never_passes_strict_proof():
    source = source_fixture()
    source.pop("bn1.running_var")
    with pytest.raises(verified.PretrainedVerificationError, match="state keys"):
        verified._load_and_prove_encoder(fixture_model(), source, 1)
    source = source_fixture()
    source["bn1.running_var"] = torch.ones(63)
    with pytest.raises(verified.PretrainedVerificationError, match="shape or dtype"):
        verified._load_and_prove_encoder(fixture_model(), source, 1)
