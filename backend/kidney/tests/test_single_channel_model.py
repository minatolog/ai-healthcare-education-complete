"""Synthetic random-encoder checks; no ImageNet download or real training."""
import pytest
import torch
from torch.nn import functional as F

from kidney_segmentation.model import KidneyUNet, segmentation_loss

torch.set_num_threads(2)


def synthetic_batch():
    generator = torch.Generator().manual_seed(123)
    inputs = torch.randn(2, 1, 64, 64, generator=generator)
    target = torch.zeros((2, 64, 64), dtype=torch.long)
    target[:, 10:40, 10:40] = 1
    target[:, 20:30, 20:30] = 2
    return inputs, target


def test_one_channel_forward_and_exactly_one_decoder_optimizer_step():
    torch.manual_seed(42)
    model = KidneyUNet(pretrained=False, input_channels=1)
    images, target = synthetic_batch()
    encoder_before = {name: value.detach().clone() for name, value in model.network.encoder.state_dict().items()}
    decoder_before = {name: parameter.detach().clone() for name, parameter in model.network.decoder.named_parameters()}
    head_before = {name: parameter.detach().clone() for name, parameter in model.network.segmentation_head.named_parameters()}
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)

    model.train()
    assert model.input_channels == 1 and model.input_mode == "single_2d_slice"
    assert model.network.encoder.conv1.in_channels == 1
    assert not model.network.encoder.training
    assert all(not layer.training for layer in model.network.encoder.modules() if isinstance(layer, torch.nn.BatchNorm2d))
    logits = model(images)
    assert logits.shape == (2, 3, 64, 64)
    ce = F.cross_entropy(logits, target)
    loss = segmentation_loss(logits, target)
    foreground_dice_loss = loss - ce
    assert torch.isfinite(loss) and torch.isfinite(ce) and torch.isfinite(foreground_dice_loss)
    assert 0 <= foreground_dice_loss.item() <= 1
    optimizer.zero_grad()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.network.encoder.parameters())
    optimizer.step()

    # Includes BatchNorm running_mean, running_var and num_batches_tracked.
    for name, value in model.network.encoder.state_dict().items():
        assert torch.equal(value, encoder_before[name]), name
    assert any(not torch.equal(parameter.detach(), decoder_before[name]) for name, parameter in model.network.decoder.named_parameters())
    assert any(not torch.equal(parameter.detach(), head_before[name]) for name, parameter in model.network.segmentation_head.named_parameters())


def test_reference_targets_affect_loss_but_are_not_forward_inputs():
    torch.manual_seed(7)
    model = KidneyUNet(pretrained=False, input_channels=1).eval()
    images, target = synthetic_batch()
    with torch.inference_mode():
        logits_before = model(images)
        reference_loss = segmentation_loss(logits_before, target)
        changed_target = torch.zeros_like(target)
        changed_loss = segmentation_loss(logits_before, changed_target)
        logits_after = model(images)
    torch.testing.assert_close(logits_before, logits_after, rtol=0, atol=0)
    assert reference_loss.item() != changed_loss.item()
    with pytest.raises(TypeError):
        model(images, target)


def test_default_three_channel_architecture_and_weights_are_unchanged():
    torch.manual_seed(42)
    default = KidneyUNet(pretrained=False).eval()
    torch.manual_seed(42)
    explicit = KidneyUNet(pretrained=False, input_channels=3).eval()
    assert default.input_channels == 3 and default.input_mode == "adjacent_slices_2_5d"
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name
    assert default(torch.zeros(1, 3, 64, 64)).shape == (1, 3, 64, 64)
    with pytest.raises(ValueError, match="3 input"):
        default(torch.zeros(1, 1, 64, 64))
    with pytest.raises(ValueError, match="1 input"):
        KidneyUNet(pretrained=False, input_channels=1)(torch.zeros(1, 3, 64, 64))


@pytest.mark.parametrize("channels", [0, 2, 4, -1, True, 1.0, None])
def test_invalid_input_channel_contract_is_rejected(channels):
    with pytest.raises(ValueError, match="input_channels=1 or 3"):
        KidneyUNet(pretrained=False, input_channels=channels)
