"""SMP U-Net with one 2D slice or three adjacent CT-slice channels."""
import math
from numbers import Real
import torch
from torch import nn
import segmentation_models_pytorch as smp


class KidneyUNet(nn.Module):
    def __init__(self, pretrained=True, input_channels=3):
        super().__init__()
        if type(input_channels) is not int or input_channels not in (1, 3):
            raise ValueError("KidneyUNet supports input_channels=1 or 3 as an integer.")
        self.pretrained = pretrained
        self.input_channels = input_channels
        self.input_mode = "single_2d_slice" if input_channels == 1 else "adjacent_slices_2_5d"
        self.architecture = "smp_unet_resnet18"
        # SMP adapts an ImageNet first convolution to one channel by summing
        # its RGB weights. It does not repeat the input PNG into three channels.
        self.network = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet" if pretrained else None,
                                in_channels=input_channels, classes=3)
        for parameter in self.network.encoder.parameters():
            parameter.requires_grad_(False)
        self.network.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        # Freezing parameters alone would still update encoder BatchNorm stats.
        self.network.encoder.eval()
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError("Expected [batch, %d input channel(s), height, width]" % self.input_channels)
        return self.network(images)


def validate_tumour_ce_weight(value):
    """Validate a fixed relative tumour CE weight, without fitting data statistics."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("tumour_ce_weight must be a positive finite number, not a boolean.")
    return float(value)


def segmentation_loss(logits, target, *, tumour_ce_weight=1.0):
    """Class-weighted CE plus unchanged, equally averaged foreground soft-Dice.

    PyTorch's weighted CE mean divides by the sum of reference-class weights in
    the minibatch. A weight of 2 is a relative cost, not an absolute doubling of
    every tumour gradient. The default retains the original unweighted CE call.
    """
    weight = validate_tumour_ce_weight(tumour_ce_weight)
    if weight == 1.0:
        ce = nn.functional.cross_entropy(logits, target)
    else:
        class_weights = torch.tensor([1.0, 1.0, weight], dtype=logits.dtype, device=logits.device)
        ce = nn.functional.cross_entropy(logits, target, weight=class_weights, reduction="mean")
    probabilities = logits.softmax(dim=1)
    one_hot = nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).to(probabilities.dtype)
    intersection = (probabilities * one_hot).sum((0, 2, 3))[1:]
    denominator = (probabilities + one_hot).sum((0, 2, 3))[1:]
    dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return ce + dice_loss
