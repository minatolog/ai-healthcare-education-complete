"""Deterministic legacy-compatible 2.5D processing without mask selection."""
from dataclasses import asdict, dataclass
import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class PreprocessConfig:
    hu_min: float = -79.0
    hu_max: float = 304.0
    size: int = 256
    mean: float = 0.456
    std: float = 0.224
    version: str = "kits-legacy-window-u8-cv2-context3-v1"

    def __post_init__(self):
        if not np.isfinite([self.hu_min, self.hu_max, self.mean, self.std]).all() or self.hu_max <= self.hu_min or self.std <= 0:
            raise ValueError("Invalid HU window or normalization")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 32 or self.size % 32:
            raise ValueError("U-Net image size must be a positive multiple of 32")

    def to_dict(self):
        return asdict(self)


def validate_volume(volume):
    volume = np.asarray(volume)
    if volume.ndim != 3 or min(volume.shape) == 0 or not np.isfinite(volume).all():
        raise ValueError("Expected a finite non-empty [axial slices, height, width] HU volume")
    return volume


def context_indices(z, n_slices):
    if not 0 <= z < n_slices:
        raise ValueError("Slice index is outside the patient volume")
    return (max(0, z - 1), z, min(n_slices - 1, z + 1))


def display_slice(slice_hu, config=PreprocessConfig()):
    """Quantize BEFORE bilinear resize, as verified in old preprocess.py."""
    scaled = (np.clip(slice_hu.astype(np.float32), config.hu_min, config.hu_max)
              - config.hu_min) / (config.hu_max - config.hu_min)
    image = (scaled * 255.0).astype(np.uint8)
    return cv2.resize(image, (config.size, config.size), interpolation=cv2.INTER_LINEAR)


def preprocess_slice(volume, z, config=PreprocessConfig()):
    # No mask, label, ROI or tumour-presence input is accepted here.
    indices = context_indices(z, len(volume))
    image = np.stack([display_slice(volume[i], config) for i in indices])
    normalized = (image.astype(np.float32) / 255.0 - config.mean) / config.std
    return torch.from_numpy(np.ascontiguousarray(normalized)), indices


def preprocess_mask(mask, config=PreprocessConfig()):
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError("Expected a finite 2D segmentation mask")
    if not np.isin(mask, [0, 1, 2]).all():
        raise ValueError("Mask labels must be 0=background, 1=kidney, 2=tumour")
    resized = cv2.resize(mask.astype(np.uint8), (config.size, config.size), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized.astype(np.int64))
