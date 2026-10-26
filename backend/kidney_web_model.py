"""Lightweight kidney U-Net used by the website's kidney activity.

The team's research pipeline in ``backend/kidney`` needs a frozen ImageNet
encoder and a checkpoint that lives outside this repository. This module is a
small, self-contained alternative so the teaching website can run a real
segmentation model on the 40-case PNG handoff in ``data/kidney``.

It trains on the 20 ``train`` cases, selects the epoch with the best
patient-mean validation tumour Dice, and reports kidney and tumour Dice on the
10 ``test`` cases. Everything here is educational; the model is not a medical
device and was trained on a small teaching subset.
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / 'data/kidney/data2d/images'
MASK_DIR = ROOT / 'data/kidney/data2d/masks'
SPLIT_PATH = ROOT / 'data/kidney/patient_splits.csv'

MODEL_VERSION = 'kidney_unet_teaching_v1'
INPUT_SIZE = 128
CLASS_NAMES = ('background', 'kidney', 'tumour')
BASE_CHANNELS = 16
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
TUMOUR_LOSS_WEIGHT = 3.0
TUMOUR_PIXEL_THRESHOLD = 0.001  # 0.1 % of the 128x128 slice
OVERLAY_KIDNEY = (16, 185, 129)
OVERLAY_TUMOUR = (239, 68, 68)


class KidneyTrainingError(RuntimeError):
    """Raised when the teaching kidney model cannot be trained or loaded."""


# ── Model ─────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """Three-level U-Net: 16/32/64/128 channels at 128/64/32/16 pixels."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 base: int = BASE_CHANNELS) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool(skip1))
        skip3 = self.down3(self.pool(skip2))
        middle = self.bottleneck(self.pool(skip3))
        up3 = self.conv3(torch.cat([self.up3(middle), skip3], dim=1))
        up2 = self.conv2(torch.cat([self.up2(up3), skip2], dim=1))
        up1 = self.conv1(torch.cat([self.up1(up2), skip1], dim=1))
        return self.head(up1)


# ── Data ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SliceRef:
    case_id: str
    z: int

    @property
    def image_path(self) -> Path:
        return IMAGE_DIR / f'{self.case_id}_z{self.z:03d}.png'

    @property
    def mask_path(self) -> Path:
        return MASK_DIR / f'{self.case_id}_z{self.z:03d}.png'


def case_pools(split_path: Path = SPLIT_PATH) -> dict[str, str]:
    """Return ``case_id -> pool`` for the reviewed 40-case handoff."""

    import csv

    pools: dict[str, str] = {}
    with split_path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            pools[row['patient_id']] = row['pool']
    return pools


_SLICE_CACHE: dict[str, list[int]] | None = None


def case_slices() -> dict[str, list[int]]:
    """Return every case's sorted source slice numbers, cached once."""

    global _SLICE_CACHE
    if _SLICE_CACHE is None:
        collected: dict[str, list[int]] = {}
        for path in IMAGE_DIR.glob('case_*_z*.png'):
            stem = path.stem
            case_id, _, z_text = stem.rpartition('_z')
            if not z_text.isdigit():
                continue
            collected.setdefault(case_id, []).append(int(z_text))
        for values in collected.values():
            values.sort()
        _SLICE_CACHE = collected
    return _SLICE_CACHE


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        resized = handle.convert('L').resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        resized = handle.convert('L').resize((INPUT_SIZE, INPUT_SIZE), Image.NEAREST)
        return np.asarray(resized, dtype=np.int64)


def load_case_arrays(case_id: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load one case as ``(images[N,1,H,W], masks[N,H,W], slice_numbers)``."""

    slices = case_slices().get(case_id)
    if not slices:
        raise KidneyTrainingError(f'No PNG slices found for {case_id}.')
    images = np.stack([_load_image(IMAGE_DIR / f'{case_id}_z{z:03d}.png') for z in slices])
    masks = np.stack([_load_mask(MASK_DIR / f'{case_id}_z{z:03d}.png') for z in slices])
    return images[:, None, :, :], masks, slices


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Training ──────────────────────────────────────────────────────────────
def _dice_from_confusion(confusion: np.ndarray) -> list[float | None]:
    """Per-class Dice from a 3x3 confusion matrix (undefined -> ``None``)."""

    scores: list[float | None] = []
    for index in range(confusion.shape[0]):
        true_positive = confusion[index, index]
        denominator = confusion[index, :].sum() + confusion[:, index].sum()
        scores.append(None if denominator == 0 else float(2 * true_positive / denominator))
    return scores


def _evaluate(model: nn.Module, cases: dict[str, tuple[np.ndarray, np.ndarray]], device: torch.device) -> dict[str, object]:
    """Patient-mean Dice, plus per-case evidence and a pooled confusion matrix."""

    model.eval()
    case_rows: list[dict[str, object]] = []
    pooled = np.zeros((3, 3), dtype=np.int64)
    with torch.no_grad():
        for case_id, (images, masks) in cases.items():
            case_confusion = np.zeros((3, 3), dtype=np.int64)
            for start in range(0, len(images), 8):
                batch = torch.from_numpy(images[start:start + 8]).to(device)
                logits = model(batch)
                predicted = logits.argmax(dim=1).cpu().numpy()
                truth = masks[start:start + 8]
                for index in range(predicted.shape[0]):
                    flat = truth[index].ravel() * 3 + predicted[index].ravel()
                    counts = np.bincount(flat, minlength=9).reshape(3, 3)
                    case_confusion += counts
            pooled += case_confusion
            scores = _dice_from_confusion(case_confusion)
            case_rows.append({
                'case_id': case_id,
                'slice_count': int(len(images)),
                'kidney_dice': _round(scores[1]),
                'tumour_dice': _round(scores[2]),
                'tumour_pixels_in_reference': int(case_confusion[2, :].sum()),
                'tumour_pixels_predicted': int(case_confusion[:, 2].sum()),
            })

    def mean_of(key: str) -> float | None:
        values = [row[key] for row in case_rows if row[key] is not None]
        return None if not values else float(np.mean(values))

    pooled_scores = _dice_from_confusion(pooled)
    return {
        'case_count': len(case_rows),
        'patient_mean_kidney_dice': _round(mean_of('kidney_dice')),
        'patient_mean_tumour_dice': _round(mean_of('tumour_dice')),
        'pooled_kidney_dice': _round(pooled_scores[1]),
        'pooled_tumour_dice': _round(pooled_scores[2]),
        'cases': case_rows,
    }


ProgressCallback = Callable[[dict[str, object]], None]


def train(
    *,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 42,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Train on the 20 train cases and select the best validation epoch."""

    pools = case_pools()
    train_cases = sorted(case for case, pool in pools.items() if pool == 'train')
    validation_cases = sorted(case for case, pool in pools.items() if pool == 'validation')
    test_cases = sorted(case for case, pool in pools.items() if pool == 'test')
    if len(train_cases) != 20 or len(validation_cases) != 10 or len(test_cases) != 10:
        raise KidneyTrainingError('The reviewed split must stay at 20/10/10 cases.')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _select_device()

    training_data = {case: load_case_arrays(case)[:2] for case in train_cases}
    validation_data = {case: load_case_arrays(case)[:2] for case in validation_cases}
    test_data = {case: load_case_arrays(case)[:2] for case in test_cases}

    model = SmallUNet().to(device)
    weights = torch.tensor([1.0, 1.0, TUMOUR_LOSS_WEIGHT], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    samples = [
        (images[index], masks[index])
        for images, masks in training_data.values()
        for index in range(len(images))
    ]

    history: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(samples)))
        random.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            inputs = torch.from_numpy(
                np.stack([samples[index][0] for index in batch_indices])
            ).to(device)
            targets = torch.from_numpy(
                np.stack([samples[index][1] for index in batch_indices])
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)

        validation = _evaluate(model, validation_data, device)
        score = validation['patient_mean_tumour_dice'] or 0.0
        entry = {
            'epoch': epoch,
            'train_loss': round(total_loss / len(samples), 4),
            'validation_kidney_dice': _round(validation['patient_mean_kidney_dice']),
            'validation_tumour_dice': _round(validation['patient_mean_tumour_dice']),
        }
        history.append(entry)
        if score > best_score:
            best_score = score
            best = entry
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if progress is not None:
            progress({'event': 'training_progress', 'total_epochs': epochs, **entry})

    if best_state is None or best is None:
        raise KidneyTrainingError('Training produced no usable checkpoint.')
    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_data, device)

    return {
        'model_version': MODEL_VERSION,
        'input_size': INPUT_SIZE,
        'class_names': list(CLASS_NAMES),
        'device': str(device),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'tumour_loss_weight': TUMOUR_LOSS_WEIGHT,
        'split': {
            'train': train_cases,
            'validation': validation_cases,
            'test': test_cases,
            'train_slice_count': sum(len(value[0]) for value in training_data.values()),
            'validation_slice_count': sum(len(value[0]) for value in validation_data.values()),
            'test_slice_count': sum(len(value[0]) for value in test_data.values()),
        },
        'history': history,
        'selected_epoch': int(best['epoch']),
        'selected_validation_tumour_dice': best['validation_tumour_dice'],
        'validation': _evaluate(model, validation_data, device),
        'test': test_metrics,
        'state_dict': best_state,
        'educational_only': True,
    }


def _round(value: object) -> float | None:
    return None if value is None else round(float(value), 4)


def save_checkpoint(path: Path, result: dict[str, object]) -> Path:
    """Persist weights plus the recorded metrics for the website."""

    payload = dict(result)
    state = payload.pop('state_dict')
    torch.save({'config': payload, 'state_dict': state}, path)
    return path


def load_checkpoint(path: Path, device: torch.device | None = None) -> tuple[SmallUNet, dict[str, object]]:
    if not path.is_file():
        raise KidneyTrainingError(f'Missing kidney checkpoint: {path}')
    device = device or _select_device()
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as error:  # pragma: no cover - corrupted file
        raise KidneyTrainingError(f'Could not load {path}: {error}') from error
    config = payload.get('config') or {}
    model = SmallUNet().to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, config


# ── Prediction ────────────────────────────────────────────────────────────
class Predictor:
    """Cached segmentation predictor for one loaded checkpoint."""

    def __init__(self, model: SmallUNet, config: dict[str, object], device: torch.device) -> None:
        self.model = model
        self.config = config
        self.device = device
        self._masks: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _slice_outputs(self, case_id: str, z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (case_id, z)
        cached = self._masks.get(key)
        if cached is not None:
            return cached
        image = _load_image(IMAGE_DIR / f'{case_id}_z{z:03d}.png')
        with torch.no_grad():
            logits = self.model(torch.from_numpy(image[None, None]).to(self.device))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        predicted = probabilities.argmax(axis=0).astype(np.uint8)
        self._masks[key] = (image, probabilities[2], predicted)
        return self._masks[key]

    def predict_case(self, case_id: str) -> dict[str, object]:
        slices = case_slices().get(case_id)
        if not slices:
            raise KidneyTrainingError(f'No PNG slices found for {case_id}.')

        tumour_slices: list[dict[str, object]] = []
        reference_tumour_slices = 0
        highest = 0.0
        for z in slices:
            _, tumour_probability, predicted = self._slice_outputs(case_id, z)
            fraction = float((predicted == 2).mean())
            slice_score = float(tumour_probability.max())
            highest = max(highest, slice_score)
            if fraction >= TUMOUR_PIXEL_THRESHOLD:
                tumour_slices.append({
                    'z': z,
                    'tumour_pixel_fraction': round(fraction, 5),
                    'max_tumour_probability': round(slice_score, 4),
                })
            reference = _load_mask(MASK_DIR / f'{case_id}_z{z:03d}.png')
            if (reference == 2).any():
                reference_tumour_slices += 1

        tumour_slices.sort(key=lambda row: (-row['tumour_pixel_fraction'], row['z']))
        return {
            'case_id': case_id,
            'slice_count': len(slices),
            'tumour_slice_count': len(tumour_slices),
            'tumour_slice_ratio': round(len(tumour_slices) / len(slices), 4),
            'estimated_tumour_probability': round(highest, 4),
            'predicted_label': 1 if tumour_slices else 0,
            'reference_tumour_slice_count': reference_tumour_slices,
            'actual_label': 1 if reference_tumour_slices else 0,
            'tumour_slices': tumour_slices[:12],
            'selected_epoch': self.config.get('selected_epoch'),
            'model_version': self.config.get('model_version'),
            'educational_only': True,
        }

    def overlay_png(self, case_id: str, z: int) -> bytes:
        """Return the CT slice with predicted kidney/tumour outlines."""

        image, _, predicted = self._slice_outputs(case_id, z)
        rgb = np.repeat((image * 255).astype(np.uint8)[..., None], 3, axis=2)
        overlay = rgb.astype(np.float32)
        for class_index, colour in enumerate((None, OVERLAY_KIDNEY, OVERLAY_TUMOUR)):
            mask = predicted == class_index
            if class_index == 0 or not mask.any():
                continue
            edge = mask ^ _erode(mask)
            overlay[mask] = overlay[mask] * 0.55 + np.array(colour, dtype=np.float32) * 0.45
            overlay[edge] = np.array(colour, dtype=np.float32)
        result = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode='RGB')
        result = result.resize((256, 256), Image.NEAREST)
        buffer = io.BytesIO()
        result.save(buffer, format='PNG')
        return buffer.getvalue()


def _erode(mask: np.ndarray) -> np.ndarray:
    """3x3 binary erosion without SciPy."""

    padded = np.pad(mask.astype(np.uint8), 1, mode='constant')
    neighbours = sum(
        padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )
    return neighbours == 9


def describe_metrics(payload: dict[str, object]) -> str:
    """Short human-readable summary used in logs and the API status."""

    test = payload.get('test') or {}
    return json.dumps({
        'selected_epoch': payload.get('selected_epoch'),
        'test_kidney_dice': test.get('patient_mean_kidney_dice'),
        'test_tumour_dice': test.get('patient_mean_tumour_dice'),
    })
