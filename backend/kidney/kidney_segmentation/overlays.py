"""Separate model prediction and optional reference annotation display assets."""
from pathlib import Path
import numpy as np
from PIL import Image
from .preprocessing import PreprocessConfig, display_slice, preprocess_mask


def render_overlay(image, mask, alpha=0.4):
    if not 0 <= alpha <= 1 or image.shape != mask.shape or not np.isin(mask, [0, 1, 2]).all():
        raise ValueError("Overlay requires aligned three-class labels and valid opacity")
    image = np.asarray(image, dtype=np.uint8)
    rgb = np.repeat(image[..., None], 3, axis=-1)
    colors = np.array([[0, 0, 0], [50, 200, 100], [255, 80, 80]], dtype=np.float32)
    active = mask > 0
    blended = rgb.astype(np.float32)
    blended[active] = (1 - alpha) * blended[active] + alpha * colors[mask[active].astype(np.int64)]
    return Image.fromarray(blended.round().astype(np.uint8))


def save_overlays(directory, volume, prediction, patient_id, config=PreprocessConfig(), ground_truth=None):
    directory = Path(directory)
    if prediction.shape != (len(volume), config.size, config.size):
        raise ValueError("Prediction is not aligned to resized full volume")
    if ground_truth is not None and ground_truth.shape != volume.shape:
        raise ValueError("Reference annotation is not aligned to source volume")
    directory.mkdir(parents=True, exist_ok=True)
    assets = []
    for z in range(len(volume)):
        image = display_slice(volume[z], config)
        display_path = directory / ("slice_%04d_ct.png" % z)
        prediction_path = directory / ("slice_%04d_predicted_segmentation.png" % z)
        Image.fromarray(image).save(display_path)
        render_overlay(image, prediction[z]).save(prediction_path)
        asset = {"patient_id": patient_id, "slice_index": z,
                 "ct_image": display_path.name, "predicted_overlay": prediction_path.name,
                 "predicted_overlay_type": "model_segmentation", "ground_truth_overlay": None}
        if ground_truth is not None:
            reference_path = directory / ("slice_%04d_ground_truth_annotation.png" % z)
            render_overlay(image, preprocess_mask(ground_truth[z], config).numpy()).save(reference_path)
            asset.update(ground_truth_overlay=reference_path.name, ground_truth_overlay_type="reference_annotation")
        assets.append(asset)
    return {"patient_id": patient_id, "legend": {"kidney": "green", "tumour": "red"},
            "description": "Predicted segmentation and ground-truth annotation are separate images; neither is Grad-CAM.",
            "assets": assets}
