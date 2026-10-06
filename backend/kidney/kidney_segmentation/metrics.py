"""Pixel metrics accumulated per patient; empty classes never become perfect."""
import numpy as np
from . import CLASSES

METRIC_NAMES = ("dice", "iou", "precision", "recall", "f1")


def confusion_matrix(target, prediction):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Target and prediction must have matching non-empty shapes")
    if not np.isin(target, [0, 1, 2]).all() or not np.isin(prediction, [0, 1, 2]).all():
        raise ValueError("Expected three-class integer segmentation labels")
    return np.bincount((3 * target.astype(np.int64) + prediction.astype(np.int64)).ravel(), minlength=9).reshape(3, 3)


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix)
    if matrix.shape != (3, 3) or (matrix < 0).any() or not np.equal(matrix, matrix.astype(np.int64)).all():
        raise ValueError("Expected a non-negative integer 3x3 confusion matrix")
    result = {}
    for index, name in CLASSES.items():
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        fp, fn = predicted - tp, support - tp
        result[name] = {
            "dice": _divide(2 * tp, 2 * tp + fp + fn),
            "iou": _divide(tp, tp + fp + fn),
            "precision": _divide(tp, tp + fp),
            "recall": _divide(tp, tp + fn),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
            "support_pixels": support, "predicted_pixels": predicted,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
        }
    return result


def patient_report(patient_id, matrix):
    return {"patient_id": patient_id, "per_class": metrics_from_confusion(matrix),
            "confusion_matrix": np.asarray(matrix).tolist(),
            "confusion_matrix_convention": "rows_actual_columns_predicted"}


def summarize_patients(reports):
    if not reports or len({r["patient_id"] for r in reports}) != len(reports):
        raise ValueError("Expected at least one report and unique patient IDs")
    means, counts = {}, {}
    for name in CLASSES.values():
        means[name], counts[name] = {}, {}
        for metric in METRIC_NAMES:
            values = [r["per_class"][name][metric] for r in reports
                      if r["per_class"][name][metric] is not None]
            means[name][metric] = float(np.mean(values)) if values else None
            counts[name][metric] = len(values)
    return {"unit": "patient", "patient_count": len(reports),
            "patient_mean": means, "contributing_patient_counts": counts,
            "patients": reports,
            "absent_class_policy": "undefined denominators return null; means exclude null",
            "metric_scope": "pixels in all slices after deterministic in-plane resize"}
