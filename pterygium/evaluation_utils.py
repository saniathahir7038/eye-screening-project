"""Testable metric and spatial-attention helpers for Phase 4."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, roc_auc_score


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.size != scores.size or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires equal non-empty arrays containing both binary classes")
    predictions = (scores >= threshold).astype(np.int32)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    positive_total = true_positive + false_negative
    negative_total = true_negative + false_positive
    sensitivity = true_positive / positive_total
    specificity = true_negative / negative_total
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy_ci95": wilson_interval(true_positive + true_negative, len(labels)),
        "sensitivity_ci95": wilson_interval(true_positive, positive_total),
        "specificity_ci95": wilson_interval(true_negative, negative_total),
    }, predictions


def spatial_attention_metrics(heatmap, boxes, corner_fraction=0.20, border_fraction=0.10):
    heatmap = np.asarray(heatmap, dtype=np.float64)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all() or np.any(heatmap < 0):
        raise ValueError("Heatmap must be a finite non-negative two-dimensional array")
    height, width = heatmap.shape
    total = float(heatmap.sum())
    if total <= 0:
        total = 1.0
    corner_height = max(1, int(math.ceil(height * corner_fraction)))
    corner_width = max(1, int(math.ceil(width * corner_fraction)))
    border_height = max(1, int(math.ceil(height * border_fraction)))
    border_width = max(1, int(math.ceil(width * border_fraction)))
    border_mask = np.zeros_like(heatmap, dtype=bool)
    border_mask[:border_height, :] = True
    border_mask[-border_height:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    result = {
        "corner_attention_fraction": float(heatmap[:corner_height, :corner_width].sum() / total),
        "border_attention_fraction": float(heatmap[border_mask].sum() / total),
        "box_area_fraction": None,
        "box_attention_fraction": None,
        "box_attention_lift": None,
        "maximum_point_inside_box": None,
        "top10_box_iou": None,
    }
    if not boxes:
        return result
    box_mask = np.zeros_like(heatmap, dtype=bool)
    for box in boxes:
        x1 = max(0, min(width, int(math.floor(float(box["x"])))))
        y1 = max(0, min(height, int(math.floor(float(box["y"])))))
        x2 = max(0, min(width, int(math.ceil(float(box["x"]) + float(box["width"])))))
        y2 = max(0, min(height, int(math.ceil(float(box["y"]) + float(box["height"])))))
        box_mask[y1:y2, x1:x2] = True
    area_fraction = float(box_mask.mean())
    attention_fraction = float(heatmap[box_mask].sum() / total)
    maximum = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    cutoff = float(np.quantile(heatmap, 0.90))
    top_mask = heatmap >= cutoff
    union = np.logical_or(top_mask, box_mask).sum()
    result.update({
        "box_area_fraction": area_fraction,
        "box_attention_fraction": attention_fraction,
        "box_attention_lift": attention_fraction / area_fraction if area_fraction else None,
        "maximum_point_inside_box": bool(box_mask[maximum]),
        "top10_box_iou": float(np.logical_and(top_mask, box_mask).sum() / union) if union else 0.0,
    })
    return result
