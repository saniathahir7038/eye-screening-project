"""Small, testable helpers for pterygium model training."""
from __future__ import annotations

import math

import numpy as np


def confusion_values(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int32)
    predictions = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int32)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
    }


def select_threshold(labels, scores):
    """Maximise validation balanced accuracy, breaking ties toward sensitivity."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("Labels and scores must be non-empty and have equal length")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Threshold selection requires both binary classes")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite probabilities from 0 to 1")
    ordered_scores = np.unique(scores)
    midpoints = (ordered_scores[:-1] + ordered_scores[1:]) / 2
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], midpoints)))
    rows = [confusion_values(labels, scores, threshold) for threshold in candidates]
    best = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["sensitivity"],
            row["specificity"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return best, rows


def balanced_class_weights(labels):
    labels = [int(value) for value in labels]
    counts = {label: labels.count(label) for label in (0, 1)}
    if not all(counts.values()):
        raise ValueError("Class weights require both binary classes")
    total = len(labels)
    return {label: total / (2 * count) for label, count in counts.items()}


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else None
