"""One-pass audit of the fixed experimental model on the original SLID test split.

This split was excluded from the new head's training and threshold selection,
but earlier project model iterations had already evaluated it. It is not a
new external or smartphone validation set.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from export_tflite import tflite_predict
from framing import auto_frame
from prepare_dataset import preprocess_image
from screening import (BASELINE_MODEL_PATH, BASELINE_SUMMARY_PATH,
                       DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH,
                       assess_image_quality, model_threshold)
from training_utils import confusion_values


ROOT = Path(__file__).resolve().parent
PHASE2 = ROOT / "artifacts/phase2"
TFLITE = ROOT / "android_app/app/src/main/assets/pterygium_model.tflite"
PREVIOUS_TFLITE = ROOT / "artifacts/phase6/pterygium_float16.tflite"


def read_rows(name: str) -> list[dict]:
    with (PHASE2 / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate() -> dict:
    test = read_rows("test")
    development = read_rows("train") + read_rows("validation")
    test_groups = {row["similarity_group"] for row in test}
    development_groups = {row["similarity_group"] for row in development}
    if test_groups & development_groups:
        raise ValueError("A similarity group crosses development and test")
    if {row["filename"] for row in test} & {row["filename"] for row in development}:
        raise ValueError("A filename crosses development and test")

    labels = np.asarray([int(row["label"]) for row in test], dtype=np.int32)
    processed = []
    quality_rejected = []
    app_inputs = []
    app_full_inputs = []
    app_full_indices = []
    auto_framed = []
    for row in test:
        with Image.open(PHASE2 / row["processed_image_path"]) as opened:
            processed.append(np.asarray(opened.convert("RGB"), dtype=np.float32))
        with Image.open(ROOT / "data/images" / row["image_path"]) as opened:
            original = opened.convert("RGB")
            framed = auto_frame(original)
            quality_rejected.append(not assess_image_quality(framed.image).accepted)
            auto_framed.append(framed.adjusted)
            app_image, _ = preprocess_image(framed.image, 224, 0.20)
            app_inputs.append(np.asarray(app_image, dtype=np.float32))
            if framed.adjusted:
                full_image, _ = preprocess_image(original, 224, 0.20)
                app_full_inputs.append(np.asarray(full_image, dtype=np.float32))
                app_full_indices.append(len(app_inputs) - 1)
    images = np.stack(processed)
    threshold = model_threshold(DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH)
    model = tf.keras.models.load_model(DEFAULT_MODEL_PATH, compile=False)
    keras_scores = model.predict(images, batch_size=16, verbose=0).reshape(-1)
    tflite_scores, tflite_metadata = tflite_predict(TFLITE, images)
    decisions_match = (keras_scores >= threshold) == (tflite_scores >= threshold)
    if not decisions_match.all():
        raise ValueError("Keras and APK model disagree on a test decision")
    stats = confusion_values(labels, tflite_scores, threshold)
    app_scores, _ = tflite_predict(TFLITE, np.stack(app_inputs))
    full_scores = {}
    if app_full_inputs:
        extra_scores, _ = tflite_predict(TFLITE, np.stack(app_full_inputs))
        full_scores = dict(zip(app_full_indices, extra_scores))
    app_inconclusive = [index for index, full_score in full_scores.items()
                        if (app_scores[index] >= threshold) != (full_score >= threshold)]
    accepted_indices = [index for index in range(len(test))
                        if not quality_rejected[index] and index not in app_inconclusive]
    app_stats = confusion_values(labels[accepted_indices], app_scores[accepted_indices], threshold)
    previous_threshold = model_threshold(BASELINE_MODEL_PATH, BASELINE_SUMMARY_PATH)
    phase6_summary = json.loads((ROOT / "artifacts/phase6/phase6_summary.json").read_text(encoding="utf-8"))
    if hashlib.sha256(PREVIOUS_TFLITE.read_bytes()).hexdigest() != phase6_summary["android_asset_sha256"]:
        raise ValueError("Previous APK model does not match the saved Phase 6 checksum")
    previous_scores, _ = tflite_predict(PREVIOUS_TFLITE, np.stack(app_inputs))
    previous_stats = confusion_values(labels[accepted_indices],
                                      previous_scores[accepted_indices], previous_threshold)
    rows = []
    for index, row in enumerate(test):
        rows.append({
            "filename": row["filename"],
            "true_label": "pterygium" if labels[index] else "no_pterygium_annotation",
            "predicted_label": "suspected_pterygium" if tflite_scores[index] >= threshold else "no_visible_pattern",
            "correct": bool((tflite_scores[index] >= threshold) == labels[index]),
            "quality_rejected_on_original": bool(quality_rejected[index]),
            "auto_framed": bool(auto_framed[index]),
            "app_inconclusive": index in app_inconclusive,
            "app_tflite_score": float(app_scores[index]),
            "previous_app_score": float(previous_scores[index]),
            "previous_predicted_label": "suspected_pterygium" if previous_scores[index] >= previous_threshold else "no_visible_pattern",
            "keras_score": float(keras_scores[index]),
            "tflite_score": float(tflite_scores[index]),
        })
    result = {
        "source": "Original SLID test.csv; slit-lamp images, not smartphone photos",
        "independence_limit": "Excluded from this head's training and threshold selection, but prior project versions evaluated this split",
        "model_sha256": hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest(),
        "tflite_sha256": hashlib.sha256(TFLITE.read_bytes()).hexdigest(),
        "threshold": threshold,
        "count": len(test),
        "pterygium_count": int(labels.sum()),
        "no_pterygium_annotation_count": int((labels == 0).sum()),
        "quality_rejected_on_original_count": int(sum(quality_rejected)),
        "auto_framed_count": int(sum(auto_framed)),
        "app_inconclusive_count": len(app_inconclusive),
        "app_determinate_count": len(accepted_indices),
        "app_path_counts": app_stats,
        "previous_app_path_counts": previous_stats,
        "previous_threshold": previous_threshold,
        "previous_tflite_sha256": hashlib.sha256(PREVIOUS_TFLITE.read_bytes()).hexdigest(),
        "changed_decisions": [row for row in rows if row["predicted_label"] != row["previous_predicted_label"]],
        "app_vs_processed_decision_agreement": float(np.mean(
            (app_scores >= threshold) == (tflite_scores >= threshold))),
        "keras_tflite_decision_agreement": float(decisions_match.mean()),
        "max_keras_tflite_score_difference": float(np.max(np.abs(keras_scores - tflite_scores))),
        "tflite_metadata": tflite_metadata,
        "counts": stats,
        "errors": [row for row in rows if not row["correct"]],
        "rows": rows,
    }
    return result


if __name__ == "__main__":
    result = evaluate()
    output = ROOT / "artifacts/domain_adapt_v1/slid_test_evaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
