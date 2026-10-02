"""Export the current classifier with the feature map needed for mobile Grad-CAM.

The Android implementation uses the exact final BN/ReLU6/GAP/Dense chain to
calculate Grad-CAM weights. Its map must agree with Streamlit's gradient map
before these assets are installed in the app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
import tensorflow as tf

from evaluate import build_gradcam_model, gradcam
from export_tflite import load_images, read_validation_rows
from screening import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, model_threshold


ROOT = Path(__file__).resolve().parent
MAGIC = b"PGCAM001"
MAP_SIZE = 7
CHANNELS = 1280


def parameters(model: tf.keras.Model) -> tuple[np.ndarray, ...]:
    backbone = model.get_layer("mobilenetv2_backbone")
    batch_norm = backbone.get_layer("Conv_1_bn")
    dense = model.get_layer("pterygium_score")
    gamma, beta, mean, variance = (np.asarray(item, dtype=np.float32)
                                  for item in batch_norm.get_weights())
    weight = np.asarray(dense.get_weights()[0][:, 0], dtype=np.float32)
    if any(item.shape != (CHANNELS,) for item in (weight, gamma, beta, mean, variance)):
        raise ValueError("Unexpected Grad-CAM parameter shape")
    return weight, gamma, beta, mean, variance, float(batch_norm.epsilon)


def mobile_gradcam(conv: np.ndarray, params: tuple[np.ndarray, ...]) -> np.ndarray:
    """Mirror Android's pre-resize Grad-CAM math; positive sigmoid factor cancels."""
    weight, gamma, beta, mean, variance, epsilon = params
    scale = gamma / np.sqrt(variance + epsilon)
    bn = (conv - mean) * scale + beta
    active = np.logical_and(bn > 0, bn < 6)
    channel_weights = weight * scale * active.mean(axis=(0, 1))
    heatmap = np.maximum(np.sum(conv * channel_weights, axis=-1), 0)
    maximum = float(np.max(heatmap))
    return heatmap / maximum if maximum > 0 else heatmap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--validation", type=Path, default=ROOT / "artifacts/phase2/validation.csv")
    parser.add_argument("--phase2", type=Path, default=ROOT / "artifacts/phase2")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/domain_adapt_v1/mobile_heatmap")
    parser.add_argument("--android-assets", type=Path,
                        default=ROOT / "android_app/app/src/main/assets")
    parser.add_argument("--reuse-conversion", action="store_true",
                        help="Use an existing temporary export when debugging parity")
    args = parser.parse_args()
    threshold = model_threshold(args.model, args.summary)
    rows = read_validation_rows(args.validation, args.phase2)
    images = load_images(rows)
    model = tf.keras.models.load_model(args.model, compile=False)
    explain_model = build_gradcam_model(model)
    params = parameters(model)
    args.output.mkdir(parents=True, exist_ok=True)
    tflite_path = args.output / "pterygium_explain.tflite"
    if not args.reuse_conversion or not tflite_path.is_file():
        converter = tf.lite.TFLiteConverter.from_keras_model(explain_model)
        tflite_path.write_bytes(converter.convert())
    payload = tflite_path.read_bytes()
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=2)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    outputs = interpreter.get_output_details()
    if tuple(input_detail["shape"]) != (1, 224, 224, 3) or len(outputs) != 2:
        raise ValueError("Unexpected explainability model tensors")
    conv_detail = next((item for item in outputs if tuple(item["shape"]) ==
                        (1, MAP_SIZE, MAP_SIZE, CHANNELS)), None)
    score_detail = next((item for item in outputs if tuple(item["shape"]) == (1, 1)), None)
    if conv_detail is None or score_detail is None:
        raise ValueError(f"Unexpected explainability outputs: {outputs}")
    max_score_difference = 0.0
    max_heatmap_difference = 0.0
    decision_agreement = 0
    for image in images:
        reference_map, reference_score = gradcam(explain_model, image)
        interpreter.set_tensor(input_detail["index"], image[None, ...])
        interpreter.invoke()
        conv = interpreter.get_tensor(conv_detail["index"])[0]
        score = float(interpreter.get_tensor(score_detail["index"])[0, 0])
        low_res_map = mobile_gradcam(conv, params)
        resized = tf.image.resize(low_res_map[..., None], (224, 224),
                                  method="bilinear")[..., 0].numpy()
        max_score_difference = max(max_score_difference, abs(score - reference_score))
        max_heatmap_difference = max(max_heatmap_difference,
                                     float(np.max(np.abs(resized - reference_map))))
        decision_agreement += (score >= threshold) == (reference_score >= threshold)
    if decision_agreement != len(images) or max_score_difference > 1e-5 or max_heatmap_difference > 0.01:
        raise ValueError("Mobile Grad-CAM export did not match Streamlit: "
                         f"decisions {decision_agreement}/{len(images)}, "
                         f"score difference {max_score_difference:.8f}, "
                         f"heatmap difference {max_heatmap_difference:.8f}")

    weight, gamma, beta, mean, variance, epsilon = params
    params_path = args.output / "pterygium_gradcam.bin"
    params_path.write_bytes(MAGIC + struct.pack("<if", CHANNELS, epsilon) +
                            b"".join(item.astype("<f4").tobytes() for item in
                                     (weight, gamma, beta, mean, variance)))
    args.android_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite_path, args.android_assets / "pterygium_model.tflite")
    shutil.copy2(params_path, args.android_assets / params_path.name)
    report = {
        "validation_images": len(images),
        "score_decision_agreement": decision_agreement,
        "max_score_difference": max_score_difference,
        "max_heatmap_difference": max_heatmap_difference,
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "parameters_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
        "tflite_output_indices": {"score": int(score_detail["index"]),
                                  "conv": int(conv_detail["index"])},
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
