"""Compare full and conservatively auto-framed scores using the Android TFLite model.

Usage: python probe_phone_image.py photo.jpg [another-photo.png ...]
The script prints results only; it does not copy or retain private photographs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf

from framing import auto_frame
from screening import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, model_threshold


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "android_app/app/src/main/assets/pterygium_model.tflite"
THRESHOLD = model_threshold(DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    rgb = image.convert("RGB").copy()
    width, height = rgb.size
    ImageDraw.Draw(rgb).rectangle((0, 0, int(np.ceil(width * 0.2)) - 1,
                                   int(np.ceil(height * 0.2)) - 1), fill=(127, 127, 127))
    resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None, ...]


def score(interpreter: tf.lite.Interpreter, image: Image.Image) -> float:
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], preprocess(image))
    interpreter.invoke()
    return float(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--orientation-audit", action="store_true",
                        help="Also score 90 and 270 degree rotations for diagnostic comparison")
    args = parser.parse_args()
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH), num_threads=2)
    interpreter.allocate_tensors()
    for path in args.images:
        with Image.open(path) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        framing = auto_frame(original)
        framed = framing.image
        original_score = score(interpreter, original)
        framed_score = score(interpreter, framed) if framing.adjusted else original_score
        output = {
            "image": str(path),
            "original_size": list(original.size),
            "framed_size": list(framed.size),
            "removed_bottom_fraction": round(framing.removed_bottom_fraction, 4),
            "original_score": round(original_score, 6),
            "framed_score": round(framed_score, 6),
            "threshold": THRESHOLD,
            "decision_agrees": (original_score >= THRESHOLD) == (framed_score >= THRESHOLD),
            "note": "Scores are not probabilities; smartphone performance is not validated.",
        }
        if args.orientation_audit:
            output["rotated_90_score"] = round(score(interpreter, original.transpose(Image.Transpose.ROTATE_90)), 6)
            output["rotated_270_score"] = round(score(interpreter, original.transpose(Image.Transpose.ROTATE_270)), 6)
        print(json.dumps(output))


if __name__ == "__main__":
    main()
