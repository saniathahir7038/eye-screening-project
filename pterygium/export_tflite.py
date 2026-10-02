"""Export a screening model to TensorFlow Lite and verify validation parity."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from screening import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, model_threshold


ROOT = Path(__file__).resolve().parent


def read_validation_rows(path: Path, phase2_dir: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 61 or {row["split"] for row in rows} != {"validation"}:
        raise ValueError("Expected the locked 61-image validation split")
    for row in rows:
        image_path = (phase2_dir / row["processed_image_path"]).resolve()
        if not image_path.is_relative_to(phase2_dir.resolve()) or not image_path.is_file():
            raise ValueError(f"Missing or unsafe validation image: {row['processed_image_path']}")
        row["absolute_image_path"] = str(image_path)
    return rows


def load_images(rows: list[dict]) -> np.ndarray:
    images = []
    for row in rows:
        image = tf.io.decode_png(tf.io.read_file(row["absolute_image_path"]), channels=3)
        image = tf.ensure_shape(image, (224, 224, 3))
        images.append(np.asarray(image, dtype=np.float32))
    return np.stack(images)


def convert_model(model: tf.keras.Model, path: Path, float16: bool) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    payload = converter.convert()
    path.write_bytes(payload)


def tflite_predict(path: Path, images: np.ndarray) -> tuple[np.ndarray, dict]:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=2)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    if tuple(input_detail["shape"]) != (1, 224, 224, 3) or input_detail["dtype"] != np.float32:
        raise ValueError(f"Unexpected TFLite input: {input_detail['shape']} {input_detail['dtype']}")
    if tuple(output_detail["shape"]) != (1, 1) or output_detail["dtype"] != np.float32:
        raise ValueError(f"Unexpected TFLite output: {output_detail['shape']} {output_detail['dtype']}")
    scores = []
    for image in images:
        interpreter.set_tensor(input_detail["index"], image[np.newaxis, ...])
        interpreter.invoke()
        scores.append(float(interpreter.get_tensor(output_detail["index"])[0, 0]))
    metadata = {
        "input_name": input_detail["name"],
        "input_shape": [int(value) for value in input_detail["shape"]],
        "input_dtype": input_detail["dtype"].__name__,
        "output_name": output_detail["name"],
        "output_shape": [int(value) for value in output_detail["shape"]],
        "output_dtype": output_detail["dtype"].__name__,
    }
    return np.asarray(scores, dtype=np.float64), metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(name: str, path: Path, keras_scores: np.ndarray, threshold: float,
            images: np.ndarray) -> dict:
    scores, metadata = tflite_predict(path, images)
    differences = np.abs(scores - keras_scores)
    return {
        "name": name,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "maximum_absolute_score_difference": float(differences.max()),
        "mean_absolute_score_difference": float(differences.mean()),
        "decision_agreement": float(np.mean((scores >= threshold) == (keras_scores >= threshold))),
        "metadata": metadata,
        "scores": scores,
    }


def export(args: argparse.Namespace) -> dict:
    threshold = model_threshold(args.model, args.summary)
    rows = read_validation_rows(args.validation, args.phase2)
    images = load_images(rows)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    model = tf.keras.models.load_model(args.model, compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3) or tuple(model.output_shape[1:]) != (1,):
        raise ValueError(f"Unexpected model shape: {model.input_shape} -> {model.output_shape}")
    keras_scores = model.predict(images, batch_size=16, verbose=0).reshape(-1).astype(np.float64)

    args.output.mkdir(parents=True, exist_ok=True)
    float32_path = args.output / "pterygium_float32.tflite"
    float16_path = args.output / "pterygium_float16.tflite"
    convert_model(model, float32_path, float16=False)
    convert_model(model, float16_path, float16=True)
    candidates = [
        compare("float32", float32_path, keras_scores, threshold, images),
        compare("float16", float16_path, keras_scores, threshold, images),
    ]
    candidates[0]["accepted"] = (
        candidates[0]["maximum_absolute_score_difference"] <= 1e-5
        and candidates[0]["decision_agreement"] == 1.0
    )
    candidates[1]["accepted"] = (
        candidates[1]["maximum_absolute_score_difference"] <= 0.01
        and candidates[1]["decision_agreement"] == 1.0
    )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    if not accepted:
        raise ValueError("No TFLite candidate passed the validation equivalence gate")
    selected = min(accepted, key=lambda candidate: candidate["size_bytes"])
    args.android_asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(selected["path"]), args.android_asset)

    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append({
            "filename": row["filename"],
            "label": int(labels[index]),
            "keras_score": float(keras_scores[index]),
            "float32_score": float(candidates[0]["scores"][index]),
            "float16_score": float(candidates[1]["scores"][index]),
        })
    with (args.output / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    for candidate in candidates:
        candidate.pop("scores")
    result = {
        "status": "conversion_verified",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": str(args.model.resolve()),
        "threshold": threshold,
        "validation_images": len(rows),
        "test_set_inference_performed": False,
        "preprocessing": "RGB float32 values in [0,255], top-left 20% neutral mask, bilinear resize to 224x224; model performs MobileNetV2 normalization",
        "candidates": candidates,
        "selected": selected["name"],
        "android_asset": str(args.android_asset.resolve()),
        "android_asset_sha256": sha256(args.android_asset),
    }
    (args.output / "phase6_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "# TensorFlow Lite export", "", "Status: **conversion verified; clinical performance unvalidated**", "",
        "## Evaluation boundary", "",
        "Conversion equivalence was checked on the 61-image SLID validation split. This is not smartphone validation.", "",
        "## Candidates", "", "| Candidate | Size | Maximum score difference | Decision agreement | Accepted |",
        "|---|---:|---:|---:|---|",
    ]
    for candidate in candidates:
        report.append(
            f"| {candidate['name']} | {candidate['size_bytes'] / (1024 * 1024):.2f} MB | "
            f"{candidate['maximum_absolute_score_difference']:.8f} | "
            f"{candidate['decision_agreement']:.1%} | {'yes' if candidate['accepted'] else 'no'} |"
        )
    report.extend([
        "", "## Selected mobile model", "",
        f"The {selected['name']} candidate was selected and copied to the Android assets directory. SHA-256: `{result['android_asset_sha256']}`.", "",
    ])
    (args.output / "phase6-report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--phase2", type=Path, default=ROOT / "artifacts/phase2")
    parser.add_argument("--validation", type=Path, default=ROOT / "artifacts/phase2/validation.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/domain_adapt_v1/export")
    parser.add_argument(
        "--android-asset", type=Path,
        default=ROOT / "android_app/app/src/main/assets/pterygium_model.tflite",
    )
    result = export(parser.parse_args())
    return 0 if result["status"] == "conversion_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
