"""Evaluate the locked screening model on an external normal/pterygium image set.

Expected layout: DATA_DIR/normal/* and DATA_DIR/pterygium/*. Images are not
copied into the repository. Thresholds and model weights are never refitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np

from prepare_dataset import preprocess_image
from screening import (DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, assess_image_quality,
                       decode_uploaded_image, load_runtime, model_threshold)
from framing import auto_frame
from probe_phone_image import MODEL_PATH as TFLITE_MODEL_PATH


ROOT = Path(__file__).resolve().parent
SLID_MANIFEST = ROOT / "artifacts/phase2/split_manifest.csv"


def pixel_digest(image):
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def reference_hashes():
    with SLID_MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return ({row["pixel_sha256"] for row in rows},
            [(row["filename"], imagehash.hex_to_hash(row["phash"]),
              imagehash.hex_to_hash(row["dhash"])) for row in rows])


def score_batches(model, arrays, batch_size=16):
    if not arrays:
        return []
    results = []
    for start in range(0, len(arrays), batch_size):
        batch = np.stack(arrays[start:start + batch_size]).astype(np.float32)
        results.extend(float(value) for value in model.predict(batch, verbose=0).reshape(-1))
    return results


def score_tflite(interpreter, arrays):
    from export_tflite import score_output_detail
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = score_output_detail(interpreter)["index"]
    results = []
    for array in arrays:
        interpreter.set_tensor(input_index, array.astype(np.float32)[None, ...])
        interpreter.invoke()
        results.append(float(interpreter.get_tensor(output_index)[0, 0]))
    return results


def evaluate(data_dir: Path, backend: str = "keras") -> dict:
    known_pixels, known_hashes = reference_hashes()
    if backend == "keras":
        runtime = load_runtime()
        threshold = runtime.threshold
        model_path = runtime.model_path
        score_inputs = lambda arrays: score_batches(runtime.model, arrays)
    elif backend == "tflite":
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH), num_threads=2)
        interpreter.allocate_tensors()
        threshold = model_threshold(DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH)
        model_path = str(TFLITE_MODEL_PATH)
        score_inputs = lambda arrays: score_tflite(interpreter, arrays)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    rows = []
    framed_arrays, full_arrays, full_indices = [], [], []
    for class_name, label in (("normal", 0), ("pterygium", 1)):
        paths = sorted(path for path in (data_dir / class_name).iterdir()
                       if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not paths:
            raise ValueError(f"No images found in {data_dir / class_name}")
        for path in paths:
            image = decode_uploaded_image(path.read_bytes())
            framing = auto_frame(image)
            quality = assess_image_quality(framing.image)
            processed, _ = preprocess_image(framing.image, 224, 0.20)
            phash = imagehash.phash(processed)
            dhash = imagehash.dhash(processed)
            digest = pixel_digest(image)
            closest = min(known_hashes, key=lambda row: (phash - row[1], dhash - row[2]))
            row = {
                "file": f"{class_name}/{path.name}", "label": label,
                "width": image.width, "height": image.height,
                "pixel_sha256": digest,
                "quality_accepted": quality.accepted,
                "quality_issues": list(quality.issues),
                "removed_bottom_fraction": framing.removed_bottom_fraction,
                "exact_slid_pixel_overlap": digest in known_pixels,
                "nearest_slid_file": closest[0],
                "nearest_slid_phash_distance": phash - closest[1],
                "nearest_slid_dhash_distance": dhash - closest[2],
                "score": None, "full_photo_score": None,
                "decision": "quality_rejected" if not quality.accepted else None,
            }
            rows.append(row)
            if quality.accepted:
                framed_arrays.append(np.asarray(processed, dtype=np.float32))
                if framing.adjusted:
                    full_processed, _ = preprocess_image(image, 224, 0.20)
                    full_arrays.append(np.asarray(full_processed, dtype=np.float32))
                    full_indices.append(len(rows) - 1)

    framed_scores = iter(score_inputs(framed_arrays))
    full_scores = dict(zip(full_indices, score_inputs(full_arrays)))
    for index, row in enumerate(rows):
        if not row["quality_accepted"]:
            continue
        score = next(framed_scores)
        full_score = full_scores.get(index)
        row["score"] = score
        row["full_photo_score"] = full_score
        if full_score is not None and (score >= threshold) != (full_score >= threshold):
            row["decision"] = "inconclusive"
        else:
            row["decision"] = "suspected" if score >= threshold else "no_visible_pattern"

    determinate = [row for row in rows if row["decision"] in {"suspected", "no_visible_pattern"}]
    seen_pixels = set()
    unique_determinate = []
    for row in determinate:
        if row["pixel_sha256"] not in seen_pixels:
            unique_determinate.append(row)
            seen_pixels.add(row["pixel_sha256"])
    counts = Counter((row["label"], row["decision"]) for row in determinate)
    tp, tn = counts[(1, "suspected")], counts[(0, "no_visible_pattern")]
    fp, fn = counts[(0, "suspected")], counts[(1, "no_visible_pattern")]
    unique_counts = Counter((row["label"], row["decision"]) for row in unique_determinate)
    return {
        "source": "Yoo 2020, Mendeley Data v2, DOI 10.17632/t75wjsw6bw.2",
        "modality_note": "Web-collected external-eye photographs; not verified as camera-native smartphone images.",
        "threshold": threshold,
        "backend": backend,
        "model_path": model_path,
        "total_images": len(rows),
        "normal_images": sum(row["label"] == 0 for row in rows),
        "pterygium_images": sum(row["label"] == 1 for row in rows),
        "quality_rejected": sum(row["decision"] == "quality_rejected" for row in rows),
        "inconclusive": sum(row["decision"] == "inconclusive" for row in rows),
        "auto_framed": sum(row["removed_bottom_fraction"] > 0 for row in rows),
        "exact_slid_pixel_overlap": sum(row["exact_slid_pixel_overlap"] for row in rows),
        "determinate": len(determinate),
        "unique_determinate": len(unique_determinate),
        "internal_duplicate_determinate": len(determinate) - len(unique_determinate),
        "true_positive": tp, "true_negative": tn,
        "false_positive": fp, "false_negative": fn,
        "unique_true_positive": unique_counts[(1, "suspected")],
        "unique_true_negative": unique_counts[(0, "no_visible_pattern")],
        "unique_false_positive": unique_counts[(0, "suspected")],
        "unique_false_negative": unique_counts[(1, "no_visible_pattern")],
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "accuracy": (tp + tn) / len(determinate) if determinate else None,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--backend", choices=("keras", "tflite"), default="keras")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.data_dir, args.backend)
    summary = {key: value for key, value in result.items() if key != "rows"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
