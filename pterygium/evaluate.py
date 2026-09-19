"""Run the locked Phase 4 test evaluation and Grad-CAM review exactly once."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
import tensorflow as tf

from evaluation_utils import classification_metrics, spatial_attention_metrics
from train import configure_determinism, load_and_validate_split, make_dataset


ROOT = Path(__file__).resolve().parent


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_data_fingerprint(rows):
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["filename"]):
        digest.update(row["filename"].encode())
        digest.update(row["label"].encode())
        digest.update(row["similarity_group"].encode())
        digest.update(bytes.fromhex(sha256_file(row["absolute_image_path"])))
    return digest.hexdigest()


def input_fingerprints(model_path, phase3_summary_path, test_rows):
    phase3 = json.loads(phase3_summary_path.read_text(encoding="utf-8"))
    if phase3.get("status") != "passed" or phase3.get("test_set_evaluated") is not False:
        raise ValueError("Phase 3 did not preserve a valid locked test set")
    return {
        "model_sha256": sha256_file(model_path),
        "phase3_summary_sha256": sha256_file(phase3_summary_path),
        "test_data_sha256": test_data_fingerprint(test_rows),
        "threshold": float(phase3["selected_threshold"]),
    }


def build_gradcam_model(model):
    backbone = model.get_layer("mobilenetv2_backbone")
    convolution = backbone.get_layer("Conv_1")
    feature_extractor = tf.keras.Model(backbone.input, [convolution.output, backbone.output])
    inputs = tf.keras.Input((224, 224, 3), name="gradcam_eye_image")
    normalized = model.get_layer("mobilenet_normalization")(inputs)
    conv_output, features = feature_extractor(normalized, training=False)
    pooled = model.get_layer("global_average_pooling")(features)
    dropped = model.get_layer("dropout")(pooled, training=False)
    score = model.get_layer("pterygium_score")(dropped)
    return tf.keras.Model(inputs, [conv_output, score], name="pterygium_gradcam")


def gradcam(grad_model, image_array):
    tensor = tf.convert_to_tensor(image_array[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        convolution, score = grad_model(tensor, training=False)
        target = score[:, 0]
    gradients = tape.gradient(target, convolution)
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(convolution * weights[:, None, None, :], axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    heatmap = tf.image.resize(heatmap[..., None], (224, 224), method="bilinear")[..., 0]
    return heatmap.numpy(), float(score.numpy()[0, 0])


def overlay_image(image, heatmap, boxes, predicted_positive):
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32)
    colours = plt.get_cmap("jet")(np.clip(heatmap, 0, 1))[..., :3] * 255
    overlay = Image.fromarray(np.uint8(np.clip(0.58 * image_array + 0.42 * colours, 0, 255)))
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        x1, y1 = float(box["x"]), float(box["y"])
        x2, y2 = x1 + float(box["width"]), y1 + float(box["height"])
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
    colour = (255, 255, 255) if predicted_positive else (220, 220, 220)
    draw.rectangle((0, 204, 224, 224), fill=(0, 0, 0))
    draw.text((5, 207), "Predicted positive" if predicted_positive else "Predicted negative", fill=colour)
    return overlay


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_confusion_plot(labels, predictions, path):
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["Normal candidate", "Pterygium"])
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title("Locked test confusion matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=170)
    plt.close(display.figure_)


def create_roc_plot(labels, scores, auc_value, path):
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"MobileNetV2 (AUC={auc_value:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Locked test ROC curve",
             xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def create_score_plot(labels, scores, threshold, path):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(scores[labels == 0], bins=np.linspace(0, 1, 21), alpha=0.70, label="Normal candidate")
    axis.hist(scores[labels == 1], bins=np.linspace(0, 1, 21), alpha=0.70, label="Pterygium")
    axis.axvline(threshold, color="black", linestyle="--", label=f"Locked threshold {threshold:.4f}")
    axis.set(xlabel="Pterygium model score", ylabel="Test images", title="Locked test score distribution",
             xlim=(0, 1))
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def create_review_sheet(selected_rows, overlays_dir, path):
    columns, tile_width, tile_height = 4, 300, 285
    rows_count = max(1, (len(selected_rows) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for index, row in enumerate(selected_rows):
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        overlay = Image.open(overlays_dir / row["gradcam_path"]).convert("RGB")
        sheet.paste(overlay.resize((250, 250), Image.Resampling.LANCZOS), (x + 25, y + 30))
        title = f"{row['filename']} y={row['label']} p={row['prediction']} s={float(row['pterygium_score']):.3f}"
        draw.text((x + 7, y + 7), title, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def select_review_rows(rows, limit=16):
    selected = []
    seen = set()
    priorities = [
        [row for row in rows if row["correct"] == 0],
        sorted(rows, key=lambda row: row["threshold_margin"]),
        sorted([row for row in rows if row["label"] == 1],
               key=lambda row: row["box_attention_lift"] if row["box_attention_lift"] is not None else 999),
        sorted(rows, key=lambda row: row["corner_attention_fraction"], reverse=True),
    ]
    for group in priorities:
        for row in group:
            if row["filename"] not in seen:
                selected.append(row)
                seen.add(row["filename"])
            if len(selected) >= limit:
                return selected
    return selected


def aggregate_attention(rows):
    positives = [row for row in rows if row["label"] == 1]
    return {
        "mean_corner_attention_fraction": float(np.mean([row["corner_attention_fraction"] for row in rows])),
        "maximum_corner_attention_fraction": float(np.max([row["corner_attention_fraction"] for row in rows])),
        "mean_border_attention_fraction": float(np.mean([row["border_attention_fraction"] for row in rows])),
        "positive_mean_box_attention_fraction": float(np.mean([row["box_attention_fraction"] for row in positives])),
        "positive_mean_box_attention_lift": float(np.mean([row["box_attention_lift"] for row in positives])),
        "positive_maximum_point_inside_box_count": int(sum(row["maximum_point_inside_box"] for row in positives)),
        "positive_images": len(positives),
    }


def evaluate(args):
    configure_determinism(args.seed)
    split_rows = load_and_validate_split(args.phase2)
    phase3_summary_path = args.phase3 / "phase3_summary.json"
    model_path = args.phase3 / "final_model.keras"
    fingerprints = input_fingerprints(model_path, phase3_summary_path, split_rows["test"])
    lock_path = args.output / "evaluation_lock.json"
    if not args.smoke_test and lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("input_fingerprints") != fingerprints:
            raise ValueError("A locked test evaluation exists for different model, threshold or test data")
        summary = json.loads((args.output / "phase4_summary.json").read_text(encoding="utf-8"))
        print(json.dumps({**summary, "execution": "reused_locked_results"}, indent=2))
        return summary

    model = tf.keras.models.load_model(model_path, compile=False)
    grad_model = build_gradcam_model(model)
    if args.smoke_test:
        row = split_rows["validation"][0]
        image = Image.open(row["absolute_image_path"]).convert("RGB")
        array = np.asarray(image, dtype=np.float32)
        heatmap, grad_score = gradcam(grad_model, array)
        direct_score = float(model(array[None, ...], training=False).numpy()[0, 0])
        difference = abs(grad_score - direct_score)
        if heatmap.shape != (224, 224) or difference > 1e-6 or not np.isfinite(heatmap).all():
            raise ValueError("Grad-CAM smoke test failed")
        print(json.dumps({"status": "passed", "filename": row["filename"],
                          "heatmap_shape": list(heatmap.shape), "score_difference": difference}, indent=2))
        return {"status": "passed"}

    args.output.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output / "plots"
    overlays_dir = args.output / "gradcam"
    review_dir = args.output / "review"
    for directory in (plots_dir, overlays_dir, review_dir):
        directory.mkdir(exist_ok=True)

    test_dataset = make_dataset(split_rows["test"], args.batch_size, False, args.seed)
    scores = model.predict(test_dataset, verbose=0).reshape(-1)
    labels = np.asarray([int(row["label"]) for row in split_rows["test"]], dtype=np.int32)
    metrics, predictions = classification_metrics(labels, scores, fingerprints["threshold"])
    output_rows = []
    maximum_score_difference = 0.0
    for source, label, score, prediction in zip(split_rows["test"], labels, scores, predictions):
        image = Image.open(source["absolute_image_path"]).convert("RGB")
        array = np.asarray(image, dtype=np.float32)
        heatmap, grad_score = gradcam(grad_model, array)
        maximum_score_difference = max(maximum_score_difference, abs(float(score) - grad_score))
        boxes = json.loads(source["processed_pterygium_regions"])
        spatial = spatial_attention_metrics(heatmap, boxes)
        overlay_name = source["filename"]
        overlay_image(image, heatmap, boxes, bool(prediction)).save(overlays_dir / overlay_name)
        output_rows.append({
            "filename": source["filename"],
            "label": int(label),
            "pterygium_score": round(float(score), 8),
            "prediction": int(prediction),
            "correct": int(label == prediction),
            "threshold_margin": round(abs(float(score) - fingerprints["threshold"]), 8),
            **spatial,
            "gradcam_path": overlay_name,
        })
    if maximum_score_difference > 1e-6:
        raise ValueError(f"Grad-CAM reconstruction changed predictions by {maximum_score_difference}")

    prediction_fields = list(output_rows[0])
    write_csv(args.output / "test_predictions.csv", output_rows, prediction_fields)
    create_confusion_plot(labels, predictions, plots_dir / "confusion-matrix.png")
    create_roc_plot(labels, scores, metrics["roc_auc"], plots_dir / "roc-curve.png")
    create_score_plot(labels, scores, fingerprints["threshold"], plots_dir / "score-distribution.png")
    review_rows = select_review_rows(output_rows)
    create_review_sheet(review_rows, overlays_dir, review_dir / "gradcam-priority-review.png")
    error_rows = [row for row in output_rows if not row["correct"]]
    positive_low_overlap = sorted(
        [row for row in output_rows if row["label"] == 1], key=lambda row: row["box_attention_lift"]
    )[:12]
    high_corner_attention = sorted(
        output_rows, key=lambda row: row["corner_attention_fraction"], reverse=True
    )[:8]
    create_review_sheet(error_rows, overlays_dir, review_dir / "errors-review.png")
    create_review_sheet(positive_low_overlap, overlays_dir, review_dir / "positive-low-overlap-review.png")
    create_review_sheet(high_corner_attention, overlays_dir, review_dir / "high-corner-attention-review.png")
    attention = aggregate_attention(output_rows)
    misclassified = [row["filename"] for row in output_rows if not row["correct"]]
    summary = {
        "status": "passed",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_images": len(output_rows),
        **metrics,
        "misclassified_images": misclassified,
        "gradcam_images": len(output_rows),
        "gradcam_prediction_max_difference": maximum_score_difference,
        "attention_summary": attention,
        "model_sha256": fingerprints["model_sha256"],
        "test_data_sha256": fingerprints["test_data_sha256"],
    }
    (args.output / "phase4_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase 4: locked test evaluation and Grad-CAM", "",
        "Status: **passed**", "",
        "## Locked test metrics", "",
        f"The selected Phase 3 model and threshold ({metrics['threshold']:.6f}) were applied once to all "
        f"{len(output_rows)} held-out test images.", "",
        "| Metric | Result | 95% confidence interval |", "|---|---:|---:|",
        f"| Accuracy | {metrics['accuracy']:.3f} | {metrics['accuracy_ci95'][0]:.3f}–{metrics['accuracy_ci95'][1]:.3f} |",
        f"| Sensitivity | {metrics['sensitivity']:.3f} | {metrics['sensitivity_ci95'][0]:.3f}–{metrics['sensitivity_ci95'][1]:.3f} |",
        f"| Specificity | {metrics['specificity']:.3f} | {metrics['specificity_ci95'][0]:.3f}–{metrics['specificity_ci95'][1]:.3f} |",
        f"| Precision | {metrics['precision']:.3f} | — |",
        f"| F1-score | {metrics['f1_score']:.3f} | — |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.3f} | — |",
        f"| ROC-AUC | {metrics['roc_auc']:.3f} | — |", "",
        f"Confusion matrix: TP={metrics['true_positive']}, TN={metrics['true_negative']}, "
        f"FP={metrics['false_positive']}, FN={metrics['false_negative']}.", "",
        f"Misclassified images: {', '.join(misclassified) if misclassified else 'none'}.", "",
        "## Grad-CAM checks", "",
        f"Generated overlays for all {len(output_rows)} test images. The Grad-CAM reconstruction matched direct model "
        f"scores within {maximum_score_difference:.2e}.", "",
        f"Across positive images, the mean fraction of heatmap energy inside the annotated pterygium boxes was "
        f"{attention['positive_mean_box_attention_fraction']:.3f}; relative to box area, the mean attention lift was "
        f"{attention['positive_mean_box_attention_lift']:.2f}. The maximum-attention point fell inside a lesion box in "
        f"{attention['positive_maximum_point_inside_box_count']} of {attention['positive_images']} positive images.", "",
        f"Mean attention in the standardized upper-left corner was "
        f"{attention['mean_corner_attention_fraction']:.3f}; the maximum across test images was "
        f"{attention['maximum_corner_attention_fraction']:.3f}.", "",
        "The quantitative attention checks are descriptive and do not prove clinical reasoning. The priority review "
        "sheet must be inspected before accepting the phase.", "",
        "## Outputs", "",
        "- `test_predictions.csv`", "- `plots/confusion-matrix.png`", "- `plots/roc-curve.png`",
        "- `plots/score-distribution.png`", "- `gradcam/`", "- `review/gradcam-priority-review.png`",
        "- `review/errors-review.png`", "- `review/positive-low-overlap-review.png`",
        "- `review/high-corner-attention-review.png`", "- `evaluation_lock.json`", "",
        "The small slit-lamp test set and unavailable patient identifiers limit generalization. These results do not "
        "establish performance on ordinary smartphone photographs.", "",
    ]
    (args.output / "phase4-report.md").write_text("\n".join(report), encoding="utf-8")
    lock = {
        "created_at_utc": summary["evaluated_at_utc"],
        "input_fingerprints": fingerprints,
        "summary_sha256": sha256_file(args.output / "phase4_summary.json"),
        "note": "The locked test set was evaluated after Phase 3 model and threshold selection.",
    }
    temporary_lock = lock_path.with_suffix(".tmp")
    temporary_lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    temporary_lock.replace(lock_path)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, default=ROOT / "artifacts/phase2")
    parser.add_argument("--phase3", type=Path, default=ROOT / "artifacts/phase3")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/phase4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    result = evaluate(args)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
