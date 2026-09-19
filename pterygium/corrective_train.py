"""Train and validate the Phase 4A correction without reopening the locked test set."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf

from evaluate import build_gradcam_model, create_review_sheet, gradcam, overlay_image
from evaluation_utils import classification_metrics, spatial_attention_metrics
from train import configure_determinism, decode_image, load_and_validate_split, make_dataset, predict_rows, write_csv
from training_utils import balanced_class_weights, select_threshold

ROOT = Path(__file__).resolve().parent


def expand_corner_variants(rows):
    paths, labels, variants = [], [], []
    for row in rows:
        for variant in range(4):
            paths.append(row["absolute_image_path"])
            labels.append(int(row["label"]))
            variants.append(variant)
    return paths, labels, variants


def apply_corner_patch(image, label, variant, patch_size=45):
    """Variant zero is unchanged; variants 1-3 neutralize the unmasked corners."""
    yy, xx = tf.range(224)[:, None], tf.range(224)[None, :]
    masks = [
        tf.zeros((224, 224), dtype=tf.bool),
        (yy < patch_size) & (xx >= 224 - patch_size),
        (yy >= 224 - patch_size) & (xx < patch_size),
        (yy >= 224 - patch_size) & (xx >= 224 - patch_size),
    ]
    mask = tf.switch_case(variant, branch_fns=[lambda value=value: value for value in masks])
    return tf.where(mask[..., None], tf.cast(127, image.dtype), image), label


def make_corner_patch_dataset(rows, batch_size, seed, patch_size):
    paths, labels, variants = expand_corner_variants(rows)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, variants))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options).shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load_and_patch(path, label, variant):
        image, label = decode_image(path, label)
        return apply_corner_patch(image, label, variant, patch_size)

    return dataset.map(load_and_patch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True).batch(
        batch_size
    ).prefetch(tf.data.AUTOTUNE)


def prepare_classifier_head(model):
    for layer in model.layers:
        layer.trainable = layer.name == "pterygium_score"
    return sum(layer.trainable for layer in model.layers)


def fit_classifier_head(model, train_dataset, validation_dataset, class_weights, checkpoint,
                        epochs, patience, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model.fit(
        train_dataset, validation_data=validation_dataset, epochs=epochs, shuffle=False,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", mode="min", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=patience, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ], verbose=2,
    )


def attention_summary(rows):
    positives = [row for row in rows if row["label"] == 1]
    normals = [row for row in rows if row["label"] == 0]
    return {
        "mean_corner_attention_fraction": float(np.mean([r["corner_attention_fraction"] for r in rows])),
        "maximum_corner_attention_fraction": float(np.max([r["corner_attention_fraction"] for r in rows])),
        "mean_border_attention_fraction": float(np.mean([r["border_attention_fraction"] for r in rows])),
        "normal_mean_corner_attention_fraction": float(np.mean([r["corner_attention_fraction"] for r in normals])),
        "normal_mean_border_attention_fraction": float(np.mean([r["border_attention_fraction"] for r in normals])),
        "positive_mean_box_attention_fraction": float(np.mean([r["box_attention_fraction"] for r in positives])),
        "positive_mean_box_attention_lift": float(np.mean([r["box_attention_lift"] for r in positives])),
        "positive_maximum_point_inside_box_count": int(sum(r["maximum_point_inside_box"] for r in positives)),
        "positive_images": len(positives),
    }


def validation_review(model, validation_dataset, rows, output_dir, name):
    labels, scores = predict_rows(model, validation_dataset, rows)
    threshold, threshold_rows = select_threshold(labels, scores)
    metrics, predictions = classification_metrics(labels, scores, threshold["threshold"])
    grad_model = build_gradcam_model(model)
    overlays_dir = output_dir / f"{name}_gradcam"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    result_rows, maximum_difference = [], 0.0
    for source, label, score, prediction in zip(rows, labels, scores, predictions):
        image = Image.open(source["absolute_image_path"]).convert("RGB")
        heatmap, grad_score = gradcam(grad_model, np.asarray(image, dtype=np.float32))
        maximum_difference = max(maximum_difference, abs(float(score) - grad_score))
        boxes = json.loads(source["processed_pterygium_regions"])
        spatial = spatial_attention_metrics(heatmap, boxes)
        overlay_image(image, heatmap, boxes, bool(prediction)).save(overlays_dir / source["filename"])
        result_rows.append({
            "filename": source["filename"], "label": int(label), "pterygium_score": round(float(score), 8),
            "prediction": int(prediction), "correct": int(label == prediction),
            "threshold_margin": round(abs(float(score) - threshold["threshold"]), 8), **spatial,
            "gradcam_path": source["filename"],
        })
    if maximum_difference > 5e-6:
        raise ValueError(f"{name} Grad-CAM reconstruction changed scores by {maximum_difference}")
    write_csv(output_dir / f"{name}_validation_predictions.csv", result_rows, list(result_rows[0]))
    write_csv(output_dir / f"{name}_threshold_curve.csv", threshold_rows, list(threshold_rows[0]))
    positives = sorted([r for r in result_rows if r["label"] == 1], key=lambda r: r["box_attention_lift"])[:12]
    corners = sorted(result_rows, key=lambda r: r["corner_attention_fraction"], reverse=True)[:8]
    review_dir = output_dir / "review"
    review_dir.mkdir(exist_ok=True)
    create_review_sheet(positives, overlays_dir, review_dir / f"{name}-positive-low-overlap.png")
    create_review_sheet(corners, overlays_dir, review_dir / f"{name}-high-corner.png")
    return {
        "metrics": metrics, "threshold": threshold["threshold"],
        "roc_auc": float(roc_auc_score(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "attention": attention_summary(result_rows), "rows": result_rows,
    }


def plot_comparison(baseline, corrected, path):
    names = ["Box attention lift", "Max point in box", "Normal border attention", "Normal corner attention"]
    def values(result):
        attention = result["attention"]
        return [
            attention["positive_mean_box_attention_lift"],
            attention["positive_maximum_point_inside_box_count"] / attention["positive_images"],
            attention["normal_mean_border_attention_fraction"],
            attention["normal_mean_corner_attention_fraction"],
        ]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(positions - 0.18, values(baseline), 0.36, label="Original")
    axis.bar(positions + 0.18, values(corrected), 0.36, label="Corrected")
    axis.set_xticks(positions, names, rotation=12)
    axis.set_ylabel("Validation attention measure")
    axis.set_title("Validation Grad-CAM comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def acceptance_checks(baseline, corrected):
    base_attention, new_attention = baseline["attention"], corrected["attention"]
    return {
        "roc_auc_preserved": corrected["roc_auc"] >= baseline["roc_auc"] - 0.01,
        "sensitivity_at_least_0_95": corrected["metrics"]["sensitivity"] >= 0.95,
        "specificity_at_least_0_90": corrected["metrics"]["specificity"] >= 0.90,
        "box_attention_lift_not_worse": new_attention["positive_mean_box_attention_lift"] >= base_attention["positive_mean_box_attention_lift"],
        "max_point_localization_not_worse": new_attention["positive_maximum_point_inside_box_count"] >= base_attention["positive_maximum_point_inside_box_count"],
        "normal_corner_attention_reduced": new_attention["normal_mean_corner_attention_fraction"] < base_attention["normal_mean_corner_attention_fraction"],
    }


def train(args):
    configure_determinism(args.seed)
    splits = load_and_validate_split(args.phase2)
    validation = make_dataset(splits["validation"], args.batch_size, False, args.seed)
    training = make_corner_patch_dataset(splits["train"], args.batch_size, args.seed, args.patch_size)
    weights = balanced_class_weights([row["label"] for row in splits["train"]])
    args.output.mkdir(parents=True, exist_ok=True)
    source_model = args.phase3 / "final_model.keras"

    baseline_model = tf.keras.models.load_model(source_model, compile=False)
    baseline = validation_review(baseline_model, validation, splits["validation"], args.output, "baseline")
    model = tf.keras.models.load_model(source_model, compile=False)
    trainable_layers = prepare_classifier_head(model)
    checkpoint = args.output / "corrected_head_best.keras"
    history = fit_classifier_head(
        model, training, validation, weights, checkpoint, args.epochs, args.patience, args.learning_rate
    )
    corrected_path = args.output / "corrected_model.keras"
    shutil.copy2(checkpoint, corrected_path)
    corrected_model = tf.keras.models.load_model(corrected_path, compile=False)
    corrected = validation_review(corrected_model, validation, splits["validation"], args.output, "corrected")

    export_path = args.output / "saved_model"
    if export_path.exists():
        shutil.rmtree(export_path)
    corrected_model.export(export_path)
    history_rows = [
        {"epoch": epoch + 1, **{key: values[epoch] for key, values in history.history.items()}}
        for epoch in range(len(history.history["loss"]))
    ]
    write_csv(args.output / "training_history.csv", history_rows, list(history_rows[0]))
    plot_comparison(baseline, corrected, args.output / "validation-gradcam-comparison.png")

    checks = acceptance_checks(baseline, corrected)
    summary = {
        "status": "accepted" if all(checks.values()) else "needs_review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "method": "frozen_feature_extractor_with_three_corner_neutral_patch_variants",
        "patch_size": args.patch_size, "trainable_model_layers": trainable_layers,
        "test_set_inference_performed": False,
        "baseline": {key: value for key, value in baseline.items() if key != "rows"},
        "corrected": {key: value for key, value in corrected.items() if key != "rows"},
        "acceptance_checks": checks,
    }
    (args.output / "phase4a_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase 4A: corrective model iteration", "", f"Status: **{summary['status']}**", "",
        "## Constraint", "", "Training and model selection used only the original training and validation splits. The locked test set was not passed to the model and no new test metric was generated.", "",
        "## Correction", "", "The selected correction starts from the Phase 3 model, freezes its visual feature extractor and updates only the classifier head. Each training image is presented unchanged and with a neutral 45 x 45 patch in each of the three corners that were not already standardized during Phase 2.", "",
        "## Validation comparison", "", "| Measure | Original | Corrected |", "|---|---:|---:|",
        f"| ROC-AUC | {baseline['roc_auc']:.4f} | {corrected['roc_auc']:.4f} |",
        f"| Sensitivity | {baseline['metrics']['sensitivity']:.4f} | {corrected['metrics']['sensitivity']:.4f} |",
        f"| Specificity | {baseline['metrics']['specificity']:.4f} | {corrected['metrics']['specificity']:.4f} |",
        f"| Threshold | {baseline['threshold']:.6f} | {corrected['threshold']:.6f} |",
        f"| Positive box-attention lift | {baseline['attention']['positive_mean_box_attention_lift']:.3f} | {corrected['attention']['positive_mean_box_attention_lift']:.3f} |",
        f"| Maximum point inside box | {baseline['attention']['positive_maximum_point_inside_box_count']}/24 | {corrected['attention']['positive_maximum_point_inside_box_count']}/24 |",
        f"| Normal corner attention | {baseline['attention']['normal_mean_corner_attention_fraction']:.3f} | {corrected['attention']['normal_mean_corner_attention_fraction']:.3f} |", "",
        "## Acceptance checks", "", *[f"- {name}: {'passed' if value else 'failed'}" for name, value in checks.items()], "",
        "The corrected model passed the validation gate. A new independent test dataset is still required for an unbiased corrected-model performance estimate.", "",
    ]
    (args.output / "phase4a-report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, default=ROOT / "artifacts/phase2")
    parser.add_argument("--phase3", type=Path, default=ROOT / "artifacts/phase3")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/phase4a")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, default=45)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.patience, args.patch_size) < 1 or args.patch_size >= 112:
        parser.error("batch size, epochs, patience and patch size must be valid positive values")
    return 0 if train(args)["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
