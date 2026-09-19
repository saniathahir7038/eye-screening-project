"""Train and select a MobileNetV2 pterygium classifier using the locked Phase 2 split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf

from training_utils import balanced_class_weights, confusion_values, json_number, select_threshold


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


def configure_determinism(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def read_manifest(path, split):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"filename", "label", "split", "similarity_group", "processed_image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid split manifest: {path}")
    for row in rows:
        if row["split"] != split or row["label"] not in {"0", "1"}:
            raise ValueError(f"Unexpected row in {path}: {row['filename']}")
    return rows


def load_and_validate_split(phase2_dir):
    by_split = {split: read_manifest(phase2_dir / f"{split}.csv", split) for split in SPLITS}
    all_rows = [row for split in SPLITS for row in by_split[split]]
    names = [row["filename"] for row in all_rows]
    if len(names) != 408 or len(set(names)) != 408:
        raise ValueError("Locked split must contain 408 unique filenames")
    expected = {
        "train": Counter({"0": 172, "1": 114}),
        "validation": Counter({"0": 37, "1": 24}),
        "test": Counter({"0": 36, "1": 25}),
    }
    actual = {split: Counter(row["label"] for row in rows) for split, rows in by_split.items()}
    if actual != expected:
        raise ValueError(f"Locked split counts changed: {actual}")
    group_splits = defaultdict(set)
    phase2_root = phase2_dir.resolve()
    for row in all_rows:
        group_splits[row["similarity_group"]].add(row["split"])
        image_path = (phase2_root / row["processed_image_path"]).resolve()
        if not image_path.is_relative_to(phase2_root) or not image_path.is_file():
            raise ValueError(f"Unsafe or missing processed image: {row['processed_image_path']}")
        row["absolute_image_path"] = str(image_path)
    violations = [group for group, splits in group_splits.items() if len(splits) != 1]
    if violations:
        raise ValueError(f"Similarity groups cross split boundaries: {violations}")
    return by_split


def decode_image(path, label):
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = tf.ensure_shape(image, (224, 224, 3))
    return tf.cast(image, tf.float32), tf.cast(label, tf.float32)


def make_dataset(rows, batch_size, training, seed):
    paths = [row["absolute_image_path"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if training:
        dataset = dataset.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def augmentation(seed):
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.028, fill_mode="reflect", seed=seed + 1),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed + 2),
        tf.keras.layers.RandomZoom((-0.08, 0.08), (-0.08, 0.08), fill_mode="reflect", seed=seed + 3),
        tf.keras.layers.RandomContrast(0.10, seed=seed + 4),
        tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255), seed=seed + 5),
    ], name="training_augmentation")


def build_model(seed, weights="imagenet"):
    inputs = tf.keras.Input((224, 224, 3), name="eye_image")
    x = augmentation(seed)(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_normalization")(x)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=weights, name="mobilenetv2_backbone"
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.30, seed=seed + 6, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="pterygium_score")(x)
    return tf.keras.Model(inputs, outputs, name="pterygium_mobilenetv2")


def metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="sensitivity"),
    ]


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics(),
    )


def fit_stage(model, train_dataset, validation_dataset, class_weights, checkpoint,
              epochs, patience, learning_rate):
    compile_model(model, learning_rate)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.3, patience=max(2, patience // 2), min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        shuffle=False,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )


def enable_fine_tuning(model, trainable_tail):
    backbone = model.get_layer("mobilenetv2_backbone")
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - trainable_tail)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return sum(layer.trainable for layer in backbone.layers)


def predict_rows(model, dataset, rows):
    scores = model.predict(dataset, verbose=0).reshape(-1)
    if len(scores) != len(rows):
        raise ValueError("Prediction count does not match manifest")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    return labels, scores


def stage_validation(model_path, dataset, rows):
    model = tf.keras.models.load_model(model_path, compile=False)
    labels, scores = predict_rows(model, dataset, rows)
    return {
        "model": model,
        "labels": labels,
        "scores": scores,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
    }


def combine_histories(stage_histories):
    rows = []
    overall_epoch = 0
    for stage, history in stage_histories:
        count = len(history.history.get("loss", []))
        for stage_epoch in range(count):
            overall_epoch += 1
            row = {"stage": stage, "stage_epoch": stage_epoch + 1, "overall_epoch": overall_epoch}
            for key, values in history.history.items():
                row[key] = json_number(values[stage_epoch])
            rows.append(row)
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history_rows, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs = [row["overall_epoch"] for row in history_rows]
    axes[0].plot(epochs, [row["loss"] for row in history_rows], label="Training")
    axes[0].plot(epochs, [row["val_loss"] for row in history_rows], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[1].plot(epochs, [row["auc"] for row in history_rows], label="Training")
    axes[1].plot(epochs, [row["val_auc"] for row in history_rows], label="Validation")
    axes[1].set(title="ROC-AUC", xlabel="Epoch", ylabel="AUC", ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_threshold(rows, selected, path):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    thresholds = [row["threshold"] for row in rows]
    axis.plot(thresholds, [row["sensitivity"] for row in rows], label="Sensitivity")
    axis.plot(thresholds, [row["specificity"] for row in rows], label="Specificity")
    axis.plot(thresholds, [row["balanced_accuracy"] for row in rows], label="Balanced accuracy")
    axis.axvline(selected["threshold"], color="black", linestyle="--", label="Selected threshold")
    axis.set(xlabel="Threshold", ylabel="Validation metric", xlim=(0, 1), ylim=(0, 1.02),
             title="Validation-only threshold selection")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(args):
    configure_determinism(args.seed)
    split_rows = load_and_validate_split(args.phase2)
    train_dataset = make_dataset(split_rows["train"], args.batch_size, True, args.seed)
    validation_dataset = make_dataset(split_rows["validation"], args.batch_size, False, args.seed)
    weights = balanced_class_weights([row["label"] for row in split_rows["train"]])

    if args.smoke_test:
        model = build_model(args.seed)
        compile_model(model, args.head_learning_rate)
        images, labels = next(iter(train_dataset))
        result = model.train_on_batch(images, labels, return_dict=True)
        predictions = model(images[:2], training=False).numpy().reshape(-1)
        print(json.dumps({"status": "passed", "batch_shape": list(images.shape),
                          "train_on_batch": {key: json_number(value) for key, value in result.items()},
                          "prediction_count": len(predictions)}, indent=2))
        return {"status": "passed"}

    args.output.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output / "plots"
    plots_dir.mkdir(exist_ok=True)
    head_path = args.output / "head_best.keras"
    fine_path = args.output / "finetuned_best.keras"

    model = build_model(args.seed)
    head_history = fit_stage(
        model, train_dataset, validation_dataset, weights, head_path,
        args.head_epochs, args.patience, args.head_learning_rate,
    )
    model = tf.keras.models.load_model(head_path, compile=False)
    trainable_layers = enable_fine_tuning(model, args.trainable_tail)
    fine_history = fit_stage(
        model, train_dataset, validation_dataset, weights, fine_path,
        args.fine_tune_epochs, args.patience, args.fine_tune_learning_rate,
    )

    head_validation = stage_validation(head_path, validation_dataset, split_rows["validation"])
    fine_validation = stage_validation(fine_path, validation_dataset, split_rows["validation"])
    stages = {"frozen_head": head_validation, "fine_tuned": fine_validation}
    selected_stage = max(stages, key=lambda name: (stages[name]["roc_auc"], -stages[name]["log_loss"]))
    selected = stages[selected_stage]
    threshold, threshold_rows = select_threshold(selected["labels"], selected["scores"])

    final_path = args.output / "final_model.keras"
    selected_source = head_path if selected_stage == "frozen_head" else fine_path
    shutil.copy2(selected_source, final_path)
    export_path = args.output / "saved_model"
    if export_path.exists():
        shutil.rmtree(export_path)
    selected["model"].export(export_path)

    reloaded = tf.keras.models.load_model(final_path, compile=False)
    _, reload_scores = predict_rows(reloaded, validation_dataset, split_rows["validation"])
    reload_max_difference = float(np.max(np.abs(reload_scores - selected["scores"])))
    if reload_max_difference > 1e-6:
        raise ValueError(f"Saved-model reload changed predictions by {reload_max_difference}")

    history_rows = combine_histories([("frozen_head", head_history), ("fine_tuned", fine_history)])
    history_fields = list(history_rows[0])
    write_csv(args.output / "training_history.csv", history_rows, history_fields)
    threshold_fields = list(threshold_rows[0])
    write_csv(args.output / "validation_threshold_curve.csv", threshold_rows, threshold_fields)
    prediction_rows = []
    for row, score in zip(split_rows["validation"], selected["scores"]):
        prediction_rows.append({
            "filename": row["filename"],
            "label": row["label"],
            "pterygium_score": round(float(score), 8),
            "prediction": int(score >= threshold["threshold"]),
            "threshold": round(threshold["threshold"], 8),
        })
    write_csv(
        args.output / "validation_predictions.csv", prediction_rows,
        ["filename", "label", "pterygium_score", "prediction", "threshold"],
    )
    plot_history(history_rows, plots_dir / "training-history.png")
    plot_threshold(threshold_rows, threshold, plots_dir / "validation-threshold.png")

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "tensorflow": tf.__version__,
        "seed": args.seed,
        "input_size": [224, 224, 3],
        "batch_size": args.batch_size,
        "head_epochs_max": args.head_epochs,
        "fine_tune_epochs_max": args.fine_tune_epochs,
        "early_stopping_patience": args.patience,
        "head_learning_rate": args.head_learning_rate,
        "fine_tune_learning_rate": args.fine_tune_learning_rate,
        "trainable_backbone_tail_requested": args.trainable_tail,
        "trainable_backbone_layers_actual": trainable_layers,
        "dropout": 0.30,
        "class_weights": {str(key): value for key, value in weights.items()},
        "augmentation": {
            "horizontal_flip": True,
            "rotation_degrees_approx": 10,
            "translation_fraction": 0.05,
            "zoom_fraction": 0.08,
            "contrast_fraction": 0.10,
            "brightness_fraction": 0.08,
        },
        "threshold_policy": "Maximum validation balanced accuracy; ties prefer sensitivity, specificity, then proximity to 0.5.",
        "test_set_used_for_training_or_selection": False,
    }
    (args.output / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "passed",
        "selected_stage": selected_stage,
        "selected_threshold": threshold["threshold"],
        "validation_roc_auc": selected["roc_auc"],
        "validation_log_loss": selected["log_loss"],
        "validation_sensitivity": threshold["sensitivity"],
        "validation_specificity": threshold["specificity"],
        "validation_balanced_accuracy": threshold["balanced_accuracy"],
        "validation_confusion": {
            key: threshold[key] for key in ("true_positive", "true_negative", "false_positive", "false_negative")
        },
        "frozen_head_validation_roc_auc": head_validation["roc_auc"],
        "fine_tuned_validation_roc_auc": fine_validation["roc_auc"],
        "frozen_head_epochs_run": len(head_history.history["loss"]),
        "fine_tune_epochs_run": len(fine_history.history["loss"]),
        "saved_model_reload_max_prediction_difference": reload_max_difference,
        "train_images": len(split_rows["train"]),
        "validation_images": len(split_rows["validation"]),
        "test_images_reserved": len(split_rows["test"]),
        "test_set_evaluated": False,
    }
    (args.output / "phase3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase 3: MobileNetV2 training", "",
        "Status: **passed**", "",
        "## Training design", "",
        f"MobileNetV2 used ImageNet weights with a global-average-pooling layer, 30% dropout and one sigmoid output. "
        f"The frozen-head stage ran for {summary['frozen_head_epochs_run']} epochs and the fine-tuning stage ran for "
        f"{summary['fine_tune_epochs_run']} epochs. {trainable_layers} final non-BatchNormalization backbone layers "
        "were trainable during fine-tuning.", "",
        f"Balanced class weights were normal={weights[0]:.4f} and pterygium={weights[1]:.4f}. Augmentation was applied "
        "only to training images. The validation and test images were never augmented.", "",
        "## Model selection", "",
        f"Frozen-head validation ROC-AUC: {head_validation['roc_auc']:.4f}. Fine-tuned validation ROC-AUC: "
        f"{fine_validation['roc_auc']:.4f}. Selected stage: **{selected_stage}**.", "",
        f"The validation-only threshold is **{threshold['threshold']:.6f}**. At that threshold, validation sensitivity "
        f"is {threshold['sensitivity']:.3f}, specificity is {threshold['specificity']:.3f}, and balanced accuracy is "
        f"{threshold['balanced_accuracy']:.3f}.", "",
        "## Data separation", "",
        f"Training used {len(split_rows['train'])} images and model selection used {len(split_rows['validation'])} images. "
        f"All {len(split_rows['test'])} test images remain reserved for Phase 4. No test prediction or test metric was "
        "generated in this phase.", "",
        "## Saved outputs", "",
        "- `final_model.keras`", "- `saved_model/`", "- `training_config.json`",
        "- `training_history.csv`", "- `validation_predictions.csv`",
        "- `validation_threshold_curve.csv`", "- `plots/training-history.png`",
        "- `plots/validation-threshold.png`", "",
        f"Reloading the saved Keras model changed validation predictions by at most {reload_max_difference:.2e}.", "",
        "These are development results from a small slit-lamp dataset. They are not clinical performance claims and "
        "do not establish accuracy on smartphone photographs.", "",
    ]
    (args.output / "phase3-report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2", type=Path, default=ROOT / "artifacts/phase2")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/phase3")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--trainable-tail", type=int, default=30)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.head_epochs, args.fine_tune_epochs, args.patience, args.trainable_tail) < 1:
        raise ValueError("Batch size, epochs, patience and trainable tail must be positive")
    result = train(args)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
