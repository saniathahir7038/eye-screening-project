"""Exploratory head adaptation with separate external train/validation/test groups.

The external dataset has been inspected as a whole, so this is an engineering
experiment, not an unbiased clinical validation. Never overwrite the accepted
model or APK from this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from framing import auto_frame
from prepare_dataset import preprocess_image
from screening import decode_uploaded_image, BASELINE_MODEL_PATH
from training_utils import confusion_values


ROOT = Path(__file__).resolve().parent
SEED = 20261002


def external_images(root: Path):
    records = []
    seen = set()
    for folder, label in (("normal", 0), ("pterygium", 1)):
        for path in sorted((root / folder).iterdir()):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            image = decode_uploaded_image(path.read_bytes())
            digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode()
                                    + image.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            framed = auto_frame(image).image
            processed, _ = preprocess_image(framed, 224, 0.20)
            records.append({
                "path": f"{folder}/{path.name}", "label": label,
                "pixels": np.asarray(processed, dtype=np.float32),
                "phash": imagehash.phash(processed),
                "dhash": imagehash.dhash(processed),
            })
    return records


def group_near_duplicates(records):
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if (records[left]["phash"] - records[right]["phash"] <= 8
                    and records[left]["dhash"] - records[right]["dhash"] <= 10):
                parent[find(right)] = find(left)
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(index)
    if any(len({records[index]["label"] for index in group}) != 1
           for group in groups.values()):
        raise ValueError("Near-duplicate group crosses external class labels")
    return list(groups.values())


def split_external(records):
    groups = group_near_duplicates(records)
    rng = np.random.default_rng(SEED)
    splits = {"train": [], "validation": [], "test": []}
    for label in (0, 1):
        class_groups = [group for group in groups if records[group[0]]["label"] == label]
        rng.shuffle(class_groups)
        n_test = max(1, round(len(class_groups) * 0.20))
        n_validation = max(1, round(len(class_groups) * 0.20))
        for name, selected in (("test", class_groups[:n_test]),
                               ("validation", class_groups[n_test:n_test + n_validation]),
                               ("train", class_groups[n_test + n_validation:])):
            for group in selected:
                splits[name].extend(group)
    return splits


def slid_images(split):
    manifest = ROOT / "artifacts/phase2" / f"{split}.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for row in rows:
        path = ROOT / "artifacts/phase2" / row["processed_image_path"]
        with Image.open(path) as image:
            records.append({"path": row["filename"], "label": int(row["label"]),
                            "pixels": np.asarray(image.convert("RGB"), dtype=np.float32)})
    return records


def embeddings(model, records, batch_size=16):
    features = []
    for start in range(0, len(records), batch_size):
        batch = np.stack([row["pixels"] for row in records[start:start + batch_size]])
        features.append(model.predict(batch, verbose=0))
    return np.concatenate(features)


def metrics(labels, scores, threshold):
    result = confusion_values(labels, scores, threshold)
    result["count"] = len(labels)
    result["accuracy"] = (result["true_positive"] + result["true_negative"]) / len(labels)
    return result


def threshold_candidates(*score_sets):
    values = np.unique(np.concatenate(score_sets))
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], (values[:-1] + values[1:]) / 2)))


def choose_threshold(external_labels, external_scores, slid_labels, slid_scores):
    best = None
    for threshold in threshold_candidates(external_scores, slid_scores):
        external = metrics(external_labels, external_scores, threshold)
        slid = metrics(slid_labels, slid_scores, threshold)
        # Screening experiment: preserve at least 90% sensitivity on *both*
        # validation domains before optimizing false-positive reduction.
        meets_sensitivity_floor = min(external["sensitivity"], slid["sensitivity"]) >= 0.90
        key = (meets_sensitivity_floor,
               min(external["specificity"], slid["specificity"]),
               min(external["balanced_accuracy"], slid["balanced_accuracy"]),
               external["sensitivity"] + slid["sensitivity"])
        if best is None or key > best[0]:
            best = (key, float(threshold), external, slid)
    return best


def train_and_evaluate(data_dir: Path, output_dir: Path):
    tf.keras.utils.set_random_seed(SEED)
    external = external_images(data_dir)
    splits = split_external(external)
    slid_train = slid_images("train")
    slid_validation = slid_images("validation")
    model = tf.keras.models.load_model(BASELINE_MODEL_PATH, compile=False)
    extractor = tf.keras.Model(model.input, model.get_layer("global_average_pooling").output)
    external_features = embeddings(extractor, external)
    slid_train_features = embeddings(extractor, slid_train)
    slid_validation_features = embeddings(extractor, slid_validation)
    ext_labels = np.array([row["label"] for row in external], dtype=np.int32)
    slid_train_labels = np.array([row["label"] for row in slid_train], dtype=np.int32)
    slid_val_labels = np.array([row["label"] for row in slid_validation], dtype=np.int32)
    train_indices = np.array(splits["train"])
    val_indices = np.array(splits["validation"])
    test_indices = np.array(splits["test"])
    x_train = np.concatenate((slid_train_features, external_features[train_indices]))
    y_train = np.concatenate((slid_train_labels, ext_labels[train_indices]))
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    ext_val = scaler.transform(external_features[val_indices])
    slid_val = scaler.transform(slid_validation_features)

    best = None
    for strength in (0.01, 0.1, 1.0):
        for external_weight in (1.0, 3.0, 8.0):
            weights = np.concatenate((np.ones(len(slid_train)),
                                      np.full(len(train_indices), external_weight)))
            classifier = LogisticRegression(C=strength, max_iter=2000, class_weight="balanced")
            classifier.fit(x_train_scaled, y_train, sample_weight=weights)
            ext_scores = classifier.predict_proba(ext_val)[:, 1]
            slid_scores = classifier.predict_proba(slid_val)[:, 1]
            selected = choose_threshold(ext_labels[val_indices], ext_scores,
                                        slid_val_labels, slid_scores)
            key, threshold, ext_metrics, slid_metrics = selected
            if best is None or key > best[0]:
                best = (key, classifier, strength, external_weight, threshold,
                        ext_metrics, slid_metrics)

    _, classifier, strength, external_weight, threshold, ext_val_metrics, slid_val_metrics = best
    test_scores = classifier.predict_proba(scaler.transform(external_features[test_indices]))[:, 1]
    ext_test_metrics = metrics(ext_labels[test_indices], test_scores, threshold)
    old_test_scores = model.predict(np.stack([external[index]["pixels"] for index in test_indices]),
                                    verbose=0).reshape(-1)
    old_test_metrics = metrics(ext_labels[test_indices], old_test_scores, 0.18100688606500626)
    old_slid_scores = model.predict(np.stack([row["pixels"] for row in slid_validation]),
                                    verbose=0).reshape(-1)
    old_slid_metrics = metrics(slid_val_labels, old_slid_scores, 0.18100688606500626)

    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients = classifier.coef_[0] / scaler.scale_
    bias = float(classifier.intercept_[0] - np.dot(coefficients, scaler.mean_))
    model.get_layer("pterygium_score").set_weights((coefficients.astype(np.float32)[:, None],
                                                  np.array([bias], dtype=np.float32)))
    embedded_scores = model.predict(np.stack([external[index]["pixels"] for index in test_indices]),
                                    verbose=0).reshape(-1)
    max_embedding_difference = float(np.max(np.abs(embedded_scores - test_scores)))
    if max_embedding_difference > 1e-4:
        raise AssertionError(f"Integrated head differs from trained classifier: {max_embedding_difference}")
    candidate_path = output_dir / "candidate.keras"
    model.save(candidate_path)
    result = {
        "note": "Exploratory adaptation, not independent clinical validation",
        "source_model": str(BASELINE_MODEL_PATH), "candidate_model": str(candidate_path),
        "seed": SEED, "regularization_C": strength,
        "external_train_weight": external_weight, "threshold": threshold,
        "threshold_selection_rule": "At least 90% sensitivity in both validation domains, then maximize minimum specificity",
        "integrated_head_max_score_difference": max_embedding_difference,
        "external_unique_count": len(external),
        "external_group_count": len(group_near_duplicates(external)),
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "split_files": {name: [external[index]["path"] for index in indices]
                        for name, indices in splits.items()},
        "external_validation": ext_val_metrics,
        "slid_validation": slid_val_metrics,
        "external_test_candidate": ext_test_metrics,
        "external_test_old_model": old_test_metrics,
        "slid_validation_old_model": old_slid_metrics,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
