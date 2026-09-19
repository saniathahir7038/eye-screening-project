"""Prepare, review, and split the audited SLID binary pterygium subset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
CLASS_NAMES = {0: "no_visible_pterygium_annotation", 1: "pterygium_annotation_present"}


@dataclass
class UnionFind:
    parent: dict[str, str]

    @classmethod
    def create(cls, names):
        return cls({name: name for name in names})

    def find(self, name):
        root = name
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[name] != name:
            parent = self.parent[name]
            self.parent[name] = root
            name = parent
        return root

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def safe_source(images_dir, relative_path):
    root = images_dir.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Unsafe or missing image path: {relative_path}")
    return path


def detect_top_left_black_box(image, threshold=16, minimum_fraction=0.03):
    """Return (width, height) for a solid, near-black rectangle anchored at (0, 0)."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    mask = rgb.max(axis=2) <= threshold
    if not mask[0, 0]:
        return 0, 0
    box_width = int(np.argmax(~mask[0])) if (~mask[0]).any() else width
    box_height = int(np.argmax(~mask[:, 0])) if (~mask[:, 0]).any() else height
    if box_width < width * minimum_fraction or box_height < height * minimum_fraction:
        return 0, 0
    if mask[:box_height, :box_width].mean() < 0.98:
        return 0, 0
    return box_width, box_height


def preprocess_image(image, output_size=224, corner_mask_fraction=0.20):
    """Standardize the information-box region while preserving the full ocular field."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    black_width, black_height = detect_top_left_black_box(rgb)
    mask_width = max(1, int(np.ceil(width * corner_mask_fraction)))
    mask_height = max(1, int(np.ceil(height * corner_mask_fraction)))
    standardized = rgb.copy()
    ImageDraw.Draw(standardized).rectangle(
        (0, 0, mask_width - 1, mask_height - 1), fill=(127, 127, 127)
    )
    crop_box = (0, 0, width, height)
    processed = standardized.resize((output_size, output_size), Image.Resampling.BILINEAR)
    return processed, {
        "black_box_width": black_width,
        "black_box_height": black_height,
        "corner_mask_width": mask_width,
        "corner_mask_height": mask_height,
        "corner_mask_fraction": corner_mask_fraction,
        "crop_left": 0,
        "crop_top": 0,
        "crop_right": width,
        "crop_bottom": height,
    }


def map_boxes(boxes, crop, output_size):
    mapped, clipped = [], 0
    crop_width = crop["crop_right"] - crop["crop_left"]
    crop_height = crop["crop_bottom"] - crop["crop_top"]
    for box in boxes:
        x1 = (box["x"] - crop["crop_left"]) * output_size / crop_width
        y1 = (box["y"] - crop["crop_top"]) * output_size / crop_height
        x2 = (box["x"] + box["width"] - crop["crop_left"]) * output_size / crop_width
        y2 = (box["y"] + box["height"] - crop["crop_top"]) * output_size / crop_height
        bounded = [max(0, min(output_size, value)) for value in (x1, y1, x2, y2)]
        if any(abs(a - b) > 1e-6 for a, b in zip((x1, y1, x2, y2), bounded)):
            clipped += 1
        if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
            raise ValueError("Preprocessing removed an entire pterygium annotation")
        mapped.append({
            "x": round(bounded[0], 3),
            "y": round(bounded[1], 3),
            "width": round(bounded[2] - bounded[0], 3),
            "height": round(bounded[3] - bounded[1], 3),
        })
    return mapped, clipped


def image_metrics(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    vertical = np.diff(gray, axis=0)
    horizontal = np.diff(gray, axis=1)
    return {
        "brightness_mean": round(float(gray.mean()), 3),
        "contrast_std": round(float(gray.std()), 3),
        "edge_energy": round(float(vertical.var() + horizontal.var()), 3),
    }


def read_binary_manifest(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"filename", "image_path", "label", "pterygium_regions", "pixel_sha256", "issues"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Binary manifest is empty or missing required columns")
    names = [row["filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Binary manifest contains duplicate filenames")
    for row in rows:
        if row["label"] not in {"0", "1"} or row["issues"]:
            raise ValueError(f"Ineligible binary manifest row: {row['filename']}")
    return rows


def load_decisions(path):
    if not path.exists():
        return {"confirmed_pairs": [], "rejected_pairs": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("confirmed_pairs", "rejected_pairs"):
        if key not in data or not all(isinstance(pair, list) and len(pair) == 2 for pair in data[key]):
            raise ValueError(f"Invalid near-duplicate decision file: {key}")
    return data


def max_shifted_correlation(first, second, maximum_shift=4):
    """Compare small grayscale previews while allowing minor capture translations."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    best = -1.0
    for y_shift in range(-maximum_shift, maximum_shift + 1):
        for x_shift in range(-maximum_shift, maximum_shift + 1):
            first_y = slice(max(0, y_shift), min(first.shape[0], first.shape[0] + y_shift))
            first_x = slice(max(0, x_shift), min(first.shape[1], first.shape[1] + x_shift))
            second_y = slice(max(0, -y_shift), min(second.shape[0], second.shape[0] - y_shift))
            second_x = slice(max(0, -x_shift), min(second.shape[1], second.shape[1] - x_shift))
            a = first[first_y, first_x].ravel()
            b = second[second_y, second_x].ravel()
            a -= a.mean()
            b -= b.mean()
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator:
                best = max(best, float(np.dot(a, b) / denominator))
    return best


def filename_distance(first, second):
    first_stem, second_stem = Path(first).stem, Path(second).stem
    if first_stem.isdigit() and second_stem.isdigit():
        return abs(int(first_stem) - int(second_stem))
    return None


def duplicate_candidates(records, decisions, review_phash=8, review_dhash=10,
                         review_correlation=0.97):
    by_name = {row["filename"]: row for row in records}
    union = UnionFind.create(by_name)
    decision = {}
    for pair in decisions["confirmed_pairs"]:
        decision[tuple(sorted(pair))] = "confirmed"
    for pair in decisions["rejected_pairs"]:
        decision[tuple(sorted(pair))] = "rejected"
    if len(decision) != len(decisions["confirmed_pairs"]) + len(decisions["rejected_pairs"]):
        raise ValueError("A near-duplicate pair cannot be both confirmed and rejected")
    for pair, value in decision.items():
        if any(name not in by_name for name in pair):
            raise ValueError(f"Unknown filename in near-duplicate decisions: {pair}")
        if value == "confirmed":
            union.union(*pair)

    candidates = []
    for index, first in enumerate(records):
        for second in records[index + 1:]:
            exact = bool(first["pixel_sha256"] == second["pixel_sha256"])
            phash_distance = first["phash_obj"] - second["phash_obj"]
            dhash_distance = first["dhash_obj"] - second["dhash_obj"]
            pair = tuple(sorted((first["filename"], second["filename"])))
            manual = decision.get(pair, "")
            relaxed = phash_distance <= review_phash and dhash_distance <= review_dhash
            distance = filename_distance(*pair)
            correlation = (max_shifted_correlation(first["similarity_array"], second["similarity_array"])
                           if exact or relaxed or manual else -1.0)
            review_candidate = relaxed and (distance == 1 or correlation >= review_correlation)
            if exact or review_candidate or manual:
                same_label = first["label"] == second["label"]
                if exact:
                    status = "exact_grouped"
                elif manual == "confirmed":
                    status = "confirmed_grouped"
                elif manual == "rejected":
                    status = "reviewed_not_grouped"
                else:
                    status = "review_required"
                if exact or manual == "confirmed":
                    union.union(*pair)
                candidates.append({
                    "first_filename": first["filename"],
                    "second_filename": second["filename"],
                    "first_label": first["label"],
                    "second_label": second["label"],
                    "phash_distance": phash_distance,
                    "dhash_distance": dhash_distance,
                    "filename_distance": "" if distance is None else distance,
                    "shifted_correlation": round(correlation, 6),
                    "exact_pixels": int(exact),
                    "same_label": int(same_label),
                    "review_status": status,
                })
    groups = defaultdict(list)
    for name in by_name:
        groups[union.find(name)].append(name)
    for members in groups.values():
        labels = {by_name[name]["label"] for name in members}
        if len(labels) != 1:
            raise ValueError(f"Duplicate group contains conflicting labels: {sorted(members)}")
        group_id = "group_" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:12]
        for name in members:
            by_name[name]["similarity_group"] = group_id
            by_name[name]["similarity_group_size"] = len(members)
    return candidates


def choose_groups(groups, target, rng):
    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=lambda group: len(group), reverse=True)
    choices = {0: []}
    for index, group in enumerate(groups):
        size = len(group)
        for total, selected in sorted(list(choices.items()), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in choices:
                choices[new_total] = selected + [index]
        if target in choices:
            break
    selected_indexes = set(choices.get(target, choices[max(choices)]))
    selected = [group for index, group in enumerate(groups) if index in selected_indexes]
    remaining = [group for index, group in enumerate(groups) if index not in selected_indexes]
    return selected, remaining


def assign_splits(records, seed=20260916):
    target = {
        1: {"train": 114, "validation": 24, "test": 25},
        0: {"train": 172, "validation": 37, "test": 36},
    }
    by_name = {row["filename"]: row for row in records}
    grouped = defaultdict(list)
    for row in records:
        grouped[row["similarity_group"]].append(row["filename"])
    for label in (0, 1):
        groups = [members for members in grouped.values()
                  if int(by_name[members[0]]["label"]) == label]
        rng = random.Random(seed + label)
        train, remaining = choose_groups(groups, target[label]["train"], rng)
        validation, test = choose_groups(remaining, target[label]["validation"], rng)
        for split, split_groups in zip(SPLITS, (train, validation, test)):
            for group in split_groups:
                for name in group:
                    by_name[name]["split"] = split
    if any("split" not in row for row in records):
        raise ValueError("Not every record received a split")
    return target


def write_csv(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in fields})


def natural_key(name):
    stem = Path(name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def quantile_samples(records, count):
    ordered = sorted(records, key=lambda row: natural_key(row["filename"]))
    if len(ordered) <= count:
        return ordered
    return [ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)]


def fitted(image, size):
    canvas = Image.new("RGB", size, "white")
    thumb = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
    return canvas


def make_crop_sheet(records, images_dir, processed_dir, path):
    selected = quantile_samples([r for r in records if r["label"] == "0"], 8)
    selected += quantile_samples([r for r in records if r["label"] == "1"], 8)
    tile_width, tile_height = 440, 230
    sheet = Image.new("RGB", (tile_width * 2, tile_height * 8), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, row in enumerate(selected):
        col, line = index % 2, index // 2
        x, y = col * tile_width, line * tile_height
        original = Image.open(safe_source(images_dir, row["image_path"]))
        processed = Image.open(processed_dir / row["filename"])
        sheet.paste(fitted(original, (205, 180)), (x + 5, y + 30))
        sheet.paste(fitted(processed, (205, 180)), (x + 225, y + 30))
        draw.text((x + 5, y + 5), f"{row['filename']} label={row['label']} original | processed", fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def make_overlay_sheet(records, images_dir, path):
    positives = quantile_samples([r for r in records if r["label"] == "1"], 12)
    tile_width, tile_height = 420, 300
    sheet = Image.new("RGB", (tile_width * 3, tile_height * 4), "white")
    font = ImageFont.load_default(size=16)
    for index, row in enumerate(positives):
        image = Image.open(safe_source(images_dir, row["image_path"])).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in json.loads(row["pterygium_regions"]):
            draw.rectangle((box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]),
                           outline="red", width=max(3, image.width // 500))
        thumb = fitted(image, (400, 255))
        x, y = (index % 3) * tile_width, (index // 3) * tile_height
        sheet.paste(thumb, (x + 10, y + 35))
        ImageDraw.Draw(sheet).text((x + 10, y + 8), row["filename"], fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def make_duplicate_sheets(candidates, records, images_dir, output_dir):
    by_name = {row["filename"]: row for row in records}
    pages = []
    for page_number in range((len(candidates) + 7) // 8):
        page_rows = candidates[page_number * 8:(page_number + 1) * 8]
        sheet = Image.new("RGB", (900, 260 * len(page_rows)), "white")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=16)
        for line, candidate in enumerate(page_rows):
            y = line * 260
            first = by_name[candidate["first_filename"]]
            second = by_name[candidate["second_filename"]]
            first_image = Image.open(safe_source(images_dir, first["image_path"]))
            second_image = Image.open(safe_source(images_dir, second["image_path"]))
            sheet.paste(fitted(first_image, (420, 210)), (10, y + 40))
            sheet.paste(fitted(second_image, (420, 210)), (470, y + 40))
            title = (f"{first['filename']} label={first['label']} | {second['filename']} label={second['label']} "
                     f"pHash={candidate['phash_distance']} dHash={candidate['dhash_distance']} "
                     f"corr={candidate['shifted_correlation']:.3f} "
                     f"{candidate['review_status']}")
            draw.text((10, y + 10), title, fill="black", font=font)
        path = output_dir / f"near-duplicate-review-{page_number + 1:02d}.png"
        sheet.save(path)
        pages.append(path.name)
    return pages


def prepare(args):
    source_rows = read_binary_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    processed_dir = args.output / "processed_images"
    processed_dir.mkdir(parents=True, exist_ok=True)
    records = []
    boxes_clipped = 0
    for source in sorted(source_rows, key=lambda row: natural_key(row["filename"])):
        source_path = safe_source(args.images, source["image_path"])
        with Image.open(source_path) as image:
            image.load()
            original_width, original_height = image.size
            processed, crop = preprocess_image(image, args.size)
        processed_path = processed_dir / source["filename"]
        processed.save(processed_path, format="PNG", optimize=True)
        boxes = json.loads(source["pterygium_regions"])
        mapped_boxes, clipped = map_boxes(boxes, crop, args.size)
        boxes_clipped += clipped
        metrics = image_metrics(processed)
        record = {
            **source,
            "class_name": CLASS_NAMES[int(source["label"])],
            "original_width": original_width,
            "original_height": original_height,
            **crop,
            "processed_width": args.size,
            "processed_height": args.size,
            "processed_image_path": f"processed_images/{source['filename']}",
            "processed_pterygium_regions": json.dumps(mapped_boxes, separators=(",", ":")),
            "phash_obj": imagehash.phash(processed),
            "dhash_obj": imagehash.dhash(processed),
            "similarity_array": np.asarray(
                processed.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32
            ),
            **metrics,
        }
        record["phash"] = str(record["phash_obj"])
        record["dhash"] = str(record["dhash_obj"])
        records.append(record)

    decisions = load_decisions(args.decisions)
    candidates = duplicate_candidates(records, decisions)
    targets = assign_splits(records, args.seed)
    records.sort(key=lambda row: natural_key(row["filename"]))
    candidate_fields = ["first_filename", "second_filename", "first_label", "second_label",
                        "phash_distance", "dhash_distance", "filename_distance", "shifted_correlation",
                        "exact_pixels", "same_label", "review_status"]
    write_csv(args.output / "near_duplicate_candidates.csv", candidates, candidate_fields)
    fields = ["filename", "label", "class_name", "split", "similarity_group", "similarity_group_size",
              "image_path", "processed_image_path", "original_width", "original_height",
              "black_box_width", "black_box_height", "corner_mask_width", "corner_mask_height",
              "corner_mask_fraction", "crop_left", "crop_top", "crop_right", "crop_bottom",
              "processed_width", "processed_height", "pterygium_regions", "processed_pterygium_regions",
              "pixel_sha256", "phash", "dhash", "brightness_mean", "contrast_std", "edge_energy"]
    write_csv(args.output / "split_manifest.csv", records, fields)
    for split in SPLITS:
        write_csv(args.output / f"{split}.csv", [row for row in records if row["split"] == split], fields)

    review_dir = args.output / "review"
    review_dir.mkdir(exist_ok=True)
    make_crop_sheet(records, args.images, processed_dir, review_dir / "crop-review.png")
    make_overlay_sheet(records, args.images, review_dir / "pterygium-overlay-review.png")
    duplicate_pages = make_duplicate_sheets(candidates, records, args.images, review_dir)

    counts = {split: {str(label): sum(row["split"] == split and row["label"] == str(label)
                                     for row in records) for label in (0, 1)} for split in SPLITS}
    for split in SPLITS:
        counts[split]["total"] = counts[split]["0"] + counts[split]["1"]
    groups = defaultdict(list)
    for row in records:
        groups[row["similarity_group"]].append(row)
    group_split_violations = [group for group, members in groups.items()
                              if len({row["split"] for row in members}) != 1]
    unresolved = [row for row in candidates if row["review_status"] == "review_required"]
    detected = [row for row in records if row["black_box_height"]]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "output_size": args.size,
        "selected_images": len(records),
        "class_counts": dict(Counter(row["label"] for row in records)),
        "split_counts": counts,
        "target_counts": {str(label): values for label, values in targets.items()},
        "black_box_detected_images": len(detected),
        "standardized_corner_images": len(records),
        "corner_mask_fraction": 0.20,
        "black_box_height_ratio_min": round(min(row["black_box_height"] / row["original_height"] for row in detected), 6),
        "black_box_height_ratio_max": round(max(row["black_box_height"] / row["original_height"] for row in detected), 6),
        "pterygium_boxes_clipped_by_crop": boxes_clipped,
        "similarity_groups": len(groups),
        "multi_image_similarity_groups": sum(len(members) > 1 for members in groups.values()),
        "duplicate_candidates": len(candidates),
        "unresolved_review_candidates": len(unresolved),
        "cross_label_review_candidates": sum(row["first_label"] != row["second_label"] for row in unresolved),
        "group_split_violations": group_split_violations,
        "review_images": ["crop-review.png", "pterygium-overlay-review.png", *duplicate_pages],
        "status": "needs_review" if unresolved or group_split_violations or boxes_clipped else "passed",
    }
    (args.output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase 2: preprocessing and dataset split", "",
        f"Status: **{summary['status']}**", "",
        "## Dataset split", "",
        "| Split | Pterygium | Normal candidate | Total |", "|---|---:|---:|---:|",
        *[f"| {split.title()} | {counts[split]['1']} | {counts[split]['0']} | {counts[split]['total']} |"
          for split in SPLITS], "",
        "## Preprocessing", "",
        f"A fixed neutral mask was applied to the upper-left 20% × 20% region of all {len(records)} images, "
        f"then each full image was converted to RGB and resized to {args.size} × {args.size} pixels. The source "
        f"information box was detected descriptively in {len(detected)} images and occupied "
        f"{summary['black_box_height_ratio_min']:.1%} to {summary['black_box_height_ratio_max']:.1%} of image height.", "",
        "The full ocular field was retained. This avoids the lesion clipping observed when removing the complete "
        "top band and prevents information-box presence from becoming a class shortcut.", "",
        f"Pterygium boxes clipped by preprocessing: {boxes_clipped}.", "",
        "Pterygium boxes were used only for overlay review and coordinate mapping. They were not used to choose "
        "class-specific crops or to create the classification input.", "",
        "## Duplicate control", "",
        f"Similarity groups: {len(groups)}. Multi-image groups: {summary['multi_image_similarity_groups']}. "
        f"Candidate pairs: {len(candidates)}. Unresolved review candidates: {len(unresolved)}. Exact pixel matches "
        "and manually confirmed visually repeated captures were grouped; perceptual hashes alone never grouped images.", "",
        "All members of each accepted similarity group were assigned to the same split. The public data does not "
        "contain patient identifiers, so this reduces visual leakage but cannot establish a patient-independent split.", "",
        "## Reproducibility", "",
        f"Random seed: {args.seed}. Exact class targets were used when compatible with similarity groups. "
        "The split manifests reference the processed images and retain source hashes, crop coordinates, mapped lesion "
        "boxes, perceptual hashes and basic image-quality measurements.", "",
        "## Review artifacts", "",
        *[f"- `{name}`" for name in summary["review_images"]], "",
        "The quality measurements are descriptive. No image was excluded using an unvalidated brightness, contrast or "
        "sharpness threshold.", "",
    ]
    (args.output / "phase2-report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/audit/binary_labels.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/images")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/phase2")
    parser.add_argument("--decisions", type=Path, default=ROOT / "near_duplicate_decisions.json")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    if args.size < 32:
        raise ValueError("Output size must be at least 32 pixels")
    summary = prepare(args)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
