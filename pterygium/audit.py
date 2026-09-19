"""Audit SLID and derive image-level labels without preprocessing or splitting."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent
REQUIRED = {"filename", "file_size", "annotation_count", "annotation_ID",
            "attributes", "shape_coordinates"}
FIELDS = ["filename", "image_path", "label", "label_basis", "lesions",
          "annotation_rows", "pterygium_regions", "width", "height", "mode",
          "file_sha256", "pixel_sha256", "duplicate_group", "issues"]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_annotations(path):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError("Annotations CSV is missing required columns")
        for line, row in enumerate(reader, 2):
            name = row["filename"]
            if (not name or "/" in name or "\\" in name or ":" in name
                    or name in {".", ".."}):
                raise ValueError(f"Unsafe or empty filename on CSV line {line}")
            # Malformed rows fail closed rather than becoming normal controls.
            try:
                attrs = json.loads(row["attributes"])
                shape = json.loads(row["shape_coordinates"])
                if not isinstance(attrs, dict) or not isinstance(shape, dict):
                    raise ValueError("Attributes and coordinates must be objects")
                if not all(isinstance(attrs.get(k), str) for k in ("lesion", "region")):
                    raise ValueError("Missing lesion/region string")
                row = {**row, "attrs": attrs, "shape": shape,
                       "count": int(row["annotation_count"]),
                       "id": int(row["annotation_ID"]), "bytes": int(row["file_size"])}
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid annotation on CSV line {line}: {error}") from error
            grouped[name].append(row)
    if not grouped:
        raise ValueError("Annotations CSV is empty")
    return grouped


def classify(rows):
    lesions = sorted({r["attrs"]["lesion"].strip() for r in rows} - {""})
    if "Pterygium" in lesions:
        return 1, "any_pterygium_annotation", lesions
    if not lesions:
        return 0, "no_lesion_annotations", lesions
    return "", "other_disease_only", lesions


def valid_box(shape, width, height):
    if shape.get("name") != "rect":
        return False
    vals = [shape.get(k) for k in ("x", "y", "width", "height")]
    if any(type(v) not in (int, float) for v in vals):
        return False
    x, y, w, h = vals
    return 0 <= x < width and 0 <= y < height and w > 0 and h > 0 and x+w <= width and y+h <= height


def audit(annotations, images, output):
    grouped = read_annotations(annotations)
    if not images.is_dir():
        raise ValueError(f"Extracted image directory is missing: {images}")
    index = defaultdict(list)
    metadata_files = 0
    for path in sorted(images.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".png":
            if "__MACOSX" in path.relative_to(images).parts and path.name.startswith("._"):
                with path.open("rb") as stream:
                    if stream.read(4) == b"\x00\x05\x16\x07":
                        metadata_files += 1
                        continue
            index[path.name].append(path)
    records = []
    for name, rows in sorted(grouped.items()):
        label, basis, lesions = classify(rows)
        issues = []
        if {r["count"] for r in rows} != {len(rows)}:
            issues.append("annotation_count_mismatch")
        if sorted(r["id"] for r in rows) != list(range(len(rows))):
            issues.append("annotation_ids_inconsistent")
        boxes = [r["shape"] for r in rows if r["attrs"]["lesion"].strip() == "Pterygium"]
        record = dict.fromkeys(FIELDS, "")
        record.update(filename=name, label=label, label_basis=basis,
                      lesions=json.dumps(lesions), annotation_rows=len(rows),
                      pterygium_regions=json.dumps(boxes))
        paths = index.get(name, [])
        if len(paths) != 1:
            issues.append("missing_image" if not paths else "ambiguous_image_filename")
        else:
            path = paths[0]
            record["image_path"] = path.relative_to(images).as_posix()
            record["file_sha256"] = sha256(path)
            if {r["bytes"] for r in rows} != {path.stat().st_size}:
                issues.append("file_size_mismatch")
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    record.update(width=im.width, height=im.height, mode=im.mode)
                    # Include dimensions; identical RGB pixels form one exact group.
                    digest = hashlib.sha256(f"{im.width}x{im.height}:RGB:".encode())
                    digest.update(im.convert("RGB").tobytes())
                    record["pixel_sha256"] = digest.hexdigest()
                    if any(not valid_box(box, im.width, im.height) for box in boxes):
                        issues.append("invalid_pterygium_box")
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                issues.append("corrupt_or_unsafe_image")
        record["issues"] = issues
        records.append(record)
        if len(records) % 500 == 0:
            print(f"Checked {len(records)}/{len(grouped)} images", file=sys.stderr, flush=True)

    pixels = defaultdict(list)
    for record in records:
        if record["pixel_sha256"]:
            pixels[record["pixel_sha256"]].append(record)
    duplicates = []
    for digest, members in sorted(pixels.items()):
        if len(members) < 2:
            continue
        conflict = len({r["label"] for r in members}) > 1
        duplicates.append({"pixel_sha256": digest,
                           "filenames": [r["filename"] for r in members],
                           "label_conflict": conflict})
        for record in members:
            record["duplicate_group"] = digest
            if conflict:
                record["issues"].append("duplicate_label_conflict")

    selected = [r for r in records if r["label"] in (0, 1) and not r["issues"]]
    counts = Counter(r["label"] for r in records)
    issue_counts = Counter(issue for r in records for issue in r["issues"])
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "annotations_sha256": sha256(annotations),
        "annotation_rows": sum(len(rows) for rows in grouped.values()),
        "annotated_images": len(records), "png_files": sum(map(len, index.values())),
        "ignored_appledouble_metadata_files": metadata_files,
        "pterygium_annotation_rows": sum(len(json.loads(r["pterygium_regions"])) for r in records),
        "positive_images": counts[1], "normal_candidate_images": counts[0],
        "pterygium_only_images": sum(json.loads(r["lesions"]) == ["Pterygium"] for r in records),
        "other_disease_only_images": counts[""],
        "selected_images": len(selected),
        "selected_positive_images": sum(r["label"] == 1 for r in selected),
        "selected_normal_images": sum(r["label"] == 0 for r in selected),
        "images_with_issues": sum(bool(r["issues"]) for r in records),
        "issue_counts": dict(issue_counts),
        "issue_details": [{"filename": r["filename"], "label": r["label"], "issues": r["issues"]}
                          for r in records if r["issues"]],
        "unannotated_images": sorted(set(index) - set(grouped)),
        "exact_duplicate_groups": duplicates,
        "selected_exact_duplicate_groups": sum(
            len([r for r in selected if r["duplicate_group"] == d["pixel_sha256"]]) > 1
            for d in duplicates),
        "selected_subset_status": "passed" if selected and not any(
            r["issues"] for r in records if r["label"] in (0, 1)) else "needs_review",
        "status": "needs_review" if issue_counts or set(index) - set(grouped) else "passed",
        "normal_label_rule": "All lesion strings empty; this is a dataset-derived control label, not an explicit Normal field or independent clinical confirmation.",
        "pending": ["near-duplicate review and grouping", "cropping and image-quality review",
                    "visual lesion-overlay review", "patient identity unavailable; similarity is not patient identity",
                    "train/validation/test split", "training and smartphone validation"],
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in (("image_manifest.csv", records), ("binary_labels.csv", selected)):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for record in values:
                writer.writerow({**record, "issues": ";".join(record["issues"])})
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    report = ["# SLID dataset audit", "", f"Status: **{summary['status']}**", "",
              "| Check | Result |", "|---|---:|",
              *[f"| {key.replace('_', ' ')} | {summary[key]} |" for key in (
                  "annotation_rows", "annotated_images", "png_files", "positive_images",
                  "normal_candidate_images", "pterygium_only_images", "other_disease_only_images",
                  "selected_images", "images_with_issues")], "",
              f"Exact duplicate groups: {len(duplicates)}. Unannotated images: {len(summary['unannotated_images'])}.",
              f"Ignored macOS AppleDouble metadata files: {metadata_files} (recognized by path and file signature).",
              f"Selected subset integrity: **{summary['selected_subset_status']}**. "
              f"Exact duplicate groups in the selected subset: {summary['selected_exact_duplicate_groups']}.",
              "", "## Label rules", "",
              "An image is positive if any annotation says Pterygium, including mixed-disease images. "
              "An image is a normal candidate only if every lesion string is empty. "
              "Other-disease-only images are excluded. Images with audit issues are withheld from binary_labels.csv.",
              "", "The CSV has no explicit Normal label. These controls are inferred from the annotation convention; "
              "an empty anatomical-region annotation within a diseased image does not make that image normal.",
              "", "## Issues", "",
              *( [f"- {key}: {value}" for key, value in sorted(issue_counts.items())] or ["No automated integrity or annotation consistency issues found."]),
              *[f"- {r['filename']}: {', '.join(r['issues'])}; "
                f"{'other-disease-only (already excluded)' if r['label'] == '' else 'withheld from binary subset'}."
                for r in records if r["issues"]],
              "", "## Next approval checkpoint", "",
              "Review near-duplicates, cropping and lesion overlays before making any data split. "
              "Exact duplicates have been grouped but retained. Similarity groups cannot establish patient-independent splits. "
              "No preprocessing, split, training or smartphone performance claim is part of this audit.",
              "", "## Reproducibility", "",
              f"Annotations SHA-256: `{summary['annotations_sha256']}`.",
              "Image paths in the CSVs are relative to the images directory passed to the audit. "
              "File and decoded RGB pixel hashes are retained per image. See data/provenance.json for the source archive revision and checksum.",
              "", "Source: https://github.com/xumingyu-hub/SLID", ""]
    (output / "audit-report.md").write_text("\n".join(report), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/SLID/Annotations.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/images")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/audit")
    args = parser.parse_args()
    summary = audit(args.annotations, args.images, args.output)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"exact_duplicate_groups", "unannotated_images", "pending"}}, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
