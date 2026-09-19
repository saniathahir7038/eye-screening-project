import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import imagehash
import numpy as np
from PIL import Image, ImageDraw

from prepare_dataset import (assign_splits, detect_top_left_black_box, duplicate_candidates,
                             map_boxes, preprocess_image)


class PreprocessingTests(unittest.TestCase):
    def test_corner_is_standardized_without_cropping_the_eye(self):
        image = Image.new("RGB", (100, 80), "white")
        ImageDraw.Draw(image).rectangle((0, 0, 19, 14), fill="black")
        self.assertEqual(detect_top_left_black_box(image), (20, 15))
        processed, crop = preprocess_image(image, 32)
        self.assertEqual(processed.size, (32, 32))
        self.assertEqual(crop["crop_top"], 0)
        self.assertEqual(crop["crop_bottom"], 80)
        self.assertEqual(crop["corner_mask_width"], 20)
        self.assertEqual(crop["corner_mask_height"], 16)
        self.assertEqual(processed.getpixel((0, 0)), (127, 127, 127))

    def test_ordinary_dark_pixel_is_not_treated_as_information_box(self):
        image = Image.new("RGB", (100, 80), "white")
        image.putpixel((0, 0), (0, 0, 0))
        self.assertEqual(detect_top_left_black_box(image), (0, 0))

    def test_boxes_are_mapped_after_top_crop(self):
        crop = {"crop_left": 0, "crop_top": 20, "crop_right": 100, "crop_bottom": 100}
        boxes, clipped = map_boxes([{"x": 10, "y": 30, "width": 20, "height": 40}], crop, 200)
        self.assertEqual(clipped, 0)
        self.assertEqual(boxes, [{"x": 20.0, "y": 25.0, "width": 40.0, "height": 100.0}])


class SplitTests(unittest.TestCase):
    def make_records(self):
        records = []
        for label, count in ((1, 163), (0, 245)):
            for number in range(count):
                filename = f"{label}_{number}.png"
                records.append({"filename": filename, "label": str(label),
                                "similarity_group": f"g_{label}_{number}"})
        # One normal exact-duplicate group must remain atomic.
        records[-1]["similarity_group"] = records[-2]["similarity_group"]
        return records

    def test_split_is_reproducible_stratified_and_group_safe(self):
        first = self.make_records()
        second = self.make_records()
        assign_splits(first, 123)
        assign_splits(second, 123)
        self.assertEqual([(r["filename"], r["split"]) for r in first],
                         [(r["filename"], r["split"]) for r in second])
        counts = {(split, label): sum(r["split"] == split and r["label"] == str(label) for r in first)
                  for split in ("train", "validation", "test") for label in (0, 1)}
        self.assertEqual(counts[("train", 1)], 114)
        self.assertEqual(counts[("validation", 1)], 24)
        self.assertEqual(counts[("test", 1)], 25)
        self.assertEqual(counts[("train", 0)], 172)
        self.assertEqual(counts[("validation", 0)], 37)
        self.assertEqual(counts[("test", 0)], 36)
        grouped = {}
        for row in first:
            grouped.setdefault(row["similarity_group"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in grouped.values()))

    def test_exact_duplicate_pair_is_grouped(self):
        image = Image.new("RGB", (32, 32), "red")
        phash = imagehash.phash(image)
        dhash = imagehash.dhash(image)
        records = [
            {"filename": "a.png", "label": "0", "pixel_sha256": "same", "phash_obj": phash,
             "dhash_obj": dhash, "similarity_array": np.zeros((64, 64), dtype=np.float32)},
            {"filename": "b.png", "label": "0", "pixel_sha256": "same", "phash_obj": phash,
             "dhash_obj": dhash, "similarity_array": np.zeros((64, 64), dtype=np.float32)},
        ]
        candidates = duplicate_candidates(records, {"confirmed_pairs": [], "rejected_pairs": []})
        self.assertEqual(candidates[0]["review_status"], "exact_grouped")
        self.assertEqual(records[0]["similarity_group"], records[1]["similarity_group"])

    def test_hash_similarity_does_not_group_without_confirmation(self):
        image = Image.new("RGB", (32, 32), "red")
        phash = imagehash.phash(image)
        dhash = imagehash.dhash(image)
        preview = np.arange(4096, dtype=np.float32).reshape(64, 64)
        records = [
            {"filename": "1.png", "label": "0", "pixel_sha256": "first", "phash_obj": phash,
             "dhash_obj": dhash, "similarity_array": preview.copy()},
            {"filename": "2.png", "label": "0", "pixel_sha256": "second", "phash_obj": phash,
             "dhash_obj": dhash, "similarity_array": preview.copy()},
        ]
        candidates = duplicate_candidates(records, {"confirmed_pairs": [], "rejected_pairs": []})
        self.assertEqual(candidates[0]["review_status"], "review_required")
        self.assertNotEqual(records[0]["similarity_group"], records[1]["similarity_group"])


if __name__ == "__main__":
    unittest.main()
