import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from audit import audit, read_annotations, REQUIRED
from download import safe_extract, extraction_complete
import zipfile


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.rows = []

    def add_image(self, name, lesions, color="white"):
        path = self.images / name
        Image.new("RGB", (20, 20), color).save(path)
        for number, lesion in enumerate(lesions):
            self.rows.append(dict(filename=name, file_size=path.stat().st_size,
                                 annotation_count=len(lesions), annotation_ID=number,
                                 attributes=json.dumps({"region": "Cornea", "lesion": lesion}),
                                 shape_coordinates=json.dumps({"name": "rect", "x": 1,
                                                               "y": 1, "width": 4, "height": 4})))

    def csv_path(self):
        path = self.root / "Annotations.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(REQUIRED))
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def run_audit(self):
        return audit(self.csv_path(), self.images, self.root / "output")

    def test_labels_are_per_image_and_mixed_positives_are_retained(self):
        self.add_image("positive.png", ["", "Pterygium", "Pterygium", "Cataract"], "red")
        self.add_image("normal.png", ["", ""], "green")
        self.add_image("other.png", ["", "Pinguecula"], "blue")
        result = self.run_audit()
        self.assertEqual(result["positive_images"], 1)
        self.assertEqual(result["pterygium_annotation_rows"], 2)
        self.assertEqual(result["normal_candidate_images"], 1)
        self.assertEqual(result["other_disease_only_images"], 1)
        self.assertEqual(result["selected_images"], 2)
        self.assertEqual(result["status"], "passed")

    def test_missing_corrupt_and_ambiguous_files_are_withheld(self):
        self.add_image("missing.png", [""])
        self.add_image("corrupt.png", [""])
        self.add_image("ambiguous.png", [""])
        (self.images / "missing.png").unlink()
        (self.images / "corrupt.png").write_bytes(b"not a PNG")
        nested = self.images / "nested"
        nested.mkdir()
        (nested / "ambiguous.png").write_bytes((self.images / "ambiguous.png").read_bytes())
        result = self.run_audit()
        self.assertEqual(result["selected_images"], 0)
        for issue in ("missing_image", "corrupt_or_unsafe_image", "ambiguous_image_filename"):
            self.assertEqual(result["issue_counts"][issue], 1)

    def test_malformed_attributes_never_become_normal(self):
        self.add_image("unknown.png", [""])
        self.rows[0]["attributes"] = '{"region":"Cornea"}'
        with self.assertRaisesRegex(ValueError, "Missing lesion"):
            self.run_audit()

    def test_incomplete_annotations_are_withheld(self):
        self.add_image("incomplete.png", [""])
        self.rows[0]["annotation_count"] = 2
        result = self.run_audit()
        self.assertEqual(result["selected_images"], 0)
        self.assertEqual(result["issue_counts"]["annotation_count_mismatch"], 1)

    def test_invalid_lesion_box_is_withheld(self):
        self.add_image("positive.png", ["Pterygium"])
        self.rows[0]["shape_coordinates"] = json.dumps({"name": "rect", "x": 19,
                                                        "y": 0, "width": 5, "height": 2})
        result = self.run_audit()
        self.assertEqual(result["selected_images"], 0)
        self.assertEqual(result["issue_counts"]["invalid_pterygium_box"], 1)

    def test_exact_duplicates_with_conflicting_labels_are_withheld(self):
        self.add_image("positive.png", ["Pterygium"])
        self.add_image("normal.png", [""])
        result = self.run_audit()
        self.assertEqual(result["selected_images"], 0)
        self.assertEqual(result["issue_counts"]["duplicate_label_conflict"], 2)

    def test_matching_duplicate_labels_are_grouped_and_retained(self):
        self.add_image("one.png", [""])
        self.add_image("two.png", [""])
        result = self.run_audit()
        self.assertEqual(result["selected_images"], 2)
        self.assertEqual(len(result["exact_duplicate_groups"]), 1)

    def test_appledouble_metadata_is_ignored_but_real_unannotated_images_are_reported(self):
        self.add_image("one.png", [""])
        metadata = self.images / "__MACOSX"
        metadata.mkdir()
        (metadata / "._one.png").write_bytes(b"\x00\x05\x16\x07metadata")
        Image.new("RGB", (20, 20), "blue").save(metadata / "._real.png")
        result = self.run_audit()
        self.assertEqual(result["ignored_appledouble_metadata_files"], 1)
        self.assertEqual(result["png_files"], 2)
        self.assertEqual(result["unannotated_images"], ["._real.png"])

    def test_csv_path_traversal_is_rejected(self):
        self.add_image("normal.png", [""])
        self.rows[0]["filename"] = "../normal.png"
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            read_annotations(self.csv_path())

    def test_zip_path_traversal_is_rejected_before_extraction(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("fine.png", b"test")
            stream.writestr("../escaped.txt", b"test")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            safe_extract(archive, self.images)
        self.assertFalse((self.images / "fine.png").exists())
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_extraction_check_detects_missing_and_changed_bytes(self):
        archive = self.root / "good.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("sample.png", b"original")
        self.assertFalse(extraction_complete(archive, self.images))
        safe_extract(archive, self.images)
        self.assertTrue(extraction_complete(archive, self.images))
        (self.images / "sample.png").write_bytes(b"modified")
        self.assertFalse(extraction_complete(archive, self.images))


if __name__ == "__main__":
    unittest.main()
