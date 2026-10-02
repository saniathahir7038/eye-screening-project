from io import BytesIO
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from framing import FramingResult, auto_frame
from prepare_dataset import preprocess_image
from screening import (INCONCLUSIVE, ImageInputError, NO_VISIBLE_PATTERN, QualityConfig, ScreeningRuntime,
                       SUSPECTED_PTERYGIUM, assess_image_quality, create_heatmap_overlay,
                       decode_uploaded_image, load_runtime, model_threshold, run_screening)


def encoded_image(image, image_format="PNG"):
    stream = BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


class ScreeningTests(unittest.TestCase):
    def test_decode_upload_orients_and_converts_to_rgb(self):
        image = Image.new("L", (320, 240), 90)
        decoded = decode_uploaded_image(encoded_image(image))
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (320, 240))

    def test_decode_rejects_empty_and_non_image_uploads(self):
        for data in (b"", b"not an image"):
            with self.subTest(data=data):
                with self.assertRaises(ImageInputError):
                    decode_uploaded_image(data)

    def test_quality_gate_accepts_clear_training_like_image(self):
        yy, xx = np.indices((300, 400))
        pattern = np.where(((xx // 8) + (yy // 8)) % 2, 155, 75).astype(np.uint8)
        array = np.repeat(pattern[..., None], 3, axis=2)
        result = assess_image_quality(Image.fromarray(array))
        self.assertTrue(result.accepted)
        self.assertFalse(result.issues)

    def test_quality_gate_reports_multiple_failures(self):
        result = assess_image_quality(Image.new("RGB", (100, 300), (4, 4, 4)))
        self.assertFalse(result.accepted)
        self.assertGreaterEqual(len(result.issues), 4)

    def test_quality_thresholds_can_be_configured(self):
        image = Image.new("RGB", (224, 224), (100, 100, 100))
        result = assess_image_quality(image, QualityConfig(min_sharpness=0, min_contrast=0))
        self.assertTrue(result.accepted)

    def test_auto_framing_trims_a_synthetic_skin_margin(self):
        yy, xx = np.indices((224, 224))
        eye = np.where(((xx // 8) + (yy // 8)) % 2, 155, 75).astype(np.uint8)
        canvas = np.zeros((300, 224, 3), dtype=np.uint8)
        canvas[:] = (180, 120, 80)
        canvas[:224] = np.repeat(eye[..., None], 3, axis=2)
        result = auto_frame(Image.fromarray(canvas))
        self.assertTrue(result.adjusted)
        self.assertEqual(result.image.width, 224)
        self.assertLess(result.image.height, 300)
        self.assertGreaterEqual(result.image.height, 225)

    def test_heatmap_overlay_validates_shape(self):
        image = Image.new("RGB", (224, 224), (100, 100, 100))
        overlay = create_heatmap_overlay(image, np.ones((224, 224), dtype=np.float32))
        self.assertEqual(overlay.size, image.size)
        with self.assertRaises(ValueError):
            create_heatmap_overlay(image, np.ones((10, 10), dtype=np.float32))

    def test_runtime_rejects_unaccepted_summary_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.keras"
            model.write_bytes(b"placeholder")
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "status": "needs_review", "test_set_inference_performed": False,
                "corrected": {"threshold": 0.2},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "acceptance gate"):
                load_runtime(model, summary)

    def test_experimental_manifest_requires_matching_model_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.keras"
            model.write_bytes(b"candidate")
            summary = root / "manifest.json"
            summary.write_text(json.dumps({
                "status": "experimental_research_only",
                "threshold": 0.16848066449165344,
                "model_sha256": hashlib.sha256(b"candidate").hexdigest(),
            }), encoding="utf-8")
            self.assertAlmostEqual(model_threshold(model, summary), 0.16848066449165344)
            model.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "checksum"):
                model_threshold(model, summary)

    def test_run_screening_uses_exact_preprocessing_and_threshold(self):
        class FakeGradcam:
            pass

        image = Image.new("RGB", (400, 300), (80, 110, 140))
        expected, _ = preprocess_image(image, 224, 0.20)
        runtime = ScreeningRuntime(object(), FakeGradcam(), 0.4, "fake.keras")

        import evaluate
        original = evaluate.gradcam
        try:
            evaluate.gradcam = lambda model, array: (np.zeros((224, 224), np.float32), 0.5)
            result = run_screening(runtime, image)
        finally:
            evaluate.gradcam = original

        self.assertEqual(result.label, SUSPECTED_PTERYGIUM)
        self.assertTrue(result.suspected)
        self.assertEqual(result.model_score, 0.5)
        self.assertTrue(np.array_equal(np.asarray(result.processed_image), np.asarray(expected)))
        self.assertNotEqual(result.label, NO_VISIBLE_PATTERN)

    def test_run_screening_abstains_when_framing_changes_decision(self):
        image = Image.new("RGB", (400, 300), (80, 110, 140))
        framing = FramingResult(image.crop((0, 0, 400, 250)), 1 / 6)
        runtime = ScreeningRuntime(object(), object(), 0.4, "fake.keras")
        scores = iter((0.1, 0.9))  # framed input, then full photo

        import evaluate
        original = evaluate.gradcam
        try:
            evaluate.gradcam = lambda model, array: (np.zeros((224, 224), np.float32), next(scores))
            result = run_screening(runtime, image, framing)
        finally:
            evaluate.gradcam = original

        self.assertTrue(result.inconclusive)
        self.assertEqual(result.label, INCONCLUSIVE)
        self.assertEqual(result.model_score, 0.1)
        self.assertEqual(result.full_photo_score, 0.9)


if __name__ == "__main__":
    unittest.main()
