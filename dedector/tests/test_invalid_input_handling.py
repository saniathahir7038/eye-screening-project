import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from mire_segmentation.get_center import segment_and_get_center
from video_processor import DEDector


class InvalidInputHandlingTests(unittest.TestCase):
    def test_get_center_rejects_background_only_mask(self):
        segmenter = object.__new__(segment_and_get_center)

        with self.assertRaisesRegex(ValueError, "did not identify any mire foreground"):
            segmenter.get_center(np.zeros((16, 16), dtype=np.uint8))

    def test_get_center_uses_foreground_centroid(self):
        segmenter = object.__new__(segment_and_get_center)
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1, 2] = 1
        mask[3, 4] = 1

        x, y = segmenter.get_center(mask)

        self.assertEqual(x, 3)
        self.assertEqual(y, 2)

    def test_sharpness_rejects_empty_processed_frame_set(self):
        dedector = object.__new__(DEDector)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No valid mire frames were produced"):
                dedector.calculate_video_sharpness(0, 5, directory)


if __name__ == "__main__":
    unittest.main()
