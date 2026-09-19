import unittest

import numpy as np

from evaluation_utils import classification_metrics, spatial_attention_metrics, wilson_interval


class EvaluationUtilityTests(unittest.TestCase):
    def test_classification_metrics_match_confusion_counts(self):
        metrics, predictions = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], 0.5)
        self.assertEqual(predictions.tolist(), [0, 1, 1, 1])
        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["specificity"], 0.5)

    def test_wilson_interval_contains_observed_proportion(self):
        lower, upper = wilson_interval(8, 10)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 0.8)

    def test_attention_metrics_detect_box_and_corner_focus(self):
        heatmap = np.zeros((10, 10), dtype=float)
        heatmap[:2, :2] = 1
        box = [{"x": 0, "y": 0, "width": 2, "height": 2}]
        metrics = spatial_attention_metrics(heatmap, box, corner_fraction=0.2, border_fraction=0.1)
        self.assertEqual(metrics["corner_attention_fraction"], 1.0)
        self.assertEqual(metrics["box_attention_fraction"], 1.0)
        self.assertTrue(metrics["maximum_point_inside_box"])
        self.assertGreater(metrics["box_attention_lift"], 1.0)

    def test_attention_metrics_support_images_without_boxes(self):
        metrics = spatial_attention_metrics(np.ones((10, 10)), [])
        self.assertIsNone(metrics["box_attention_fraction"])
        self.assertAlmostEqual(metrics["corner_attention_fraction"], 0.04)


if __name__ == "__main__":
    unittest.main()
