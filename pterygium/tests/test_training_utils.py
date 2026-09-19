import unittest

import numpy as np

from training_utils import balanced_class_weights, confusion_values, select_threshold


class TrainingUtilityTests(unittest.TestCase):
    def test_balanced_class_weights_upweight_the_smaller_class(self):
        weights = balanced_class_weights([0, 0, 0, 1])
        self.assertAlmostEqual(weights[0], 4 / 6)
        self.assertEqual(weights[1], 2.0)

    def test_threshold_is_selected_from_validation_scores(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.10, 0.40, 0.35, 0.80])
        selected, rows = select_threshold(labels, scores)
        self.assertAlmostEqual(selected["threshold"], 0.225)
        self.assertEqual(selected["sensitivity"], 1.0)
        self.assertEqual(selected["specificity"], 0.5)
        self.assertGreater(len(rows), 3)

    def test_confusion_values_use_greater_than_or_equal_threshold(self):
        result = confusion_values([0, 1], [0.49, 0.50], 0.50)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["true_positive"], 1)

    def test_threshold_selection_rejects_one_class_input(self):
        with self.assertRaises(ValueError):
            select_threshold([1, 1], [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
