import numpy as np
import tensorflow as tf

from corrective_train import acceptance_checks, apply_corner_patch, expand_corner_variants


def test_expand_corner_variants_keeps_all_four_training_views():
    rows = [
        {"absolute_image_path": "a.png", "label": "0"},
        {"absolute_image_path": "b.png", "label": "1"},
    ]
    paths, labels, variants = expand_corner_variants(rows)
    assert paths == ["a.png"] * 4 + ["b.png"] * 4
    assert labels == [0] * 4 + [1] * 4
    assert variants == [0, 1, 2, 3] * 2


def test_corner_patch_changes_only_selected_corner():
    image = tf.zeros((224, 224, 3), dtype=tf.float32)
    unchanged, _ = apply_corner_patch(image, tf.constant(1.0), tf.constant(0), 45)
    patched, label = apply_corner_patch(image, tf.constant(1.0), tf.constant(1), 45)
    patched = patched.numpy()
    assert np.count_nonzero(unchanged.numpy()) == 0
    assert np.all(patched[:45, -45:] == 127)
    assert np.count_nonzero(patched) == 45 * 45 * 3
    assert label.numpy() == 1.0


def test_acceptance_checks_require_localization_and_corner_improvement():
    baseline = {
        "roc_auc": 0.99,
        "metrics": {"sensitivity": 1.0, "specificity": 0.97},
        "attention": {
            "positive_mean_box_attention_lift": 2.1,
            "positive_maximum_point_inside_box_count": 18,
            "normal_mean_corner_attention_fraction": 0.03,
        },
    }
    corrected = {
        "roc_auc": 0.98,
        "metrics": {"sensitivity": 0.96, "specificity": 1.0},
        "attention": {
            "positive_mean_box_attention_lift": 2.2,
            "positive_maximum_point_inside_box_count": 18,
            "normal_mean_corner_attention_fraction": 0.02,
        },
    }
    assert all(acceptance_checks(baseline, corrected).values())
