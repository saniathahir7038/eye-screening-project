import numpy as np
import pytest

from export_tflite import score_output_detail
from export_mobile_heatmap import mobile_gradcam


class FakeInterpreter:
    def __init__(self, outputs):
        self.outputs = outputs

    def get_output_details(self):
        return self.outputs


def test_score_output_found_when_feature_map_is_first():
    interpreter = FakeInterpreter([
        {"shape": [1, 7, 7, 1280], "dtype": np.float32, "index": 0},
        {"shape": [1, 1], "dtype": np.float32, "index": 1},
    ])
    assert score_output_detail(interpreter)["index"] == 1


def test_score_output_rejects_missing_or_duplicate_score():
    score = {"shape": [1, 1], "dtype": np.float32, "index": 1}
    with pytest.raises(ValueError):
        score_output_detail(FakeInterpreter([]))
    with pytest.raises(ValueError):
        score_output_detail(FakeInterpreter([score, score]))


def test_mobile_gradcam_weights_positive_channel():
    convolution = np.zeros((7, 7, 2), dtype=np.float32)
    convolution[0, 0, 0] = 2
    convolution[0, 1, 0] = 1
    params = (np.array([1, -1], dtype=np.float32),
              np.ones(2, dtype=np.float32), np.zeros(2, dtype=np.float32),
              np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), 0)
    heatmap = mobile_gradcam(convolution, params)
    assert heatmap[0, 0] == pytest.approx(1)
    assert heatmap[0, 1] == pytest.approx(0.5)
    assert heatmap[0, 2] == pytest.approx(0)
