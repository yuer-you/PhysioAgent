import json
from pathlib import Path

import pytest

from physioagent.evaluate_real_signal import evaluate_real_signal
from physioagent.real_signal import match_peak_indices, mean_heart_rate_from_annotations


DATA = Path(__file__).parents[1] / "data" / "sample_ecg.csv"


def test_match_peak_indices_is_one_to_one():
    result = match_peak_indices(
        detected_indices=[98, 101, 205, 400],
        reference_indices=[100, 200, 300],
        tolerance_samples=5,
    )
    assert result["true_positives"] == 2
    assert result["false_positives"] == 2
    assert result["false_negatives"] == 1
    assert result["sensitivity"] == pytest.approx(2 / 3)
    assert result["positive_predictive_value"] == pytest.approx(1 / 2)
    assert result["f1"] == pytest.approx(4 / 7)


def test_match_peak_indices_handles_empty_predictions():
    result = match_peak_indices([], [10, 20], tolerance_samples=2)
    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 2
    assert result["f1"] == 0.0


def test_reference_heart_rate_uses_mean_rr_interval():
    # 25 Hz 下相邻心搏相隔 20 点，即 0.8 秒，对应 75 BPM。
    # At 25 Hz, 20 samples between beats equal 0.8 seconds, corresponding to 75 BPM.
    assert mean_heart_rate_from_annotations([20, 40, 60], sampling_rate=25) == pytest.approx(75.0)


def test_reference_heart_rate_rejects_insufficient_beats():
    with pytest.raises(ValueError, match="At least two"):
        mean_heart_rate_from_annotations([20], sampling_rate=25)


def test_real_signal_evaluation_connects_tools_to_reference(tmp_path):
    reference = {
        "dataset": "synthetic unit-test signal",
        "record": "toy",
        "sampling_rate_hz": 25,
        "num_samples": 81,
        "beat_indices": [20, 40, 60],
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")

    result = evaluate_real_signal(DATA, reference_path, tolerance_seconds=0.04)
    assert result["peak_matching"]["f1"] == pytest.approx(1.0)
    assert result["reference"]["mean_heart_rate_bpm"] == pytest.approx(75.0)
    assert result["detected"]["mean_heart_rate_bpm"] == pytest.approx(75.0)
    assert result["detected"]["heart_rate_error_bpm"] == pytest.approx(0.0)
