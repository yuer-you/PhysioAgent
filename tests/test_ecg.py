import numpy as np
import pytest

from physioagent.ecg import calculate_ecg_heart_rate, detect_ecg_r_peaks


def make_synthetic_ecg(sampling_rate: int = 250) -> tuple[np.ndarray, list[int]]:
    samples = np.arange(5 * sampling_rate)
    qrs_locations = [250, 500, 750, 1000]
    signal = 0.02 * np.sin(2 * np.pi * samples / (2 * sampling_rate))
    for location in qrs_locations:
        # 窄而高的 QRS，交替使用正/负极性；后面跟随较宽的 T 波。
        polarity = 1 if location % 500 else -1
        signal += polarity * np.exp(-0.5 * ((samples - location) / 3.0) ** 2)
        signal += 0.30 * np.exp(-0.5 * ((samples - (location + 70)) / 14.0) ** 2)
    return signal, qrs_locations


def test_ecg_detector_rejects_t_waves_and_handles_inverted_qrs():
    signal, expected = make_synthetic_ecg()
    result = detect_ecg_r_peaks(signal, sampling_rate=250)
    assert result["num_peaks"] == len(expected)
    assert np.allclose(result["peak_indices"], expected, atol=3)
    assert result["detector"] == "ecg_detector_v1"


def test_ecg_heart_rate_uses_detected_r_peaks():
    signal, _ = make_synthetic_ecg()
    result = calculate_ecg_heart_rate(signal, sampling_rate=250)
    assert result["mean_heart_rate_bpm"] == pytest.approx(60.0, abs=0.5)


def test_ecg_detector_rejects_sampling_rate_below_filter_requirement():
    with pytest.raises(ValueError, match="twice the ECG highcut"):
        detect_ecg_r_peaks(np.ones(100), sampling_rate=25)
