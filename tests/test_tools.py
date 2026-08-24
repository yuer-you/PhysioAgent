from pathlib import Path

import numpy as np
import pytest

from physioagent.tools import calculate_heart_rate, calculate_statistics, detect_peaks, filter_signal, load_signal


DATA = Path(__file__).parents[1] / "data" / "sample_ecg.csv"


def test_load_signal_reads_sample_csv():
    signal = load_signal(DATA)
    assert signal.ndim == 1
    assert len(signal) == 81


def test_calculate_statistics_returns_expected_fields():
    stats = calculate_statistics(np.array([1.0, 2.0, 3.0]), sampling_rate=2)
    assert stats["num_samples"] == 3
    assert stats["duration_seconds"] == 1.5
    assert stats["mean"] == 2.0


def test_peak_detection_and_heart_rate_on_toy_ecg():
    signal = load_signal(DATA)
    peaks = detect_peaks(signal, sampling_rate=25)
    rate = calculate_heart_rate(signal, sampling_rate=25)
    assert peaks["peak_indices"] == [20, 40, 60]
    assert rate["mean_heart_rate_bpm"] == pytest.approx(75.0)


def test_filter_signal_returns_same_length():
    time = np.arange(250) / 25
    signal = np.sin(2 * np.pi * 2 * time) + 0.2 * np.sin(2 * np.pi * 10 * time)
    filtered = filter_signal(signal, sampling_rate=25, lowcut=0.5, highcut=8.0)
    assert filtered.shape == signal.shape
