"""信号处理工具。

工具函数保持为普通、无状态的 Python 函数：这能让 Agent、命令行和单元测试
使用同一份实现。真实项目中也应把模型的输出与实际信号处理代码分开。

Signal-processing tools. Each tool remains an ordinary stateless Python function so the agent, CLI,
and unit tests share one implementation. Production systems should likewise separate model output from
the code that performs actual signal processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def load_signal(file_path: str | Path, signal_column: str = "signal") -> np.ndarray:
    """从含表头的 CSV 读取一个数值信号列。

    Read one numeric signal column from a CSV file with a header.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Signal file does not exist: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.dtype.names is None or signal_column not in data.dtype.names:
        raise ValueError(f"CSV must contain a '{signal_column}' column.")
    signal = np.atleast_1d(np.asarray(data[signal_column], dtype=float))
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("Signal must contain at least one finite number.")
    return signal


def calculate_statistics(signal: np.ndarray, sampling_rate: float) -> dict[str, float | int]:
    """返回基础描述统计量；sampling_rate 单位为 Hz。

    Return basic descriptive statistics; sampling_rate is measured in Hz.
    """
    values = _validate_signal(signal)
    _validate_sampling_rate(sampling_rate)
    return {
        "num_samples": int(values.size),
        "duration_seconds": float(values.size / sampling_rate),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def detect_peaks(
    signal: np.ndarray, sampling_rate: float, min_distance_seconds: float = 0.3, prominence: float | None = None
) -> dict[str, object]:
    """检测局部峰，返回峰的位置（采样点）和数量。

    Detect local peaks and return their sample indices and count.
    """
    values = _validate_signal(signal)
    _validate_sampling_rate(sampling_rate)
    if min_distance_seconds <= 0:
        raise ValueError("min_distance_seconds must be positive.")

    distance = max(1, round(min_distance_seconds * sampling_rate))
    peaks, _ = find_peaks(values, distance=distance, prominence=prominence)
    return {"peak_indices": peaks.tolist(), "num_peaks": int(peaks.size)}


def calculate_heart_rate(
    signal: np.ndarray, sampling_rate: float, min_distance_seconds: float = 0.3, prominence: float | None = None
) -> dict[str, object]:
    """基于峰间期估计平均心率（BPM）。至少需要两个峰。

    Estimate mean heart rate in BPM from inter-peak intervals; at least two peaks are required.
    """
    peak_result = detect_peaks(signal, sampling_rate, min_distance_seconds, prominence)
    peaks = np.asarray(peak_result["peak_indices"], dtype=int)
    if peaks.size < 2:
        raise ValueError("At least two peaks are required to estimate heart rate.")

    intervals_seconds = np.diff(peaks) / sampling_rate
    return {
        **peak_result,
        "mean_rr_interval_seconds": float(np.mean(intervals_seconds)),
        "mean_heart_rate_bpm": float(60.0 / np.mean(intervals_seconds)),
    }


def filter_signal(
    signal: np.ndarray, sampling_rate: float, lowcut: float = 0.5, highcut: float = 8.0, order: int = 4
) -> np.ndarray:
    """使用零相位 Butterworth 带通滤波，保留 lowcut 到 highcut Hz 的成分。

    Apply zero-phase Butterworth band-pass filtering and retain frequencies from lowcut to highcut Hz.
    """
    values = _validate_signal(signal)
    _validate_sampling_rate(sampling_rate)
    nyquist = sampling_rate / 2
    if not 0 < lowcut < highcut < nyquist:
        raise ValueError("Expected 0 < lowcut < highcut < sampling_rate / 2.")
    if order < 1:
        raise ValueError("order must be at least 1.")
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=sampling_rate, output="sos")
    if values.size <= 3 * (2 * order + 1):
        raise ValueError("Signal is too short for this filter order.")
    return sosfiltfilt(sos, values)


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a non-empty, finite, one-dimensional array.")
    return values


def _validate_sampling_rate(sampling_rate: float) -> None:
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
