"""轻量 ECG QRS/R 峰检测。

流程借鉴 Pan–Tompkins 的核心思想，但不是论文算法的逐行复现：带通滤波突出
QRS 频段，差分平方强调快速变化，移动积分形成检测包络，再用稳健阈值选峰。
最后回到滤波波形，用绝对幅度定位 R 峰，因此也能处理倒置 QRS。

Lightweight ECG QRS/R-peak detection. The pipeline follows the core Pan–Tompkins ideas rather than
reproducing the paper line by line: band-pass filtering emphasizes QRS frequencies, squared differences
highlight fast changes, moving integration builds an envelope, and a robust threshold selects peaks.
R peaks are then refined on the filtered waveform by absolute amplitude, which also handles inverted QRS.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


@dataclass(frozen=True)
class ECGDetectorConfig:
    lowcut_hz: float = 5.0
    highcut_hz: float = 15.0
    filter_order: int = 2
    integration_window_seconds: float = 0.12
    min_distance_seconds: float = 0.25
    threshold_mad_multiplier: float = 5.0
    refinement_radius_seconds: float = 0.12


ECG_DETECTOR_V1_CONFIG = ECGDetectorConfig()


def detect_ecg_r_peaks(
    signal: np.ndarray,
    sampling_rate: float,
    config: ECGDetectorConfig = ECG_DETECTOR_V1_CONFIG,
) -> dict[str, object]:
    """检测 ECG R 峰，返回采样点索引、数量和可复现的配置。

    Detect ECG R peaks and return sample indices, count, and a reproducible configuration.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a non-empty, finite, one-dimensional array.")
    if sampling_rate <= 2 * config.highcut_hz:
        raise ValueError("sampling_rate must be greater than twice the ECG highcut frequency.")
    if values.size <= round(0.5 * sampling_rate):
        raise ValueError("ECG signal must be longer than 0.5 seconds.")

    sos = butter(
        config.filter_order,
        [config.lowcut_hz, config.highcut_hz],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    filtered = sosfiltfilt(sos, values)
    derivative = np.gradient(filtered)
    energy = derivative * derivative
    integration_width = max(1, round(config.integration_window_seconds * sampling_rate))
    envelope = np.convolve(energy, np.ones(integration_width) / integration_width, mode="same")

    median = float(np.median(envelope))
    median_absolute_deviation = float(np.median(np.abs(envelope - median)))
    threshold = median + config.threshold_mad_multiplier * median_absolute_deviation
    candidates, _ = find_peaks(
        envelope,
        height=threshold,
        distance=max(1, round(config.min_distance_seconds * sampling_rate)),
    )

    refinement_radius = max(1, round(config.refinement_radius_seconds * sampling_rate))
    refined: list[int] = []
    for candidate in candidates:
        left = max(0, int(candidate) - refinement_radius)
        right = min(values.size, int(candidate) + refinement_radius + 1)
        # 使用绝对值兼容向上和向下的 QRS 主波。
        # Use absolute amplitude to support both upright and inverted dominant QRS waves.
        refined.append(left + int(np.argmax(np.abs(filtered[left:right]))))
    peak_indices = sorted(set(refined))
    return {
        "peak_indices": peak_indices,
        "num_peaks": len(peak_indices),
        "detector": "ecg_detector_v1",
        "threshold": threshold,
        "config": asdict(config),
    }


def calculate_ecg_heart_rate(
    signal: np.ndarray,
    sampling_rate: float,
    config: ECGDetectorConfig = ECG_DETECTOR_V1_CONFIG,
) -> dict[str, object]:
    """使用 ECG R 峰间隔计算平均心率。

    Compute mean heart rate from intervals between ECG R peaks.
    """
    peak_result = detect_ecg_r_peaks(signal, sampling_rate, config)
    peaks = np.asarray(peak_result["peak_indices"], dtype=int)
    if peaks.size < 2:
        raise ValueError("At least two ECG R peaks are required to estimate heart rate.")
    intervals_seconds = np.diff(peaks) / sampling_rate
    return {
        **peak_result,
        "mean_rr_interval_seconds": float(np.mean(intervals_seconds)),
        "mean_heart_rate_bpm": float(60.0 / np.mean(intervals_seconds)),
    }
