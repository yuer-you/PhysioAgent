"""真实信号与人工心搏标注的对照评测辅助函数。

Helpers for comparing real-signal predictions with manual heartbeat annotations.
"""

from __future__ import annotations

from typing import Sequence


def match_peak_indices(
    detected_indices: Sequence[int],
    reference_indices: Sequence[int],
    tolerance_samples: int,
) -> dict[str, float | int]:
    """在给定采样点容差内，对检测峰与参考心搏做一对一匹配。

    输入必须是采样点索引。排序后的双指针匹配确保一个检测峰不能同时命中两个
    参考心搏，这比简单判断“附近是否存在标注”更严格。

    Match detected peaks one-to-one with reference heartbeats within a sample tolerance. Inputs are sample
    indices. The sorted two-pointer match prevents one detected peak from matching two reference beats,
    which is stricter than simply asking whether an annotation exists nearby.
    """
    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be non-negative.")

    detected = sorted(int(index) for index in detected_indices)
    reference = sorted(int(index) for index in reference_indices)
    if any(index < 0 for index in detected + reference):
        raise ValueError("Peak indices must be non-negative.")

    detected_cursor = 0
    reference_cursor = 0
    true_positives = 0
    while detected_cursor < len(detected) and reference_cursor < len(reference):
        detected_index = detected[detected_cursor]
        reference_index = reference[reference_cursor]
        if abs(detected_index - reference_index) <= tolerance_samples:
            true_positives += 1
            detected_cursor += 1
            reference_cursor += 1
        elif detected_index < reference_index - tolerance_samples:
            detected_cursor += 1
        else:
            reference_cursor += 1

    false_positives = len(detected) - true_positives
    false_negatives = len(reference) - true_positives
    sensitivity = _safe_ratio(true_positives, true_positives + false_negatives)
    positive_predictive_value = _safe_ratio(true_positives, true_positives + false_positives)
    f1 = _safe_ratio(2 * true_positives, 2 * true_positives + false_positives + false_negatives)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "sensitivity": sensitivity,
        "positive_predictive_value": positive_predictive_value,
        "f1": f1,
    }


def mean_heart_rate_from_annotations(reference_indices: Sequence[int], sampling_rate: float) -> float:
    """用相邻人工心搏标注的平均间隔计算参考平均心率。

    Compute reference mean heart rate from the mean interval between adjacent manual annotations.
    """
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    indices = sorted(int(index) for index in reference_indices)
    if len(indices) < 2:
        raise ValueError("At least two reference beats are required.")
    intervals = [right - left for left, right in zip(indices, indices[1:])]
    if any(interval <= 0 for interval in intervals):
        raise ValueError("Reference beat indices must be unique.")
    mean_interval_seconds = (sum(intervals) / len(intervals)) / sampling_rate
    return 60.0 / mean_interval_seconds


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
