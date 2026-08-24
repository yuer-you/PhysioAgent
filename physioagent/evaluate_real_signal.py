"""用 MIT-BIH 专家心搏标注检查当前峰检测和心率工具。

Check the current peak detector and heart-rate tool against expert MIT-BIH heartbeat annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ecg import calculate_ecg_heart_rate, detect_ecg_r_peaks
from .real_signal import match_peak_indices, mean_heart_rate_from_annotations
from .tools import calculate_heart_rate, calculate_statistics, detect_peaks, load_signal


def evaluate_real_signal(
    signal_file: str | Path,
    reference_file: str | Path,
    *,
    min_distance_seconds: float = 0.3,
    prominence: float | None = None,
    tolerance_seconds: float = 0.15,
    detector: str = "generic",
) -> dict[str, Any]:
    """执行工具并将检测峰与专家标注进行一对一匹配。

    Execute the tools and match detected peaks one-to-one with expert annotations.
    """
    if detector not in {"generic", "ecg"}:
        raise ValueError("detector must be 'generic' or 'ecg'.")
    if detector == "ecg" and prominence is not None:
        raise ValueError("prominence applies only to the generic detector.")
    reference_path = Path(reference_file)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    sampling_rate = float(reference["sampling_rate_hz"])
    expected_samples = int(reference["num_samples"])
    reference_indices = [int(index) for index in reference["beat_indices"]]
    signal = load_signal(signal_file)
    if len(signal) != expected_samples:
        raise ValueError(f"Signal has {len(signal)} samples, but reference expects {expected_samples}.")
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative.")

    if detector == "ecg":
        peak_result = detect_ecg_r_peaks(signal, sampling_rate)
    else:
        peak_result = detect_peaks(
            signal,
            sampling_rate,
            min_distance_seconds=min_distance_seconds,
            prominence=prominence,
        )
    tolerance_samples = round(tolerance_seconds * sampling_rate)
    matching = match_peak_indices(peak_result["peak_indices"], reference_indices, tolerance_samples)
    reference_heart_rate = mean_heart_rate_from_annotations(reference_indices, sampling_rate)

    tool_error: str | None = None
    tool_heart_rate: dict[str, Any] | None
    try:
        if detector == "ecg":
            tool_heart_rate = calculate_ecg_heart_rate(signal, sampling_rate)
        else:
            tool_heart_rate = calculate_heart_rate(
                signal,
                sampling_rate,
                min_distance_seconds=min_distance_seconds,
                prominence=prominence,
            )
    except ValueError as error:
        tool_heart_rate = None
        tool_error = str(error)

    detected_heart_rate = None if tool_heart_rate is None else float(tool_heart_rate["mean_heart_rate_bpm"])
    return {
        "signal_file": str(signal_file),
        "reference_file": str(reference_file),
        "dataset": reference.get("dataset"),
        "record": reference.get("record"),
        "sampling_rate_hz": sampling_rate,
        "settings": {
            "detector": detector,
            "detector_config": peak_result.get("config"),
            "generic_min_distance_seconds": min_distance_seconds if detector == "generic" else None,
            "generic_prominence": prominence if detector == "generic" else None,
            "matching_tolerance_seconds": tolerance_seconds,
            "matching_tolerance_samples": tolerance_samples,
        },
        "statistics": calculate_statistics(signal, sampling_rate),
        "reference": {
            "num_beats": len(reference_indices),
            "mean_heart_rate_bpm": reference_heart_rate,
        },
        "detected": {
            "num_peaks": int(peak_result["num_peaks"]),
            "mean_heart_rate_bpm": detected_heart_rate,
            "heart_rate_error_bpm": (
                None if detected_heart_rate is None else abs(detected_heart_rate - reference_heart_rate)
            ),
            "tool_error": tool_error,
        },
        "peak_matching": matching,
        "interpretation": (
            f"This measures the {detector} peak detector against independent expert beat annotations; "
            "it is a signal-processing evaluation, not a clinical diagnosis."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-file", default="data/real/mitdb/100_30s/signal.csv")
    parser.add_argument("--reference-file", default="data/real/mitdb/100_30s/reference.json")
    parser.add_argument("--output", default="outputs/real_signal/mitdb_100_30s_tool_metrics.json")
    parser.add_argument("--min-distance-seconds", type=float, default=0.3)
    parser.add_argument("--prominence", type=float)
    parser.add_argument("--tolerance-seconds", type=float, default=0.15)
    parser.add_argument("--detector", choices=("generic", "ecg"), default="generic")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_real_signal(
        args.signal_file,
        args.reference_file,
        min_distance_seconds=args.min_distance_seconds,
        prominence=args.prominence,
        tolerance_seconds=args.tolerance_seconds,
        detector=args.detector,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    matching = result["peak_matching"]
    detected = result["detected"]
    print(f"记录：{result['record']}，采样率：{result['sampling_rate_hz']:g} Hz")
    print(f"参考心搏 / 检测峰：{result['reference']['num_beats']} / {detected['num_peaks']}")
    print(
        f"Sensitivity={matching['sensitivity']:.3f}, "
        f"PPV={matching['positive_predictive_value']:.3f}, F1={matching['f1']:.3f}"
    )
    if detected["mean_heart_rate_bpm"] is None:
        print(f"心率工具失败：{detected['tool_error']}")
    else:
        print(
            f"参考 / 工具心率：{result['reference']['mean_heart_rate_bpm']:.2f} / "
            f"{detected['mean_heart_rate_bpm']:.2f} BPM，"
            f"绝对误差 {detected['heart_rate_error_bpm']:.2f} BPM"
        )
    print(f"完整结果：{output}")


if __name__ == "__main__":
    main()
