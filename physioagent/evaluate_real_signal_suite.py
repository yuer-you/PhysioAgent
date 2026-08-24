"""按预先固定的数据划分汇总多个真实 ECG 片段的工具指标。

Aggregate tool metrics over multiple real ECG excerpts using the predefined data split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_real_signal import evaluate_real_signal
from .real_signal_suite import load_split_manifest, select_split_records, summarize_record_results


DEFAULT_MANIFEST = "evaluation/real_signal_split_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--split",
        choices=("development", "validation", "test"),
        action="append",
        help="可重复指定；省略时评测 development + validation。",
    )
    parser.add_argument("--allow-frozen-test", action="store_true")
    parser.add_argument("--min-distance-seconds", type=float, default=0.3)
    parser.add_argument("--prominence", type=float)
    parser.add_argument("--tolerance-seconds", type=float, default=0.15)
    parser.add_argument("--detector", choices=("generic", "ecg"), default="generic")
    parser.add_argument("--output", default="outputs/real_signal/baseline_v1_dev_validation.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    splits = args.split or ["development", "validation"]
    manifest, manifest_hash = load_split_manifest(args.manifest)
    records = select_split_records(manifest, splits, allow_frozen_test=args.allow_frozen_test)
    results = []
    for row in records:
        data_dir = Path(row["data_dir"])
        signal_file = data_dir / "signal.csv"
        reference_file = data_dir / "reference.json"
        if not signal_file.is_file() or not reference_file.is_file():
            raise FileNotFoundError(
                f"Missing data for record {row['record']} in {data_dir}. "
                f"Run scripts/prepare_mitdb_split.py --split {row['split']} first."
            )
        result = evaluate_real_signal(
            signal_file,
            reference_file,
            min_distance_seconds=args.min_distance_seconds,
            prominence=args.prominence,
            tolerance_seconds=args.tolerance_seconds,
            detector=args.detector,
        )
        result["split"] = row["split"]
        results.append(result)
        heart_rate_error = result["detected"]["heart_rate_error_bpm"]
        heart_rate_text = "不可计算" if heart_rate_error is None else f"{heart_rate_error:.2f} BPM"
        print(
            f"记录 {row['record']} ({row['split']}): "
            f"F1={result['peak_matching']['f1']:.3f}, "
            f"心率误差={heart_rate_text}"
        )

    report = {
        "evaluation": "mitdb_real_signal_suite_v1",
        "manifest": args.manifest,
        "manifest_sha256": manifest_hash,
        "evaluated_splits": splits,
        "frozen_test_was_allowed": args.allow_frozen_test,
        "settings": {
            "detector": args.detector,
            "generic_min_distance_seconds": args.min_distance_seconds if args.detector == "generic" else None,
            "generic_prominence": args.prominence if args.detector == "generic" else None,
            "tolerance_seconds": args.tolerance_seconds,
        },
        "summary": summarize_record_results(results),
        "records": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"汇总：micro F1={summary['micro_f1']:.3f}，"
        f"平均心率绝对误差={summary['mean_absolute_heart_rate_error_bpm']:.2f} BPM"
    )
    print(f"完整结果：{output}")


if __name__ == "__main__":
    main()
