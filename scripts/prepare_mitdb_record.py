"""从 PhysioNet 下载一小段 MIT-BIH ECG，并转换为项目使用的 CSV。

默认下载记录 100 的前 30 秒、第 0 通道，同时保存专家心搏标注用于独立评测。
首次运行需要服务器能够访问 PhysioNet；生成后其余步骤可离线完成。

Download a short MIT-BIH ECG excerpt from PhysioNet and convert it to the CSV format used by this project.
By default, the script downloads the first 30 seconds of channel 0 from record 100 and saves expert
heartbeat annotations for independent evaluation. The first run requires PhysioNet access; later steps
can run offline once the files have been generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from physioagent.mitdb import prepare_mitdb_record


DEFAULT_RECORD = "100"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--output-dir", default="data/real/mitdb/100_30s")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    reference = prepare_mitdb_record(
        args.record,
        args.duration_seconds,
        args.channel,
        output_dir,
    )
    print(
        f"信号：{output_dir / 'signal.csv'} "
        f"({reference['num_samples']} 点, {reference['sampling_rate_hz']:g} Hz)"
    )
    print(f"参考标注：{output_dir / 'reference.json'} ({reference['num_annotated_beats']} 个心搏)")
    print(f"标注参考平均心率：{reference['reference_mean_heart_rate_bpm']:.2f} BPM")


if __name__ == "__main__":
    main()
