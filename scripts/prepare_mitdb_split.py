"""按照预先固定的真实信号划分批量准备 MIT-BIH 片段。"""

from __future__ import annotations

import argparse

from physioagent.mitdb import prepare_mitdb_record
from physioagent.real_signal_suite import load_split_manifest, select_split_records


DEFAULT_MANIFEST = "evaluation/real_signal_split_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("development", "validation", "test"), default="validation")
    parser.add_argument(
        "--allow-frozen-test",
        action="store_true",
        help="仅在检测器配置已经冻结后用于一次性准备测试记录。",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, manifest_hash = load_split_manifest(args.manifest)
    records = select_split_records(
        manifest,
        [args.split],
        allow_frozen_test=args.allow_frozen_test,
    )
    print(f"划分文件 SHA-256：{manifest_hash}")
    print(f"准备 {args.split}：{[row['record'] for row in records]}")
    for row in records:
        reference = prepare_mitdb_record(
            str(row["record"]),
            float(row["duration_seconds"]),
            int(row["channel"]),
            row["data_dir"],
        )
        print(
            f"完成记录 {row['record']}：{reference['num_samples']} 点，"
            f"{reference['num_annotated_beats']} 个标注心搏"
        )


if __name__ == "__main__":
    main()
