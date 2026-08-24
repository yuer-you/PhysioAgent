"""从项目根目录运行：python scripts/generate_sft_data.py"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接执行脚本，而不要求先把当前项目安装为 Python 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physioagent.sft_data import write_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    paths = write_datasets(args.output_dir, seed=args.seed)
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
