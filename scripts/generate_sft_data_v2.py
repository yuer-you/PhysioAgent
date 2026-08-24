"""从项目根目录运行：python scripts/generate_sft_data_v2.py"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physioagent.sft_data_v2 import V2_SEED, write_datasets_v2


def main() -> None:
    output = Path(__file__).parents[1] / "data" / "sft_v2"
    paths = write_datasets_v2(output, seed=V2_SEED)
    for split, path in paths.items():
        print(f"{split}: {path}")
    print(f"manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
