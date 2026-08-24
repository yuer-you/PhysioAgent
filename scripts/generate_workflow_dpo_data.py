"""从项目根目录运行：python scripts/generate_workflow_dpo_data.py

Run from the project root: python scripts/generate_workflow_dpo_data.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physioagent.dpo_workflow_data import WORKFLOW_DPO_SEED, write_workflow_dpo_datasets


def main() -> None:
    output = PROJECT_ROOT / "data" / "dpo_workflow_v1"
    paths = write_workflow_dpo_datasets(output, WORKFLOW_DPO_SEED)
    for split, path in paths.items():
        print(f"{split}: {path}")
    print(f"manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    main()
