"""生成 final v4；首次冻结后不得根据模型结果修改并重跑。

Generate final v4; after the first freeze, do not modify and rerun it based on model results.
"""

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physioagent.workflow_final_v4_data import write_workflow_final_v4_cases


def main() -> None:
    path = write_workflow_final_v4_cases(PROJECT_ROOT / "evaluation" / "workflow_final_cases_v4.jsonl")
    print(f"cases: {path}")
    print(f"sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
