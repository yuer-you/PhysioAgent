"""使用本地原始 Qwen 生成多步计划，并在真实 ECG 上执行。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import QwenAgent
from .qwen_demo import DEFAULT_MODEL_PATH
from .workflow import workflow_response_to_dict
from .workflow_model import ModelWorkflowAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--signal-file", default="data/real/mitdb/207_30s/signal.csv")
    parser.add_argument("--sampling-rate", type=float, default=360.0)
    parser.add_argument("--signal-profile", choices=("generic", "ecg"), default="ecg")
    parser.add_argument("--workflow-prompt-version", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--allow-conservative-recovery",
        action="store_true",
        help="仅恢复唯一可确定的缺失 steps 数组右括号，并在轨迹中记录。",
    )
    parser.add_argument(
        "--question",
        default="请先用 0.5 到 40 Hz 对 ECG 滤波，再计算平均心率。",
    )
    parser.add_argument("--output", default="outputs/workflow/qwen_zero_shot_record207.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    qwen = QwenAgent(
        args.model_path,
        prompt_version="v4",
        max_new_tokens=args.max_new_tokens,
    )
    response = ModelWorkflowAgent(
        qwen,
        prompt_version=args.workflow_prompt_version,
        allow_recovery=args.allow_conservative_recovery,
    ).run(
        args.question,
        args.signal_file,
        args.sampling_rate,
        signal_profile=args.signal_profile,
    )
    print(f"问题：{response.question}")
    print(f"模型原始计划：{response.raw_plan}")
    if response.plan_recovery and response.plan_recovery["recovery_applied"]:
        print(f"计划恢复：{response.plan_recovery['recovery_type']}")
        print(f"恢复后计划：{response.plan_recovery['effective_text']}")
    for item in response.trace:
        print(
            f"Step {item.step}: {item.tool_name} {item.arguments} | "
            f"输入={item.input_source} | 结果={item.result_summary}"
        )
    print(f"停止原因：{response.stop_reason}")
    print(f"最终回答：{response.answer}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = workflow_response_to_dict(response, args.signal_file, args.sampling_rate)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"轨迹文件：{output}")


if __name__ == "__main__":
    main()
