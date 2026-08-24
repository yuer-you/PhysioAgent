"""在真实 ECG 上演示确定性的多步工具执行和内存状态传递。

Demonstrate deterministic multi-step tool execution and in-memory state transfer on a real ECG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import RuleBasedWorkflowAgent, workflow_response_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-file", default="data/real/mitdb/207_30s/signal.csv")
    parser.add_argument("--sampling-rate", type=float, default=360.0)
    parser.add_argument("--signal-profile", choices=("generic", "ecg"), default="ecg")
    parser.add_argument(
        "--question",
        default="请先用 0.5 到 40 Hz 对 ECG 滤波，再计算平均心率。",
    )
    parser.add_argument("--output", default="outputs/workflow/rule_based_record207.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    signal_file = Path(args.signal_file)
    if not signal_file.is_file():
        raise FileNotFoundError(f"Signal file does not exist: {signal_file}")
    response = RuleBasedWorkflowAgent().run(
        args.question,
        signal_file,
        args.sampling_rate,
        signal_profile=args.signal_profile,
    )

    print(f"问题：{response.question}")
    print(f"信号 profile：{response.signal_profile}")
    print("计划：")
    for step in response.plan:
        print(f"  - {step.name} {step.arguments}")
    print("执行轨迹：")
    for item in response.trace:
        print(f"  Step {item.step}: {item.tool_name}")
        print(f"    输入来源：{item.input_source}")
        print(f"    参数：{item.arguments}")
        print(f"    结果摘要：{item.result_summary}")
    print(f"停止原因：{response.stop_reason}")
    print(f"最终回答：{response.answer}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = workflow_response_to_dict(response, signal_file, args.sampling_rate)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"轨迹文件：{output}")


if __name__ == "__main__":
    main()
