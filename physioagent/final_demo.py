"""运行最终 Workflow DPO v1 Agent：模型规划，确定性工具执行，保存完整轨迹。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .lora_model import LoRAQwenGenerator, MessageGenerator
from .workflow import WorkflowResponse, workflow_response_to_dict
from .workflow_model import ModelWorkflowAgent


DEFAULT_MODEL_PATH = "models/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER_PATH = "outputs/dpo/qwen2.5-3b-workflow-dpo-v1/final_adapter"
DEFAULT_SIGNAL_FILE = "data/real/mitdb/207_30s/signal.csv"
DEFAULT_OUTPUT = "outputs/demo/workflow_dpo_v1_record207.json"
DEFAULT_QUESTION = "请先读取 signal 列，再用 0.5 到 40 Hz 对 ECG 滤波，最后计算平均心率。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--signal-file", default=DEFAULT_SIGNAL_FILE)
    parser.add_argument("--sampling-rate", type=float, default=360.0)
    parser.add_argument("--signal-profile", choices=("generic", "ecg"), default="ecg")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def run_final_workflow(
    generator: MessageGenerator,
    *,
    question: str,
    signal_file: str | Path,
    sampling_rate: float,
    signal_profile: str,
) -> WorkflowResponse:
    source = Path(signal_file)
    if not source.is_file():
        raise FileNotFoundError(f"Signal file does not exist: {source}")
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")
    # 最终实验固定 prompt v2，且不启用恢复层；原始模型计划始终保留。
    return ModelWorkflowAgent(
        generator,
        prompt_version="v2",
        allow_recovery=False,
    ).run(
        question,
        source,
        sampling_rate,
        signal_profile=signal_profile,
    )


def build_demo_payload(
    response: WorkflowResponse,
    *,
    model_path: str,
    adapter_path: str,
    signal_file: str | Path,
    sampling_rate: float,
) -> dict[str, Any]:
    return {
        "demo": "physioagent_workflow_dpo_v1",
        "clinical_use": False,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "workflow_prompt_version": "v2",
        "strict_parser": True,
        "conservative_recovery": False,
        "workflow": workflow_response_to_dict(response, signal_file, sampling_rate),
    }


def write_demo_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return output


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    generator = LoRAQwenGenerator(
        args.model_path,
        args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )
    response = run_final_workflow(
        generator,
        question=args.question,
        signal_file=args.signal_file,
        sampling_rate=args.sampling_rate,
        signal_profile=args.signal_profile,
    )

    print("注意：本项目仅用于学习和软件评测，不用于临床诊断。")
    print(f"问题：{response.question}")
    print(f"模型原始计划：{response.raw_plan}")
    for item in response.trace:
        print(
            f"Step {item.step}: {item.tool_name} {item.arguments} | "
            f"输入={item.input_source} | 结果={item.result_summary}"
        )
    print(f"停止原因：{response.stop_reason}")
    print(f"最终回答：{response.answer}")

    payload = build_demo_payload(
        response,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        signal_file=args.signal_file,
        sampling_rate=args.sampling_rate,
    )
    output = write_demo_payload(args.output, payload)
    print(f"完整轨迹：{output}")


if __name__ == "__main__":
    main()
