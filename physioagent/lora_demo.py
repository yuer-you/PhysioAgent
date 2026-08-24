"""在一份 CSV 信号上运行训练好的 LoRA v2 Agent。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .lora_agent import LoRAAgent


DEFAULT_MODEL_PATH = "models/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER_PATH = "outputs/sft/qwen2.5-3b-lora-v2/final_adapter"
DEFAULT_SIGNAL_PATH = "data/sample_ecg.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--signal-file", default=DEFAULT_SIGNAL_PATH)
    parser.add_argument("--sampling-rate", type=float, default=25.0)
    parser.add_argument(
        "--signal-profile",
        choices=("generic", "ecg"),
        default="generic",
        help="ecg 使用已经冻结的 ecg_detector_v1；该值由运行环境提供，不由模型生成。",
    )
    parser.add_argument("--question", default="这段信号的平均心率是多少？")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sampling_rate <= 0:
        raise ValueError("--sampling-rate must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    signal_file = Path(args.signal_file)
    if not signal_file.is_file():
        raise FileNotFoundError(f"Signal file does not exist: {signal_file}")

    # 集群模型已下载到本地挂载目录；禁止程序意外访问 Hugging Face 网络。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    agent = LoRAAgent(
        args.model_path,
        args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )
    response = agent.run(
        args.question,
        signal_file,
        args.sampling_rate,
        signal_profile=args.signal_profile,
    )

    print(f"问题：{args.question}")
    print(f"信号 profile：{response.signal_profile}")
    print(f"模型原始输出：{response.raw_decision}")
    print(f"执行工具：{response.tool_name}")
    print(f"工具参数：{response.tool_arguments}")
    print(f"工具原始结果：{response.tool_result}")
    print(f"基于工具结果的回答：{response.answer}")


if __name__ == "__main__":
    main()
