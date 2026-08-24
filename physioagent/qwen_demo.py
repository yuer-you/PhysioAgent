"""本地 Qwen 工具调用演示。

Demonstrate local Qwen tool calling.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import QwenAgent


DEFAULT_MODEL_PATH = "models/Qwen2.5-3B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="从 Hugging Face 或 ModelScope 下载得到的本地模型目录。",
    )
    parser.add_argument("--question", default="这段信号的平均心率是多少？")
    parser.add_argument("--file", default=str(Path(__file__).parents[1] / "data" / "sample_ecg.csv"))
    parser.add_argument("--sampling-rate", type=float, default=25)
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3", "v4"), default="v4")
    args = parser.parse_args()

    response = QwenAgent(args.model_path, prompt_version=args.prompt_version).run(
        args.question, args.file, args.sampling_rate
    )
    print(f"提示词版本：{args.prompt_version}")
    print(f"模型原始输出：{response.raw_decision}")
    print(f"执行工具：{response.tool_name}")
    print(f"工具参数：{response.tool_arguments}")
    print(f"最终回答：{response.answer}")


if __name__ == "__main__":
    main()
