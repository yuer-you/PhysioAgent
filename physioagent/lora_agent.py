"""把 SFT LoRA 模型接入可执行的 PhysioAgent 闭环。

模型只负责输出一个工具调用 JSON；信号读取、数值计算和最终回答均由
确定性的 Python 代码完成。这样可以避免语言模型自行编造分析结果。
"""

from __future__ import annotations

from pathlib import Path

from .agent import AgentResponse, ToolExecutor, parse_tool_call
from .lora_model import LoRAQwenGenerator, MessageGenerator
from .sft_data_v2 import SFT_SYSTEM_PROMPT_V2


class LoRAAgent:
    """使用 LoRA v2 生成工具调用，再执行相应的信号处理工具。

    ``generator`` 是一个依赖注入入口：正式运行时加载 Qwen + LoRA；测试时可
    传入轻量假对象，因此 CPU 环境也能检查整个 Agent loop。
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
        *,
        generator: MessageGenerator | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        if generator is None:
            if model_path is None or adapter_path is None:
                raise ValueError("model_path and adapter_path are required when generator is not provided.")
            generator = LoRAQwenGenerator(model_path, adapter_path, max_new_tokens=max_new_tokens)
        self.generator = generator
        self.executor = ToolExecutor()

    @staticmethod
    def _build_messages(question: str) -> list[dict[str, str]]:
        if not question.strip():
            raise ValueError("question must not be empty.")
        # 必须与 LoRA v2 的训练提示词一致，避免训练和实际使用时输入分布不同。
        return [
            {"role": "system", "content": SFT_SYSTEM_PROMPT_V2},
            {"role": "user", "content": question},
        ]

    def generate_decision(self, question: str) -> str:
        """返回模型原始输出，便于调试格式错误。"""
        return self.generator.generate_messages(self._build_messages(question))

    def run(
        self,
        question: str,
        file_path: str | Path,
        sampling_rate: float,
        signal_profile: str = "generic",
    ) -> AgentResponse:
        raw_decision = self.generate_decision(question)
        call = parse_tool_call(raw_decision)
        return self.executor.run(
            call,
            file_path,
            sampling_rate,
            raw_decision=raw_decision,
            signal_profile=signal_profile,
        )
