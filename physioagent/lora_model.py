"""Qwen 基础模型与 LoRA adapter 的共享加载器。

Shared loader for the Qwen base model and LoRA adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class MessageGenerator(Protocol):
    """Agent 和评测共同依赖的最小生成接口。

    Minimal generation interface shared by the agent and evaluators.
    """

    def generate_messages(self, messages: list[dict[str, str]]) -> str: ...


class LoRAQwenGenerator:
    """将 LoRA adapter 挂到本地 Qwen，并执行确定性生成。

    Attach a LoRA adapter to local Qwen and perform deterministic generation.
    """

    def __init__(self, model_path: str | Path, adapter_path: str | Path, max_new_tokens: int = 128) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = Path(model_path)
        self.adapter_path = Path(adapter_path)
        self.max_new_tokens = max_new_tokens
        self.last_generation_info: dict[str, Any] | None = None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for Qwen LoRA inference.")
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Base model directory does not exist: {self.model_path}")
        validate_adapter_directory(self.adapter_path)

        tokenizer_source = self.adapter_path if (self.adapter_path / "tokenizer_config.json").is_file() else self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path,
            is_trainable=False,
            local_files_only=True,
        )
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self._torch = torch

    def generate_messages(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        num_generated_tokens = int(generated.shape[0])
        self.last_generation_info = {
            "num_generated_tokens": num_generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "reached_max_new_tokens": num_generated_tokens >= self.max_new_tokens,
        }
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def validate_adapter_directory(path: str | Path) -> None:
    """确认 adapter 目录同时含配置和非空权重。

    Verify that the adapter directory contains both configuration and non-empty weights.
    """
    adapter = Path(path)
    if not adapter.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory does not exist: {adapter}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in: {adapter}")
    weight_files = (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin")
    if not any(weight.is_file() and weight.stat().st_size > 0 for weight in weight_files):
        raise FileNotFoundError(f"Missing non-empty adapter weights in: {adapter}")
