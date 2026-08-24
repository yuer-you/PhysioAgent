"""使用 TRL DPOTrainer 从 Workflow LoRA v2 继续训练偏好 adapter。

默认配置针对单张 16GB A4000。首次必须使用 --dry-run --inspect-token-lengths，
该模式只检查数据、adapter 路径和 tokenizer，不加载模型权重或使用 GPU。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from .dpo_workflow_data import _validate_pair
from .lora_model import validate_adapter_directory


DEFAULT_MODEL_PATH = "models/Qwen2.5-3B-Instruct"
DEFAULT_SFT_ADAPTER = "outputs/sft/qwen2.5-3b-workflow-lora-v2/final_adapter"
DEFAULT_TRAIN_FILE = "data/dpo_workflow_v1/train.jsonl"
DEFAULT_VALIDATION_FILE = "data/dpo_workflow_v1/validation.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/dpo/qwen2.5-3b-workflow-dpo-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"DPO dataset does not exist: {source}")
    rows = []
    with source.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from error
    return rows


def validate_dpo_rows(rows: list[dict[str, Any]], split: str) -> set[str]:
    if not rows:
        raise ValueError(f"{split} DPO dataset is empty.")
    ids: set[str] = set()
    questions: set[str] = set()
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in ids:
            raise ValueError(f"Invalid or duplicate id in {split} row {index}: {row_id!r}")
        ids.add(row_id)
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if not isinstance(prompt, list) or [message.get("role") for message in prompt] != ["system", "user"]:
            raise ValueError(f"{row_id}: prompt must contain system then user messages.")
        for name, completion in (("chosen", chosen), ("rejected", rejected)):
            if not isinstance(completion, list) or len(completion) != 1:
                raise ValueError(f"{row_id}: {name} must contain one assistant message.")
            if completion[0].get("role") != "assistant" or not completion[0].get("content"):
                raise ValueError(f"{row_id}: invalid {name} assistant message.")
        metadata = row.get("metadata", {})
        if metadata.get("task_type") != "workflow_preference" or metadata.get("split") != split:
            raise ValueError(f"{row_id}: invalid preference metadata.")
        _validate_pair(row)
        question = prompt[1]["content"].strip().casefold()
        if question in questions:
            raise ValueError(f"Duplicate DPO question inside {split}: {question}")
        questions.add(question)
    return questions


def validate_dpo_training_files(train_file: str | Path, validation_file: str | Path) -> tuple[int, int]:
    train_rows = load_jsonl(train_file)
    validation_rows = load_jsonl(validation_file)
    train_questions = validate_dpo_rows(train_rows, "train")
    validation_questions = validate_dpo_rows(validation_rows, "validation")
    overlap = train_questions & validation_questions
    if overlap:
        raise ValueError(f"DPO train/validation question leakage: {next(iter(overlap))}")
    return len(train_rows), len(validation_rows)


def inspect_token_lengths(
    model_path: str | Path,
    sft_adapter_path: str | Path,
    train_file: str | Path,
    validation_file: str | Path,
    max_prompt_length: int,
    max_completion_length: int,
    max_length: int,
) -> dict[str, float | int]:
    from transformers import AutoTokenizer

    model = Path(model_path)
    adapter = Path(sft_adapter_path)
    if not model.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model}")
    validate_adapter_directory(adapter)
    tokenizer_source = adapter if (adapter / "tokenizer_config.json").is_file() else model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    rows = load_jsonl(train_file) + load_jsonl(validation_file)
    prompt_lengths: list[int] = []
    chosen_completion_lengths: list[int] = []
    rejected_completion_lengths: list[int] = []
    full_lengths: list[int] = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True
        )
        chosen_full = tokenizer.apply_chat_template(row["prompt"] + row["chosen"], tokenize=True)
        rejected_full = tokenizer.apply_chat_template(row["prompt"] + row["rejected"], tokenize=True)
        prompt_length = len(prompt_ids)
        prompt_lengths.append(prompt_length)
        chosen_completion_lengths.append(max(0, len(chosen_full) - prompt_length))
        rejected_completion_lengths.append(max(0, len(rejected_full) - prompt_length))
        full_lengths.extend((len(chosen_full), len(rejected_full)))
    summary: dict[str, float | int] = {
        "num_pairs": len(rows),
        "max_prompt_tokens": max(prompt_lengths),
        "max_chosen_completion_tokens": max(chosen_completion_lengths),
        "max_rejected_completion_tokens": max(rejected_completion_lengths),
        "max_full_tokens": max(full_lengths),
        "configured_max_prompt_length": max_prompt_length,
        "configured_max_completion_length": max_completion_length,
        "configured_max_length": max_length,
    }
    if summary["max_prompt_tokens"] > max_prompt_length:
        raise ValueError("A DPO prompt exceeds --max-prompt-length; increase it instead of truncating the system prompt.")
    if max(summary["max_chosen_completion_tokens"], summary["max_rejected_completion_tokens"]) > max_completion_length:
        raise ValueError("A DPO completion exceeds --max-completion-length.")
    if summary["max_full_tokens"] > max_length:
        raise ValueError("A DPO prompt-completion sequence exceeds --max-length.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--sft-adapter-path", default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=896)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect-token-lengths", action="store_true")
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_length": args.max_length,
        "logging_steps": args.logging_steps,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.max_prompt_length + args.max_completion_length > args.max_length:
        raise ValueError("--max-prompt-length + --max-completion-length must not exceed --max-length.")


def _ensure_safe_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"DPO output directory is not empty: {output}. Use a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")

    import torch
    import transformers
    import trl
    from datasets import Dataset, DatasetDict
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for DPO training.")
    model_path = Path(args.model_path)
    adapter_path = Path(args.sft_adapter_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
    validate_adapter_directory(adapter_path)
    output = _ensure_safe_output(args.output_dir)
    set_seed(args.seed)

    tokenizer_source = adapter_path if (adapter_path / "tokenizer_config.json").is_file() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        adapter_name="default",
        is_trainable=True,
        local_files_only=True,
    )
    # reference 是 DPO 开始前 SFT adapter 的冻结副本；DPOTrainer 会在同一模型内切换 adapter。
    model.load_adapter(
        adapter_path,
        adapter_name="reference",
        is_trainable=False,
        local_files_only=True,
    )
    model.set_adapter("default")
    reference_trainable = [
        name for name, parameter in model.named_parameters() if ".reference." in name and parameter.requires_grad
    ]
    if reference_trainable:
        raise RuntimeError("Reference adapter unexpectedly contains trainable parameters.")
    model.print_trainable_parameters()

    raw_train = load_jsonl(args.train_file)
    raw_validation = load_jsonl(args.validation_file)
    columns = ("prompt", "chosen", "rejected")
    dataset = DatasetDict(
        {
            "train": Dataset.from_list([{key: row[key] for key in columns} for row in raw_train]),
            "validation": Dataset.from_list([{key: row[key] for key in columns} for row in raw_validation]),
        }
    )
    training_config = DPOConfig(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=["sigmoid"],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_length=args.max_length,
        truncation_mode="keep_start",
        model_adapter_name="default",
        ref_adapter_name="reference",
        precompute_ref_log_probs=False,
        fp16=True,
        bf16=False,
        tf32=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    result = trainer.train()
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    trainer.save_metrics("all", metrics)
    trainer.save_state()

    final_adapter = output / "final_adapter"
    trainer.model.set_adapter("default")
    trainer.model.save_pretrained(final_adapter, selected_adapters=["default"])
    tokenizer.save_pretrained(final_adapter)
    validate_adapter_directory(final_adapter)

    run_manifest = {
        "base_model": str(model_path),
        "initial_sft_adapter": str(adapter_path),
        "reference_policy": "frozen copy of initial_sft_adapter named reference",
        "trained_adapter": str(final_adapter),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "train_file_sha256": file_sha256(args.train_file),
        "validation_file_sha256": file_sha256(args.validation_file),
        "test_file_used": False,
        "train_pairs": len(dataset["train"]),
        "validation_pairs": len(dataset["validation"]),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "loss_type": "sigmoid",
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size_per_process": args.batch_size * args.gradient_accumulation_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_length": args.max_length,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "visible_gpu_count": torch.cuda.device_count(),
        },
        "metrics": metrics,
    }
    with (output / "run_manifest.json").open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str) + "\n")
    print(f"DPO 训练完成。Adapter：{final_adapter}")


def main() -> None:
    args = build_parser().parse_args()
    _validate_cli(args)
    train_count, validation_count = validate_dpo_training_files(args.train_file, args.validation_file)
    print(f"DPO 数据检查通过：train={train_count}, validation={validation_count}, test=未读取")
    if args.dry_run:
        validate_adapter_directory(args.sft_adapter_path)
        if args.inspect_token_lengths:
            summary = inspect_token_lengths(
                args.model_path,
                args.sft_adapter_path,
                args.train_file,
                args.validation_file,
                args.max_prompt_length,
                args.max_completion_length,
                args.max_length,
            )
            print(f"DPO Token 长度检查：{json.dumps(summary, ensure_ascii=False)}")
        print("DPO dry run 完成；没有加载模型权重或使用 GPU。")
        return
    train(args)


if __name__ == "__main__":
    main()
