"""使用 TRL + PEFT 对本地 Qwen2.5-3B-Instruct 进行 LoRA SFT。

默认针对单张 16GB A4000。首次使用建议先运行 --dry-run，只检查数据，不加载模型。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

from .agent import parse_tool_call
from .workflow import parse_workflow_plan


DEFAULT_MODEL_PATH = "models/Qwen2.5-3B-Instruct"
DEFAULT_TRAIN_FILE = "data/sft/train.jsonl"
DEFAULT_VALIDATION_FILE = "data/sft/validation.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/sft/qwen2.5-3b-lora-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {source}")
    rows = []
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from error
    return rows


def validate_sft_rows(rows: list[dict[str, Any]], split: str) -> set[str]:
    """检查 TRL prompt-completion 结构和 assistant 工具调用标签。"""
    if not rows:
        raise ValueError(f"{split} dataset is empty.")
    questions: set[str] = set()
    ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in ids:
            raise ValueError(f"Invalid or duplicate id in {split} row {index}: {row_id!r}")
        ids.add(row_id)

        prompt = row.get("prompt")
        completion = row.get("completion")
        if not isinstance(prompt, list) or [message.get("role") for message in prompt] != ["system", "user"]:
            raise ValueError(f"{row_id}: prompt must contain system then user messages.")
        if not isinstance(completion, list) or len(completion) != 1 or completion[0].get("role") != "assistant":
            raise ValueError(f"{row_id}: completion must contain exactly one assistant message.")
        if not all(isinstance(message.get("content"), str) and message["content"] for message in prompt + completion):
            raise ValueError(f"{row_id}: every message must contain non-empty text.")

        question = prompt[1]["content"].strip().lower()
        if question in questions:
            raise ValueError(f"Duplicate question inside {split}: {question}")
        questions.add(question)
        metadata = row.get("metadata", {})
        if metadata.get("task_type") == "workflow":
            steps = parse_workflow_plan(completion[0]["content"])
            parsed_steps = [{"name": step.name, "arguments": step.arguments} for step in steps]
            if parsed_steps != metadata.get("expected_steps"):
                raise ValueError(f"{row_id}: workflow completion and metadata disagree.")
        else:
            call = parse_tool_call(completion[0]["content"])
            if call.name != metadata.get("tool_name") or call.arguments != metadata.get("arguments"):
                raise ValueError(f"{row_id}: completion and metadata disagree.")
    return questions


def validate_training_files(train_file: str | Path, validation_file: str | Path) -> tuple[int, int]:
    train_rows = load_jsonl(train_file)
    validation_rows = load_jsonl(validation_file)
    train_questions = validate_sft_rows(train_rows, "train")
    validation_questions = validate_sft_rows(validation_rows, "validation")
    overlap = train_questions & validation_questions
    if overlap:
        raise ValueError(f"Train/validation question leakage: {next(iter(overlap))}")
    return len(train_rows), len(validation_rows)


def inspect_token_lengths(
    model_path: str | Path,
    train_file: str | Path,
    validation_file: str | Path,
    max_length: int,
) -> dict[str, float | int]:
    """只加载 tokenizer 检查长度；不加载模型，也不需要 GPU。"""
    from transformers import AutoTokenizer

    model = Path(model_path)
    if not model.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows = load_jsonl(train_file) + load_jsonl(validation_file)
    lengths = sorted(
        len(tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True))
        for row in rows
    )
    summary: dict[str, float | int] = {
        "num_examples": len(lengths),
        "min_tokens": lengths[0],
        "mean_tokens": sum(lengths) / len(lengths),
        "p95_tokens": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
        "max_tokens": lengths[-1],
        "configured_max_length": max_length,
    }
    if lengths[-1] > max_length:
        raise ValueError(
            f"At least one example has {lengths[-1]} tokens, exceeding --max-length {max_length}. "
            "Increase max length instead of silently truncating the workflow JSON."
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只验证数据与路径，不加载训练库、模型或 GPU。")
    parser.add_argument(
        "--inspect-token-lengths",
        action="store_true",
        help="与 --dry-run 配合：额外加载 tokenizer 检查长度，但不加载模型或 GPU。",
    )
    return parser


def _validate_cli(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "logging_steps": args.logging_steps,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must satisfy 0 <= value < 1.")


def _ensure_safe_output(args: argparse.Namespace) -> Path:
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite_output_dir and not args.resume_from_checkpoint:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use a new directory, --resume-from-checkpoint, "
            "or explicitly pass --overwrite-output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def train(args: argparse.Namespace) -> None:
    # 计算节点离线；这些变量阻止依赖库在本地模型缺文件时悄悄访问网络。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WANDB_DISABLED", "true")

    import torch
    import transformers
    import trl
    from datasets import Dataset, DatasetDict
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for SFT training.")
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
    output = _ensure_safe_output(args)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    # 只把 TRL 所需字段交给 Arrow，避免 metadata 中不同工具参数形成复杂嵌套 schema。
    raw_train = load_jsonl(args.train_file)
    raw_validation = load_jsonl(args.validation_file)
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [{"prompt": row["prompt"], "completion": row["completion"]} for row in raw_train]
            ),
            "validation": Dataset.from_list(
                [{"prompt": row["prompt"], "completion": row["completion"]} for row in raw_validation]
            ),
        }
    )
    token_lengths = [
        len(tokenizer.apply_chat_template(row["prompt"] + row["completion"], tokenize=True))
        for row in raw_train + raw_validation
    ]
    max_observed_tokens = max(token_lengths)
    if max_observed_tokens > args.max_length:
        raise ValueError(
            f"At least one example has {max_observed_tokens} tokens, exceeding --max-length "
            f"{args.max_length}. Increase max length instead of silently truncating the tool call."
        )
    print(f"Token 长度检查：最大 {max_observed_tokens}，上限 {args.max_length}。")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    training_config = SFTConfig(
        output_dir=str(output),
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        fp16=True,
        bf16=False,
        tf32=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_adapter = output / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(final_adapter)
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    trainer.save_metrics("all", metrics)
    trainer.save_state()

    run_manifest = {
        "base_model": str(model_path),
        "adapter": str(final_adapter),
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "train_file_sha256": file_sha256(args.train_file),
        "validation_file_sha256": file_sha256(args.validation_file),
        "test_file_used": False,
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size_per_process": args.batch_size * args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "max_observed_tokens": max_observed_tokens,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": ["q_proj", "v_proj"]},
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
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"训练完成。LoRA adapter：{final_adapter}")


def main() -> None:
    args = build_parser().parse_args()
    _validate_cli(args)
    train_count, validation_count = validate_training_files(args.train_file, args.validation_file)
    print(f"数据检查通过：train={train_count}, validation={validation_count}, test=未读取")
    if args.dry_run:
        if args.inspect_token_lengths:
            summary = inspect_token_lengths(
                args.model_path,
                args.train_file,
                args.validation_file,
                args.max_length,
            )
            print(f"Token 长度检查：{json.dumps(summary, ensure_ascii=False)}")
        print("Dry run 完成；没有加载模型或使用 GPU。")
        return
    train(args)


if __name__ == "__main__":
    main()
