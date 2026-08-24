"""对 Workflow SFT 留出集做确定性生成评测，不执行生理信号工具。

训练中的 eval_loss 使用 teacher forcing；本模块让模型真正逐条生成完整 JSON，
并按步骤数、加载策略、语言和任务类别统计严格计划匹配率。

Run deterministic generation evaluation on the held-out Workflow SFT set without executing physiological
signal tools. Unlike teacher-forced eval_loss, this module makes the model generate complete JSON for each
case and reports strict plan-match rates by step count, loading strategy, language, and task category.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .lora_model import LoRAQwenGenerator, MessageGenerator
from .qwen_demo import DEFAULT_MODEL_PATH
from .workflow import parse_workflow_plan


DEFAULT_ADAPTER_PATH = "outputs/sft/qwen2.5-3b-workflow-lora-v2/final_adapter"
DEFAULT_VALIDATION_FILE = "data/sft_workflow_v2/validation.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/workflow/workflow_lora_v2_sft_validation"


def _steps_as_dicts(raw: str) -> list[dict[str, Any]]:
    return [{"name": step.name, "arguments": step.arguments} for step in parse_workflow_plan(raw)]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_workflow_validation_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = row.get("id")
            prompt = row.get("prompt")
            completion = row.get("completion")
            metadata = row.get("metadata", {})
            if not isinstance(row_id, str) or row_id in seen_ids:
                raise ValueError(f"Line {line_number}: invalid or duplicate id: {row_id!r}")
            seen_ids.add(row_id)
            if not isinstance(prompt, list) or [message.get("role") for message in prompt] != ["system", "user"]:
                raise ValueError(f"{row_id}: prompt must contain system then user messages.")
            if not isinstance(completion, list) or len(completion) != 1 or completion[0].get("role") != "assistant":
                raise ValueError(f"{row_id}: completion must contain one assistant message.")
            if metadata.get("task_type") != "workflow":
                raise ValueError(f"{row_id}: expected workflow metadata.")
            expected_steps = metadata.get("expected_steps")
            if _steps_as_dicts(completion[0].get("content", "")) != expected_steps:
                raise ValueError(f"{row_id}: completion and metadata disagree.")
            cases.append(
                {
                    "id": row_id,
                    "question": prompt[1]["content"],
                    "messages": prompt,
                    "expected_text": completion[0]["content"],
                    "expected_steps": expected_steps,
                    "category": metadata.get("category"),
                    "step_count": metadata.get("step_count"),
                    "load_policy": metadata.get("load_policy"),
                    "language": metadata.get("language"),
                    "column_kind": metadata.get("column_kind"),
                    "paraphrase_split": metadata.get("paraphrase_split"),
                }
            )
    if not cases:
        raise ValueError("No Workflow SFT validation cases found.")
    return cases


def evaluate_validation_cases(
    cases: list[dict[str, Any]],
    generator: MessageGenerator | None = None,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if generator is None and not dry_run:
        raise ValueError("generator is required unless dry_run=True.")
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        raw = case["expected_text"] if dry_run else generator.generate_messages(case["messages"])
        row: dict[str, Any] = {
            "id": case["id"],
            "question": case["question"],
            "category": case["category"],
            "step_count": case["step_count"],
            "load_policy": case["load_policy"],
            "language": case["language"],
            "column_kind": case["column_kind"],
            "expected_steps": case["expected_steps"],
            "raw_output": raw,
            "valid_plan": False,
            "plan_exact": False,
        }
        generation_info = getattr(generator, "last_generation_info", None) if generator else None
        if generation_info is not None:
            row["generation_info"] = dict(generation_info)
        try:
            predicted_steps = _steps_as_dicts(raw)
            row["valid_plan"] = True
            row["predicted_steps"] = predicted_steps
            row["plan_exact"] = predicted_steps == case["expected_steps"]
            if not row["plan_exact"]:
                if len(predicted_steps) != len(case["expected_steps"]):
                    row["error_type"] = "step_count_mismatch"
                elif [step["name"] for step in predicted_steps] != [
                    step["name"] for step in case["expected_steps"]
                ]:
                    row["error_type"] = "tool_sequence_mismatch"
                else:
                    row["error_type"] = "arguments_mismatch"
        except ValueError as error:
            row["error_type"] = "invalid_plan"
            row["error"] = str(error)
        results.append(row)
        if not row["plan_exact"] or index % 25 == 0 or index == len(cases):
            status = "OK" if row["plan_exact"] else "FAIL"
            print(f"[{index:03d}/{len(cases)}] {case['id']}: {status}")
    return results


def _group_summary(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[str(row[key])].append(row)
    return {
        value: {
            "correct": sum(bool(row["plan_exact"]) for row in subset),
            "total": len(subset),
            "plan_exact_rate": sum(bool(row["plan_exact"]) for row in subset) / len(subset),
        }
        for value, subset in sorted(groups.items())
    }


def summarize_validation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    return {
        "num_cases": total,
        "metrics": {
            "valid_plan_rate": sum(bool(row["valid_plan"]) for row in results) / total,
            "plan_exact_rate": sum(bool(row["plan_exact"]) for row in results) / total,
        },
        "by_category": _group_summary(results, "category"),
        "by_step_count": _group_summary(results, "step_count"),
        "by_load_policy": _group_summary(results, "load_policy"),
        "by_language": _group_summary(results, "language"),
        "by_column_kind": _group_summary(results, "column_kind"),
        "error_types": dict(
            sorted(
                {
                    error_type: sum(row.get("error_type") == error_type for row in results)
                    for error_type in {row.get("error_type") for row in results if row.get("error_type")}
                }.items()
            )
        ),
        "failed_ids": [row["id"] for row in results if not row["plan_exact"]],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument(
        "--mode-label",
        default="workflow_lora_v2",
        help="写入 summary 的模型阶段名称，例如 workflow_dpo_v1。",
    )
    parser.add_argument("--validation-file", default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="使用期望标签验证评测器，不加载模型或 GPU。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    cases = load_workflow_validation_cases(args.validation_file)
    generator = None
    if not args.dry_run:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        generator = LoRAQwenGenerator(args.model_path, args.adapter_path, args.max_new_tokens)
    results = evaluate_validation_cases(cases, generator, dry_run=args.dry_run)
    destination = Path(args.output_dir) / ("dry_run" if args.dry_run else "lora")
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "results.jsonl", results)
    summary = {
        "evaluation": "workflow_sft_validation_generation_v1",
        "mode": "expected_dry_run" if args.dry_run else args.mode_label,
        "base_model": None if args.dry_run else str(args.model_path),
        "adapter": None if args.dry_run else str(args.adapter_path),
        "validation_file": str(args.validation_file),
        "validation_sha256": _sha256(args.validation_file),
        "max_new_tokens": args.max_new_tokens,
        **summarize_validation_results(results),
    }
    with (destination / "summary.json").open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    exact = summary["metrics"]["plan_exact_rate"]
    print(f"生成式验证完成：{summary['num_cases']} 条，严格计划准确率 {exact:.1%}")
    print(f"结果目录：{destination}")


if __name__ == "__main__":
    main()
