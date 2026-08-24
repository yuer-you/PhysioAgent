"""评测原始 Qwen 的多步计划 JSON，并执行计划检查真实工具闭环。

Evaluate base-Qwen multi-step plan JSON and execute plans to test the real tool loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .agent import QwenAgent
from .lora_model import LoRAQwenGenerator
from .qwen_demo import DEFAULT_MODEL_PATH
from .real_signal import match_peak_indices
from .workflow import (
    WorkflowExecutor,
    WorkflowResponse,
    WorkflowStep,
    parse_workflow_plan,
    workflow_response_to_dict,
)
from .workflow_model import ModelWorkflowPlanner


DEFAULT_CASES_FILE = "evaluation/workflow_planning_cases_v1.jsonl"


def default_output_dir(prompt_version: str) -> str:
    return f"outputs/workflow/qwen_zero_shot_planning_{prompt_version}"


def load_workflow_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            required = {
                "id",
                "category",
                "record",
                "signal_file",
                "reference_file",
                "signal_profile",
                "question",
                "expected_steps",
            }
            if not required.issubset(case):
                raise ValueError(f"Line {line_number} is missing required fields.")
            if case["id"] in seen:
                raise ValueError(f"Duplicate case id: {case['id']}")
            seen.add(case["id"])
            expected_raw = json.dumps({"steps": case["expected_steps"]}, ensure_ascii=False)
            parse_workflow_plan(expected_raw)
            cases.append(case)
    if not cases:
        raise ValueError("No workflow planning cases found.")
    return cases


def steps_as_dicts(steps: list[WorkflowStep]) -> list[dict[str, Any]]:
    return [{"name": step.name, "arguments": step.arguments} for step in steps]


def evaluate_workflow_case(
    case: dict[str, Any],
    *,
    planner: ModelWorkflowPlanner | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    reference = json.loads(Path(case["reference_file"]).read_text(encoding="utf-8"))
    sampling_rate = float(reference["sampling_rate_hz"])
    if dry_run:
        raw_plan = json.dumps({"steps": case["expected_steps"]}, ensure_ascii=False, separators=(",", ":"))
    else:
        if planner is None:
            raise ValueError("planner is required unless dry_run=True.")
        messages = planner.build_messages(case["question"], case["signal_profile"])
        raw_plan = planner.generator.generate_messages(messages)

    row: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "record": case["record"],
        "question": case["question"],
        "expected_steps": case["expected_steps"],
        "raw_plan": raw_plan,
        "valid_plan": False,
        "plan_exact": False,
        "execution_success": False,
        "reference_check_passed": False,
        "answer_grounded": False,
        "end_to_end_success": False,
    }
    generation_info = getattr(planner.generator, "last_generation_info", None) if planner else None
    if generation_info is not None:
        row["generation_info"] = dict(generation_info)
    try:
        steps = parse_workflow_plan(raw_plan)
        predicted_steps = steps_as_dicts(steps)
        row["valid_plan"] = True
        row["predicted_steps"] = predicted_steps
        row["plan_exact"] = predicted_steps == case["expected_steps"]
    except ValueError as error:
        row["error_stage"] = "parse"
        row["error"] = str(error)
        return row

    try:
        response = WorkflowExecutor().run(
            case["question"],
            steps,
            case["signal_file"],
            sampling_rate,
            case["signal_profile"],
        )
        response.planner = "expected_dry_run" if dry_run else "model_zero_shot"
        response.raw_plan = raw_plan
        row["execution_success"] = True
        row["workflow_trace"] = workflow_response_to_dict(response, case["signal_file"], sampling_rate)
        check = validate_workflow_result(case, response, reference)
        row["reference_check"] = check
        row["reference_check_passed"] = bool(check["passed"])
        row["answer_grounded"] = workflow_answer_is_grounded(response)
    except (FileNotFoundError, ValueError) as error:
        row["error_stage"] = "execution"
        row["error"] = str(error)
        return row
    row["end_to_end_success"] = all(
        row[key]
        for key in ("plan_exact", "execution_success", "reference_check_passed", "answer_grounded")
    )
    return row


def validate_workflow_result(
    case: dict[str, Any],
    response: WorkflowResponse,
    reference: dict[str, Any],
) -> dict[str, Any]:
    expected_final_tool = case["expected_steps"][-1]["name"]
    if response.trace[-1].tool_name != expected_final_tool:
        return {"passed": False, "reason": "wrong final tool"}
    result = response.final_result
    expected_samples = int(reference["num_samples"])
    if expected_final_tool == "calculate_heart_rate":
        error = abs(float(result["mean_heart_rate_bpm"]) - float(reference["reference_mean_heart_rate_bpm"]))
        return {"passed": error <= 1.0, "heart_rate_absolute_error_bpm": error}
    if expected_final_tool == "detect_peaks":
        tolerance = round(0.15 * float(reference["sampling_rate_hz"]))
        matching = match_peak_indices(result["peak_indices"], reference["beat_indices"], tolerance)
        return {"passed": matching["f1"] >= 0.95, **matching}
    if expected_final_tool == "calculate_statistics":
        passed = (
            result["num_samples"] == expected_samples
            and math.isclose(result["duration_seconds"], float(reference["duration_seconds"]), abs_tol=1e-9)
        )
        return {"passed": bool(passed), "expected_samples": expected_samples}
    if expected_final_tool in {"filter_signal", "load_signal"}:
        passed = isinstance(result, np.ndarray) and len(result) == expected_samples and np.all(np.isfinite(result))
        return {"passed": bool(passed), "expected_samples": expected_samples}
    return {"passed": False, "reason": "unsupported final tool"}


def workflow_answer_is_grounded(response: WorkflowResponse) -> bool:
    result = response.final_result
    final_tool = response.trace[-1].tool_name
    if final_tool in {"load_signal", "filter_signal"}:
        return str(len(result)) in response.answer
    if final_tool == "calculate_statistics":
        return str(result["num_samples"]) in response.answer and f"{result['mean']:.3f}" in response.answer
    if final_tool == "detect_peaks":
        return str(result["num_peaks"]) in response.answer and str(result["peak_indices"]) in response.answer
    if final_tool == "calculate_heart_rate":
        return f"{result['mean_heart_rate_bpm']:.1f}" in response.answer and str(result["num_peaks"]) in response.answer
    return False


def summarize_workflow_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    categories = Counter(row["category"] for row in results)
    correct = Counter(row["category"] for row in results if row["end_to_end_success"])
    metrics = ("valid_plan", "plan_exact", "execution_success", "reference_check_passed", "answer_grounded", "end_to_end_success")
    return {
        "num_cases": total,
        "metrics": {name + "_rate": sum(bool(row[name]) for row in results) / total for name in metrics},
        "by_category": {
            category: {"correct": correct[category], "total": count}
            for category, count in sorted(categories.items())
        },
        "failed_ids": [row["id"] for row in results if not row["end_to_end_success"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--mode-label",
        default=None,
        help="可选的结果模式名称，例如 workflow_dpo_v1；只影响 summary 元数据。",
    )
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE)
    parser.add_argument("--workflow-prompt-version", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="模型最多生成多少个 token；较长的多步 JSON 建议使用 256。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="默认按提示词版本写入 outputs/workflow/qwen_zero_shot_planning_<version>。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_workflow_cases(args.cases_file)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    planner = None
    if not args.dry_run:
        generator = (
            LoRAQwenGenerator(
                args.model_path,
                args.adapter_path,
                max_new_tokens=args.max_new_tokens,
            )
            if args.adapter_path
            else QwenAgent(
                args.model_path,
                prompt_version="v4",
                max_new_tokens=args.max_new_tokens,
            )
        )
        planner = ModelWorkflowPlanner(generator, prompt_version=args.workflow_prompt_version)

    results = []
    for index, case in enumerate(cases, start=1):
        row = evaluate_workflow_case(case, planner=planner, dry_run=args.dry_run)
        results.append(row)
        status = "OK" if row["end_to_end_success"] else "FAIL"
        print(f"[{index:02d}/{len(cases)}] {case['id']}: {status} | {row['raw_plan']}")

    summary = summarize_workflow_results(results)
    output_root = args.output_dir or default_output_dir(args.workflow_prompt_version)
    result_variant = "dry_run" if args.dry_run else ("lora" if args.adapter_path else "base_qwen")
    output_dir = Path(output_root) / result_variant
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "evaluation": "workflow_planning_v1",
        "mode": args.mode_label or (
            "dry_run"
            if args.dry_run
            else (
                f"workflow_lora_{args.workflow_prompt_version}"
                if args.adapter_path
                else f"base_qwen_zero_shot_{args.workflow_prompt_version}"
            )
        ),
        "workflow_prompt_version": args.workflow_prompt_version,
        "max_new_tokens": None if args.dry_run else args.max_new_tokens,
        "base_model": None if args.dry_run else args.model_path,
        "adapter": None if args.dry_run else args.adapter_path,
        "cases_file": args.cases_file,
        **summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n计划严格匹配：{sum(row['plan_exact'] for row in results)}/{len(results)}")
    print(f"端到端成功：{sum(row['end_to_end_success'] for row in results)}/{len(results)}")
    print(f"完整结果：{output_dir}")


if __name__ == "__main__":
    main()
