"""评测 LoRA 工具决策到真实 ECG 执行和 grounded answer 的完整闭环。"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .agent import AgentResponse, ToolCall, ToolExecutor, parse_tool_call
from .lora_agent import LoRAAgent
from .lora_model import LoRAQwenGenerator
from .qwen_demo import DEFAULT_MODEL_PATH
from .real_signal import match_peak_indices


DEFAULT_CASES_FILE = "evaluation/real_agent_cases_v1.jsonl"
DEFAULT_ADAPTER_PATH = "outputs/sft/qwen2.5-3b-lora-v2/final_adapter"
DEFAULT_OUTPUT_DIR = "outputs/end_to_end/real_agent_integration_v1"


def load_real_agent_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
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
                "expected_name",
                "expected_arguments",
            }
            if not required.issubset(case):
                raise ValueError(f"Line {line_number} is missing required fields.")
            if case["id"] in seen_ids:
                raise ValueError(f"Duplicate case id: {case['id']}")
            seen_ids.add(case["id"])
            # 用正式解析器再次验证人工标签，避免评测答案本身非法。
            parse_tool_call(
                json.dumps(
                    {"name": case["expected_name"], "arguments": case["expected_arguments"]},
                    ensure_ascii=False,
                )
            )
            if case["category"] != case["expected_name"]:
                raise ValueError(f"{case['id']}: category and expected_name must match.")
            cases.append(case)
    if not cases:
        raise ValueError("No real-agent cases found.")
    return cases


def evaluate_case(
    case: dict[str, Any],
    *,
    agent: LoRAAgent | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    reference = json.loads(Path(case["reference_file"]).read_text(encoding="utf-8"))
    sampling_rate = float(reference["sampling_rate_hz"])
    expected_payload = {
        "name": case["expected_name"],
        "arguments": case["expected_arguments"],
    }
    raw_output = json.dumps(expected_payload, ensure_ascii=False, separators=(",", ":")) if dry_run else None
    if not dry_run:
        if agent is None:
            raise ValueError("agent is required unless dry_run=True.")
        raw_output = agent.generate_decision(case["question"])

    row: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "record": case["record"],
        "question": case["question"],
        "signal_profile": case["signal_profile"],
        "expected_name": case["expected_name"],
        "expected_arguments": case["expected_arguments"],
        "raw_output": raw_output,
        "valid_tool_call": False,
        "tool_call_exact": False,
        "execution_success": False,
        "reference_check_passed": False,
        "answer_grounded": False,
        "end_to_end_success": False,
    }
    try:
        call = parse_tool_call(raw_output or "")
        row["valid_tool_call"] = True
        row["predicted_name"] = call.name
        row["predicted_arguments"] = call.arguments
        row["tool_call_exact"] = call.name == case["expected_name"] and call.arguments == case["expected_arguments"]
    except ValueError as error:
        row["error_stage"] = "parse"
        row["error"] = str(error)
        return row

    try:
        response = ToolExecutor().run(
            call,
            case["signal_file"],
            sampling_rate,
            raw_decision=raw_output,
            signal_profile=case["signal_profile"],
        )
        row["execution_success"] = True
        row["tool_result_summary"] = summarize_tool_result(response.tool_result)
        reference_check = validate_against_reference(case, response, reference)
        row["reference_check"] = reference_check
        row["reference_check_passed"] = bool(reference_check["passed"])
        row["answer"] = response.answer
        row["answer_grounded"] = answer_is_grounded(response)
    except (FileNotFoundError, ValueError) as error:
        row["error_stage"] = "execution"
        row["error"] = str(error)
        return row

    row["end_to_end_success"] = all(
        row[key]
        for key in (
            "tool_call_exact",
            "execution_success",
            "reference_check_passed",
            "answer_grounded",
        )
    )
    return row


def validate_against_reference(
    case: dict[str, Any],
    response: AgentResponse,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """按工具类型检查结果；这不是临床正确性声明。"""
    if response.tool_name != case["expected_name"]:
        return {"passed": False, "reason": "wrong tool was executed"}
    result = response.tool_result
    expected_samples = int(reference["num_samples"])

    if response.tool_name == "load_signal":
        passed = isinstance(result, np.ndarray) and len(result) == expected_samples and np.all(np.isfinite(result))
        return {"passed": bool(passed), "expected_samples": expected_samples}
    if response.tool_name == "calculate_statistics":
        passed = (
            result["num_samples"] == expected_samples
            and math.isclose(result["duration_seconds"], float(reference["duration_seconds"]), abs_tol=1e-9)
            and all(math.isfinite(float(result[key])) for key in ("mean", "std", "min", "max"))
        )
        return {"passed": bool(passed), "expected_samples": expected_samples}
    if response.tool_name == "detect_peaks":
        tolerance_samples = round(0.15 * float(reference["sampling_rate_hz"]))
        matching = match_peak_indices(result["peak_indices"], reference["beat_indices"], tolerance_samples)
        return {"passed": matching["f1"] >= 0.95, **matching}
    if response.tool_name == "calculate_heart_rate":
        error = abs(float(result["mean_heart_rate_bpm"]) - float(reference["reference_mean_heart_rate_bpm"]))
        return {"passed": error <= 1.0, "heart_rate_absolute_error_bpm": error}
    if response.tool_name == "filter_signal":
        original = np.genfromtxt(case["signal_file"], delimiter=",", names=True, dtype=float)["signal"]
        passed = (
            isinstance(result, np.ndarray)
            and len(result) == expected_samples
            and np.all(np.isfinite(result))
            and not np.allclose(result, original)
        )
        return {"passed": bool(passed), "expected_samples": expected_samples}
    return {"passed": False, "reason": "unsupported tool"}


def answer_is_grounded(response: AgentResponse) -> bool:
    result = response.tool_result
    if response.tool_name in {"load_signal", "filter_signal"}:
        return str(len(result)) in response.answer
    if response.tool_name == "calculate_statistics":
        return (
            str(result["num_samples"]) in response.answer
            and f"{result['duration_seconds']:.2f}" in response.answer
            and f"{result['mean']:.3f}" in response.answer
        )
    if response.tool_name == "detect_peaks":
        return str(result["num_peaks"]) in response.answer and str(result["peak_indices"]) in response.answer
    if response.tool_name == "calculate_heart_rate":
        return (
            f"{result['mean_heart_rate_bpm']:.1f}" in response.answer
            and str(result["num_peaks"]) in response.answer
        )
    return False


def summarize_tool_result(result: object) -> object:
    """避免把每条 10800 点数组完整写入评测文件。"""
    if isinstance(result, np.ndarray):
        return {
            "type": "ndarray",
            "num_samples": int(result.size),
            "mean": float(np.mean(result)),
            "std": float(np.std(result)),
            "min": float(np.min(result)),
            "max": float(np.max(result)),
            "first_five": [float(value) for value in result[:5]],
        }
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    by_category = Counter(row["category"] for row in results if row["end_to_end_success"])
    category_totals = Counter(row["category"] for row in results)
    metric_names = (
        "valid_tool_call",
        "tool_call_exact",
        "execution_success",
        "reference_check_passed",
        "answer_grounded",
        "end_to_end_success",
    )
    return {
        "num_cases": total,
        "metrics": {name + "_rate": sum(bool(row[name]) for row in results) / total for name in metric_names},
        "by_category": {
            category: {"correct": by_category[category], "total": count}
            for category, count in sorted(category_totals.items())
        },
        "failed_ids": [row["id"] for row in results if not row["end_to_end_success"]],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true", help="使用人工正确调用检查数据和执行层，不加载模型。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_real_agent_cases(args.cases_file)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    agent = None
    if not args.dry_run:
        generator = LoRAQwenGenerator(args.model_path, args.adapter_path, args.max_new_tokens)
        agent = LoRAAgent(generator=generator)

    results = []
    for index, case in enumerate(cases, start=1):
        row = evaluate_case(case, agent=agent, dry_run=args.dry_run)
        results.append(row)
        status = "OK" if row["end_to_end_success"] else "FAIL"
        print(f"[{index:02d}/{len(cases)}] {case['id']}: {status} | {row['raw_output']}")

    summary = summarize_results(results)
    output_dir = Path(args.output_dir)
    if args.dry_run:
        output_dir = output_dir / "dry_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "evaluation": "real_ecg_agent_integration_v1",
        "mode": "dry_run" if args.dry_run else "lora_v2",
        "base_model": None if args.dry_run else args.model_path,
        "adapter": None if args.dry_run else args.adapter_path,
        "cases_file": args.cases_file,
        **summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n端到端成功：{sum(row['end_to_end_success'] for row in results)}/{len(results)}")
    print(f"完整结果：{output_dir}")


if __name__ == "__main__":
    main()
