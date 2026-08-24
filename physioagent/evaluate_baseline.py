"""评测未微调 Qwen 的工具选择和参数生成，不执行信号处理工具。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .agent import QwenAgent, ToolCall, parse_tool_call
from .qwen_demo import DEFAULT_MODEL_PATH


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        cases = [json.loads(line) for line in file if line.strip()]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case ids must be unique.")
    return cases


def score_call(case: dict[str, Any], call: ToolCall) -> dict[str, Any]:
    """严格匹配之外，单独衡量请求参数和多余参数。"""
    expected = case["expected_arguments"]
    predicted = call.arguments
    name_correct = call.name == case["expected_name"]
    requested_arguments_correct = name_correct and all(
        key in predicted and predicted[key] == value for key, value in expected.items()
    )
    extra_arguments = sorted(set(predicted) - set(expected)) if name_correct else []
    arguments_exact = name_correct and predicted == expected
    return {
        "predicted_name": call.name,
        "predicted_arguments": predicted,
        "name_correct": name_correct,
        "requested_arguments_correct": requested_arguments_correct,
        "extra_arguments": extra_arguments,
        "has_extra_arguments": bool(extra_arguments),
        "arguments_exact": arguments_exact,
        "exact_match": name_correct and arguments_exact,
    }


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    total = len(results)
    if total == 0:
        raise ValueError("Evaluation set is empty.")
    return {
        "valid_tool_call_rate": sum(item["valid_tool_call"] for item in results) / total,
        "tool_name_accuracy": sum(item["name_correct"] for item in results) / total,
        "requested_arguments_accuracy": sum(item["requested_arguments_correct"] for item in results) / total,
        "extra_argument_case_rate": sum(item["has_extra_arguments"] for item in results) / total,
        "arguments_exact_accuracy": sum(item["arguments_exact"] for item in results) / total,
        "exact_match_accuracy": sum(item["exact_match"] for item in results) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH),
    )
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).parents[1] / "evaluation" / "tool_calling_cases.jsonl"),
    )
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3", "v4"), default="v4")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    agent = QwenAgent(args.model_path, prompt_version=args.prompt_version)
    results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        raw = agent.generate_decision(case["question"])
        record: dict[str, Any] = {
            **case,
            "prompt_version": args.prompt_version,
            "raw_output": raw,
            "valid_tool_call": False,
            "name_correct": False,
            "requested_arguments_correct": False,
            "extra_arguments": [],
            "has_extra_arguments": False,
            "arguments_exact": False,
            "exact_match": False,
        }
        try:
            call = parse_tool_call(raw)
            record["valid_tool_call"] = True
            record.update(score_call(case, call))
        except ValueError as error:
            record["error"] = str(error)
        results.append(record)
        print(f"[{index:02d}/{len(cases)}] {case['id']}: {raw}")

    output_path = Path(args.output or f"outputs/qwen2.5-3b-prompt-{args.prompt_version}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    metrics = calculate_metrics(results)
    print(f"\n提示词版本：{args.prompt_version}")
    print(f"评测样本数：{len(results)}")
    for name, value in metrics.items():
        print(f"{name}: {value:.1%}")
    print("\n按工具类别的完全匹配率：")
    categories = sorted({item["category"] for item in results})
    for category in categories:
        subset = [item for item in results if item["category"] == category]
        accuracy = sum(item["exact_match"] for item in subset) / len(subset)
        print(f"{category}: {accuracy:.1%} ({sum(item['exact_match'] for item in subset)}/{len(subset)})")
    print(f"逐条结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
