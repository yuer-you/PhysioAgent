"""评测 LoRA SFT 模型的工具选择与参数生成，不执行信号处理工具。

Evaluate LoRA SFT tool selection and argument generation without executing signal-processing tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .agent import parse_tool_call
from .evaluate_baseline import calculate_metrics, load_cases, score_call
from .lora_model import LoRAQwenGenerator, MessageGenerator, validate_adapter_directory
from .qwen_demo import DEFAULT_MODEL_PATH
from .sft_data import SFT_SYSTEM_PROMPT
from .sft_data_v2 import SFT_SYSTEM_PROMPT_V2


DEFAULT_ADAPTER_PATH = "outputs/sft/qwen2.5-3b-lora-v1/final_adapter"
DEFAULT_OUTPUT_DIR = "outputs/sft/qwen2.5-3b-lora-v1/evaluation"
DEFAULT_SFT_TEST_FILE = "data/sft/test.jsonl"
DEFAULT_TOOL_CASES_FILE = "evaluation/tool_calling_cases.jsonl"
DEFAULT_BASELINE_RESULTS = "outputs/qwen2.5-3b-prompt-v4.jsonl"
DEFAULT_FINAL_CASES_FILE = "evaluation/final_cases_v1.jsonl"
DEFAULT_FINAL_MANIFEST = "evaluation/final_cases_v1_manifest.json"


def load_sft_test_cases(path: str | Path) -> list[dict[str, Any]]:
    """把训练数据格式转换成通用评测格式，并再次核对标签。

    Convert training records to the common evaluation format and validate labels again.
    """
    rows = load_cases(path)
    cases: list[dict[str, Any]] = []
    for row in rows:
        prompt = row.get("prompt")
        completion = row.get("completion")
        metadata = row.get("metadata", {})
        if not isinstance(prompt, list) or [message.get("role") for message in prompt] != ["system", "user"]:
            raise ValueError(f"{row.get('id')}: invalid SFT test prompt.")
        if not isinstance(completion, list) or len(completion) != 1:
            raise ValueError(f"{row.get('id')}: invalid SFT test completion.")
        expected = parse_tool_call(completion[0].get("content", ""))
        if expected.name != metadata.get("tool_name") or expected.arguments != metadata.get("arguments"):
            raise ValueError(f"{row.get('id')}: completion and metadata disagree.")
        cases.append(
            {
                "id": row["id"],
                "category": expected.name,
                "question": prompt[1]["content"],
                "messages": prompt,
                "expected_name": expected.name,
                "expected_arguments": expected.arguments,
                "language": metadata.get("language"),
                "source": metadata.get("source"),
            }
        )
    return cases


def load_tool_calling_cases(
    path: str | Path, system_prompt: str = SFT_SYSTEM_PROMPT
) -> list[dict[str, Any]]:
    """给通用文本评测问题配上指定版本的简洁 system prompt。

    Attach the requested compact system-prompt version to generic text-evaluation questions.
    """
    cases = load_cases(path)
    normalized = []
    for case in cases:
        normalized.append(
            {
                **case,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": case["question"]},
                ],
            }
        )
    return normalized


def verify_frozen_final_test(cases_path: str | Path, manifest_path: str | Path) -> str:
    """最终评测前核对文件哈希，避免考题在冻结后被无意修改。

    Verify the file hash before final evaluation so frozen questions cannot change unnoticed.
    """
    cases = Path(cases_path)
    manifest_file = Path(manifest_path)
    if not cases.is_file() or not manifest_file.is_file():
        raise FileNotFoundError("Frozen final cases and manifest must both exist.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise ValueError("Final test manifest must have status='frozen'.")
    actual = hashlib.sha256(cases.read_bytes()).hexdigest()
    expected = manifest.get("sha256")
    if actual != expected:
        raise ValueError(f"Frozen final test hash mismatch: expected {expected}, got {actual}.")
    return actual


def evaluate_cases(
    cases: list[dict[str, Any]], generator: MessageGenerator, suite_name: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        raw = generator.generate_messages(case["messages"])
        record: dict[str, Any] = {
            "id": case["id"],
            "suite": suite_name,
            "category": case["category"],
            "question": case["question"],
            "expected_name": case["expected_name"],
            "expected_arguments": case["expected_arguments"],
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
        status = "OK" if record["exact_match"] else "FAIL"
        print(f"[{index:03d}/{len(cases)}] {suite_name} {case['id']}: {status} | {raw}")
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result)
    return {
        "num_cases": len(results),
        "metrics": calculate_metrics(results),
        "by_category": {
            category: {
                "correct": sum(item["exact_match"] for item in subset),
                "total": len(subset),
                "exact_match_accuracy": sum(item["exact_match"] for item in subset) / len(subset),
            }
            for category, subset in sorted(by_category.items())
        },
        "failed_ids": [item["id"] for item in results if not item["exact_match"]],
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _baseline_comparison(path: str | Path, adapter_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline_path = Path(path)
    if not baseline_path.is_file():
        return None
    baseline = load_cases(baseline_path)
    baseline_by_id = {item["id"]: item for item in baseline}
    adapter_ids = {item["id"] for item in adapter_results}
    if set(baseline_by_id) != adapter_ids:
        raise ValueError("Prompt-v4 and LoRA tool-calling results must contain the same case ids.")
    baseline_metrics = calculate_metrics(baseline)
    adapter_metrics = calculate_metrics(adapter_results)
    return {
        "baseline_file": str(baseline_path),
        "baseline_prompt": "v4",
        "baseline_metrics": baseline_metrics,
        "adapter_metrics": adapter_metrics,
        "metric_delta_adapter_minus_baseline": {
            name: adapter_metrics[name] - baseline_metrics[name] for name in adapter_metrics
        },
        "baseline_exact_matches": sum(item["exact_match"] for item in baseline),
        "adapter_exact_matches": sum(item["exact_match"] for item in adapter_results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--sft-test-file", default=DEFAULT_SFT_TEST_FILE)
    parser.add_argument("--tool-cases-file", default=DEFAULT_TOOL_CASES_FILE)
    parser.add_argument("--final-cases-file", default=DEFAULT_FINAL_CASES_FILE)
    parser.add_argument("--final-manifest", default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--baseline-results", default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--suite",
        choices=("all", "both", "sft-test", "tool-calling", "final"),
        default="both",
        help="both 保持旧行为（开发测试 + 60 条）；all 额外运行冻结最终测试。",
    )
    parser.add_argument("--system-prompt-version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = SFT_SYSTEM_PROMPT_V2 if args.system_prompt_version == "v2" else SFT_SYSTEM_PROMPT

    suites: dict[str, list[dict[str, Any]]] = {}
    if args.suite in {"all", "both", "sft-test"}:
        suites["sft_test_100"] = load_sft_test_cases(args.sft_test_file)
    if args.suite in {"all", "both", "tool-calling"}:
        suites["tool_calling_60"] = load_tool_calling_cases(args.tool_cases_file, system_prompt)
    final_hash = None
    if args.suite in {"all", "final"}:
        final_hash = verify_frozen_final_test(args.final_cases_file, args.final_manifest)
        suites["frozen_final_100"] = load_tool_calling_cases(args.final_cases_file, system_prompt)

    print(f"加载基础模型：{args.model_path}")
    print(f"加载 LoRA adapter：{args.adapter_path}")
    generator = LoRAQwenGenerator(args.model_path, args.adapter_path, args.max_new_tokens)
    summary: dict[str, Any] = {
        "base_model": str(args.model_path),
        "adapter": str(args.adapter_path),
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "system_prompt_version": args.system_prompt_version,
        "system_prompt": system_prompt,
        "frozen_final_test_sha256": final_hash,
        "suites": {},
    }
    tool_results: list[dict[str, Any]] | None = None
    for name, cases in suites.items():
        results = evaluate_cases(cases, generator, name)
        write_jsonl(output_dir / f"{name}.jsonl", results)
        summary["suites"][name] = summarize_results(results)
        if name == "tool_calling_60":
            tool_results = results

    if tool_results is not None:
        summary["prompt_v4_comparison"] = _baseline_comparison(args.baseline_results, tool_results)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n评测汇总：")
    for name, suite_summary in summary["suites"].items():
        exact = suite_summary["metrics"]["exact_match_accuracy"]
        print(f"{name}: {suite_summary['num_cases']} 条，完全匹配率 {exact:.1%}")
    comparison = summary.get("prompt_v4_comparison")
    if comparison:
        print(
            "Prompt v4 -> LoRA（60 条）: "
            f"{comparison['baseline_exact_matches']}/60 -> {comparison['adapter_exact_matches']}/60"
        )
    print(f"完整结果：{output_dir}")


if __name__ == "__main__":
    main()
