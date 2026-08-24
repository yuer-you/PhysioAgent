"""一次性运行冻结最终测试：Prompt v4、LoRA v1、预选 LoRA v2。

Run the frozen final test once for Prompt v4, LoRA v1, and the preselected LoRA v2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate_baseline import calculate_metrics, load_cases
from .evaluate_sft import summarize_results, validate_adapter_directory, verify_frozen_final_test
from .qwen_demo import DEFAULT_MODEL_PATH


DEFAULT_FINAL_CASES = "evaluation/final_cases_v1.jsonl"
DEFAULT_FINAL_MANIFEST = "evaluation/final_cases_v1_manifest.json"
DEFAULT_V1_ADAPTER = "outputs/sft/qwen2.5-3b-lora-v1/final_adapter"
DEFAULT_V2_ADAPTER = "outputs/sft/qwen2.5-3b-lora-v2/final_adapter"
DEFAULT_OUTPUT_DIR = "outputs/final_evaluation_v1"
MODEL_ORDER = ("prompt_v4", "lora_v1", "lora_v2")
REQUIRED_SCORE_FIELDS = {
    "valid_tool_call",
    "name_correct",
    "requested_arguments_correct",
    "has_extra_arguments",
    "arguments_exact",
    "exact_match",
}


def validate_result_rows(
    rows: list[dict[str, Any]], final_cases: list[dict[str, Any]], label: str
) -> None:
    """确认恢复使用的结果完整且确实对应同一份冻结测试。

    Verify that recovery inputs are complete and correspond to the same frozen test set.
    """
    expected = {case["id"]: case for case in final_cases}
    actual = {row["id"]: row for row in rows}
    if len(rows) != len(actual):
        raise ValueError(f"{label}: duplicate result ids.")
    if set(actual) != set(expected):
        raise ValueError(f"{label}: result ids do not match frozen final cases.")
    for case_id, row in actual.items():
        case = expected[case_id]
        if row.get("expected_name") != case["expected_name"]:
            raise ValueError(f"{label}/{case_id}: expected tool label changed.")
        if row.get("expected_arguments") != case["expected_arguments"]:
            raise ValueError(f"{label}/{case_id}: expected argument label changed.")
        missing = REQUIRED_SCORE_FIELDS - set(row)
        if missing:
            raise ValueError(f"{label}/{case_id}: missing score fields: {sorted(missing)}")


def _metric_deltas(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    return {name: left[name] - right[name] for name in left}


def build_final_summary(
    results: dict[str, list[dict[str, Any]]],
    *,
    frozen_sha256: str,
    model_path: str,
    v1_adapter: str,
    v2_adapter: str,
) -> dict[str, Any]:
    model_summaries = {name: summarize_results(results[name]) for name in MODEL_ORDER}
    metrics = {name: model_summaries[name]["metrics"] for name in MODEL_ORDER}
    return {
        "evaluation": "physioagent_frozen_final_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_final_test_sha256": frozen_sha256,
        "num_cases": len(results["prompt_v4"]),
        "precommitted_selection": {
            "selected_model": "lora_v2",
            "selection_used_final_results": False,
            "selection_basis": (
                "Best development exact match before final evaluation: "
                "Prompt v4 55/60, LoRA v1 57/60, LoRA v2 59/60, LoRA v2.1 58/60."
            ),
            "strict_stretch_gate_passed": False,
            "note": "LoRA v2 was frozen as the selected candidate; final results must not trigger retraining on this test.",
        },
        "configurations": {
            "prompt_v4": {"base_model": model_path, "adapter": None, "system_prompt": "prompt_v4"},
            "lora_v1": {"base_model": model_path, "adapter": v1_adapter, "system_prompt": "sft_v1"},
            "lora_v2": {"base_model": model_path, "adapter": v2_adapter, "system_prompt": "sft_v2"},
        },
        "models": model_summaries,
        "comparisons": {
            "lora_v1_minus_prompt_v4": _metric_deltas(metrics["lora_v1"], metrics["prompt_v4"]),
            "lora_v2_minus_prompt_v4": _metric_deltas(metrics["lora_v2"], metrics["prompt_v4"]),
            "lora_v2_minus_lora_v1": _metric_deltas(metrics["lora_v2"], metrics["lora_v1"]),
        },
    }


def _load_and_validate_result(
    path: Path, final_cases: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    rows = load_cases(path)
    validate_result_rows(rows, final_cases, label)
    return rows


def _run_or_resume(
    *,
    label: str,
    result_file: Path,
    command: list[str],
    final_cases: list[dict[str, Any]],
    resume: bool,
) -> list[dict[str, Any]]:
    if result_file.is_file():
        if not resume:
            raise FileExistsError(
                f"{label} result already exists: {result_file}. Use --resume only to verify and continue an interrupted run."
            )
        print(f"恢复检查通过前先验证已有结果：{label}")
        return _load_and_validate_result(result_file, final_cases, label)
    print(f"\n开始冻结最终评测：{label}")
    subprocess.run(command, check=True)
    if not result_file.is_file():
        raise RuntimeError(f"{label} finished without creating: {result_file}")
    return _load_and_validate_result(result_file, final_cases, label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("PHYSIOAGENT_MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--lora-v1-adapter", default=DEFAULT_V1_ADAPTER)
    parser.add_argument("--lora-v2-adapter", default=DEFAULT_V2_ADAPTER)
    parser.add_argument("--final-cases", default=DEFAULT_FINAL_CASES)
    parser.add_argument("--final-manifest", default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只校验路径、adapter 和冻结哈希，不加载模型。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Base model directory does not exist: {model_path}")
    validate_adapter_directory(args.lora_v1_adapter)
    validate_adapter_directory(args.lora_v2_adapter)
    frozen_hash = verify_frozen_final_test(args.final_cases, args.final_manifest)
    final_cases = load_cases(args.final_cases)
    if len(final_cases) != 100:
        raise ValueError(f"Frozen final test must contain 100 cases, got {len(final_cases)}.")
    print(f"冻结最终测试校验通过：100 条，SHA-256={frozen_hash}")
    print("预先选定最终候选：lora_v2（最终结果不参与模型选择）")
    if args.dry_run:
        print("Dry run 完成；没有加载模型或运行最终推理。")
        return

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Final output directory is not empty: {output}. Refusing to overwrite; use --resume only after interruption."
        )
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    prompt_file = output / "prompt_v4" / "results.jsonl"
    v1_dir = output / "lora_v1"
    v2_dir = output / "lora_v2"
    commands = {
        "prompt_v4": [
            python,
            "-m",
            "physioagent.evaluate_baseline",
            "--model-path",
            str(model_path),
            "--cases",
            args.final_cases,
            "--prompt-version",
            "v4",
            "--output",
            str(prompt_file),
        ],
        "lora_v1": [
            python,
            "-m",
            "physioagent.evaluate_sft",
            "--model-path",
            str(model_path),
            "--adapter-path",
            args.lora_v1_adapter,
            "--final-cases-file",
            args.final_cases,
            "--final-manifest",
            args.final_manifest,
            "--output-dir",
            str(v1_dir),
            "--system-prompt-version",
            "v1",
            "--suite",
            "final",
        ],
        "lora_v2": [
            python,
            "-m",
            "physioagent.evaluate_sft",
            "--model-path",
            str(model_path),
            "--adapter-path",
            args.lora_v2_adapter,
            "--final-cases-file",
            args.final_cases,
            "--final-manifest",
            args.final_manifest,
            "--output-dir",
            str(v2_dir),
            "--system-prompt-version",
            "v2",
            "--suite",
            "final",
        ],
    }
    result_files = {
        "prompt_v4": prompt_file,
        "lora_v1": v1_dir / "frozen_final_100.jsonl",
        "lora_v2": v2_dir / "frozen_final_100.jsonl",
    }
    results = {
        label: _run_or_resume(
            label=label,
            result_file=result_files[label],
            command=commands[label],
            final_cases=final_cases,
            resume=args.resume,
        )
        for label in MODEL_ORDER
    }
    summary = build_final_summary(
        results,
        frozen_sha256=frozen_hash,
        model_path=str(model_path),
        v1_adapter=args.lora_v1_adapter,
        v2_adapter=args.lora_v2_adapter,
    )
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n冻结最终测试结果：")
    for label in MODEL_ORDER:
        model_summary = summary["models"][label]
        correct = model_summary["num_cases"] - len(model_summary["failed_ids"])
        exact = model_summary["metrics"]["exact_match_accuracy"]
        print(f"{label}: {correct}/{model_summary['num_cases']} ({exact:.1%})")
    print(f"统一汇总：{summary_path}")
    print("最终测试已经解封；不得依据这些结果修改训练数据、提示词或当前模型。")


if __name__ == "__main__":
    main()
