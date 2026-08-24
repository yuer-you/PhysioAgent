"""离线评测已保存的模型计划：同时报告严格成绩与保守恢复后的工程成绩。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluate_workflow_planner import (
    DEFAULT_CASES_FILE,
    load_workflow_cases,
    steps_as_dicts,
    validate_workflow_result,
    workflow_answer_is_grounded,
)
from .workflow import (
    WorkflowExecutor,
    parse_workflow_plan_with_recovery,
    workflow_response_to_dict,
)


DEFAULT_SOURCE_RESULTS = (
    "outputs/workflow/qwen_zero_shot_planning_v2/base_qwen/results.jsonl"
)
DEFAULT_OUTPUT_DIR = "outputs/workflow/qwen_zero_shot_planning_v2_recovery"


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("No saved workflow results found.")
    return rows


def evaluate_recovery_case(
    case: dict[str, Any],
    source_row: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "raw_plan": source_row["raw_plan"],
        "strict_valid_plan": bool(source_row["valid_plan"]),
        "strict_plan_exact": bool(source_row["plan_exact"]),
        "strict_end_to_end_success": bool(source_row["end_to_end_success"]),
        "recovery_applied": False,
        "effective_plan_valid": False,
        "effective_plan_exact": False,
        "effective_execution_success": False,
        "effective_reference_check_passed": False,
        "effective_answer_grounded": False,
        "operational_success": False,
    }
    try:
        recovered = parse_workflow_plan_with_recovery(source_row["raw_plan"])
    except ValueError as error:
        row["recovery_error"] = str(error)
        return row

    row.update(
        {
            "recovery_applied": recovered.recovery_applied,
            "recovery_type": recovered.recovery_type,
            "strict_parse_error": recovered.strict_error,
            "effective_plan_text": recovered.effective_text,
            "effective_steps": steps_as_dicts(recovered.steps),
            "effective_plan_valid": True,
        }
    )
    row["effective_plan_exact"] = row["effective_steps"] == case["expected_steps"]

    reference = json.loads(Path(case["reference_file"]).read_text(encoding="utf-8"))
    sampling_rate = float(reference["sampling_rate_hz"])
    try:
        response = WorkflowExecutor().run(
            case["question"],
            recovered.steps,
            case["signal_file"],
            sampling_rate,
            case["signal_profile"],
        )
        response.planner = "model_zero_shot_v2_recovered"
        response.raw_plan = source_row["raw_plan"]
        response.plan_recovery = {
            "recovery_applied": recovered.recovery_applied,
            "recovery_type": recovered.recovery_type,
            "strict_error": recovered.strict_error,
            "effective_text": recovered.effective_text,
        }
        row["effective_execution_success"] = True
        row["workflow_trace"] = workflow_response_to_dict(
            response,
            case["signal_file"],
            sampling_rate,
        )
        check = validate_workflow_result(case, response, reference)
        row["effective_reference_check"] = check
        row["effective_reference_check_passed"] = bool(check["passed"])
        row["effective_answer_grounded"] = workflow_answer_is_grounded(response)
    except (FileNotFoundError, ValueError) as error:
        row["execution_error"] = str(error)
        return row

    row["operational_success"] = all(
        row[name]
        for name in (
            "effective_plan_exact",
            "effective_execution_success",
            "effective_reference_check_passed",
            "effective_answer_grounded",
        )
    )
    return row


def summarize_recovery_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    repaired = [row for row in rows if row["recovery_applied"]]
    return {
        "num_cases": total,
        "strict_valid_plan_rate": sum(row["strict_valid_plan"] for row in rows) / total,
        "strict_end_to_end_success_rate": sum(
            row["strict_end_to_end_success"] for row in rows
        )
        / total,
        "recovery_applied_count": len(repaired),
        "repaired_ids": [row["id"] for row in repaired],
        "effective_plan_valid_rate": sum(row["effective_plan_valid"] for row in rows) / total,
        "operational_success_rate": sum(row["operational_success"] for row in rows) / total,
        "operational_failed_ids": [
            row["id"] for row in rows if not row["operational_success"]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--cases-file", default=DEFAULT_CASES_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_workflow_cases(args.cases_file)
    source_rows = load_result_rows(args.source_results)
    source_by_id = {row["id"]: row for row in source_rows}
    if set(source_by_id) != {case["id"] for case in cases}:
        raise ValueError("Source result ids do not exactly match evaluation case ids.")

    rows = [evaluate_recovery_case(case, source_by_id[case["id"]]) for case in cases]
    summary = {
        "evaluation": "workflow_recovery_v1",
        "source_results": args.source_results,
        "cases_file": args.cases_file,
        **summarize_recovery_results(rows),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"严格端到端成功：{sum(row['strict_end_to_end_success'] for row in rows)}/{len(rows)}"
    )
    print(f"应用保守恢复：{sum(row['recovery_applied'] for row in rows)}/{len(rows)}")
    print(f"恢复后工程成功：{sum(row['operational_success'] for row in rows)}/{len(rows)}")
    print(f"完整结果：{output_dir}")


if __name__ == "__main__":
    main()
