from pathlib import Path

from physioagent.evaluate_workflow_planner import load_workflow_cases
from physioagent.evaluate_workflow_recovery import (
    evaluate_recovery_case,
    summarize_recovery_results,
)


CASES = Path(__file__).parents[1] / "evaluation" / "workflow_planning_cases_v1.jsonl"


def test_offline_recovery_executes_missing_bracket_plan():
    case = next(
        case for case in load_workflow_cases(CASES) if case["id"] == "wf_load_stats_002"
    )
    source_row = {
        "id": case["id"],
        "raw_plan": (
            '{"steps":[{"name":"load_signal","arguments":{}},'
            '{"name":"calculate_statistics","arguments":{}}}'
        ),
        "valid_plan": False,
        "plan_exact": False,
        "end_to_end_success": False,
    }
    row = evaluate_recovery_case(case, source_row)
    assert row["strict_end_to_end_success"] is False
    assert row["recovery_applied"] is True
    assert row["effective_plan_exact"] is True
    assert row["effective_execution_success"] is True
    assert row["effective_reference_check_passed"] is True
    assert row["operational_success"] is True


def test_recovery_summary_keeps_strict_and_operational_metrics_separate():
    rows = [
        {
            "id": "strict_ok",
            "strict_valid_plan": True,
            "strict_end_to_end_success": True,
            "recovery_applied": False,
            "effective_plan_valid": True,
            "operational_success": True,
        },
        {
            "id": "repaired_ok",
            "strict_valid_plan": False,
            "strict_end_to_end_success": False,
            "recovery_applied": True,
            "effective_plan_valid": True,
            "operational_success": True,
        },
    ]
    summary = summarize_recovery_results(rows)
    assert summary["strict_end_to_end_success_rate"] == 0.5
    assert summary["recovery_applied_count"] == 1
    assert summary["operational_success_rate"] == 1.0
