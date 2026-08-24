from collections import Counter
from pathlib import Path

from physioagent.evaluate_real_agent import (
    evaluate_case,
    load_real_agent_cases,
    summarize_results,
)


CASES = Path(__file__).parents[1] / "evaluation" / "real_agent_cases_v1.jsonl"


class FakeAgent:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate_decision(self, question: str) -> str:
        return self.output


def test_real_agent_cases_cover_five_tools_and_four_records():
    cases = load_real_agent_cases(CASES)
    assert len(cases) == 20
    assert Counter(case["category"] for case in cases) == {
        "load_signal": 4,
        "calculate_statistics": 4,
        "detect_peaks": 4,
        "calculate_heart_rate": 4,
        "filter_signal": 4,
    }
    assert {case["record"] for case in cases} == {"100", "101", "200", "207"}


def test_real_agent_dry_run_executes_all_expected_calls():
    results = [evaluate_case(case, dry_run=True) for case in load_real_agent_cases(CASES)]
    summary = summarize_results(results)
    assert summary["metrics"]["end_to_end_success_rate"] == 1.0
    assert summary["failed_ids"] == []
    # 数组结果必须是摘要，避免评测文件意外膨胀。
    # Array results must be summarized to prevent evaluation artifacts from growing unexpectedly.
    load_result = next(row for row in results if row["category"] == "load_signal")
    assert load_result["tool_result_summary"]["type"] == "ndarray"
    assert "first_five" in load_result["tool_result_summary"]


def test_real_agent_case_scores_a_valid_fake_model_decision():
    case = next(
        case
        for case in load_real_agent_cases(CASES)
        if case["id"] == "real_hr_001"
    )
    row = evaluate_case(
        case,
        agent=FakeAgent('{"name":"calculate_heart_rate","arguments":{}}'),  # type: ignore[arg-type]
    )
    assert row["tool_call_exact"] is True
    assert row["execution_success"] is True
    assert row["reference_check_passed"] is True
    assert row["answer_grounded"] is True
    assert row["end_to_end_success"] is True


def test_real_agent_case_keeps_wrong_tool_failure_visible():
    case = next(
        case
        for case in load_real_agent_cases(CASES)
        if case["id"] == "real_hr_001"
    )
    row = evaluate_case(
        case,
        agent=FakeAgent('{"name":"calculate_statistics","arguments":{}}'),  # type: ignore[arg-type]
    )
    assert row["valid_tool_call"] is True
    assert row["tool_call_exact"] is False
    assert row["execution_success"] is True
    assert row["reference_check_passed"] is False
    assert row["end_to_end_success"] is False
