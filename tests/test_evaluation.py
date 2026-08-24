import json
from collections import Counter
from pathlib import Path

from physioagent.agent import ToolCall, parse_tool_call
from physioagent.evaluate_baseline import calculate_metrics, load_cases, score_call


CASES = Path(__file__).parents[1] / "evaluation" / "tool_calling_cases.jsonl"
FINAL_CASES = Path(__file__).parents[1] / "evaluation" / "final_cases_v1.jsonl"


def test_expanded_evaluation_set_is_balanced_and_valid():
    cases = load_cases(CASES)
    assert len(cases) == 60
    assert Counter(case["category"] for case in cases) == {
        "statistics": 12,
        "load": 12,
        "peaks": 12,
        "heart_rate": 12,
        "filter": 12,
    }
    for case in cases:
        text = json.dumps(
            {"name": case["expected_name"], "arguments": case["expected_arguments"]}
        )
        parse_tool_call(text)


def test_frozen_final_set_is_balanced_unique_and_valid():
    cases = load_cases(FINAL_CASES)
    assert len(cases) == 100
    assert Counter(case["category"] for case in cases) == {
        "statistics": 20,
        "load": 20,
        "peaks": 20,
        "heart_rate": 20,
        "filter": 20,
    }
    for case in cases:
        parse_tool_call(
            json.dumps(
                {"name": case["expected_name"], "arguments": case["expected_arguments"]}
            )
        )


def test_scoring_separates_requested_and_extra_arguments():
    case = {
        "expected_name": "filter_signal",
        "expected_arguments": {"lowcut": 0.5, "highcut": 8.0},
    }
    score = score_call(
        case,
        ToolCall("filter_signal", {"lowcut": 0.5, "highcut": 8.0, "order": 4}),
    )
    assert score["requested_arguments_correct"] is True
    assert score["extra_arguments"] == ["order"]
    assert score["arguments_exact"] is False


def test_metrics_report_extra_argument_rate():
    results = [
        {
            "valid_tool_call": True,
            "name_correct": True,
            "requested_arguments_correct": True,
            "has_extra_arguments": False,
            "arguments_exact": True,
            "exact_match": True,
        },
        {
            "valid_tool_call": True,
            "name_correct": True,
            "requested_arguments_correct": True,
            "has_extra_arguments": True,
            "arguments_exact": False,
            "exact_match": False,
        },
    ]
    metrics = calculate_metrics(results)
    assert metrics["requested_arguments_accuracy"] == 1.0
    assert metrics["extra_argument_case_rate"] == 0.5
    assert metrics["exact_match_accuracy"] == 0.5
