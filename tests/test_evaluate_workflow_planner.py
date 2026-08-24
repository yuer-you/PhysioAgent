from collections import Counter
from pathlib import Path

from physioagent.evaluate_workflow_planner import (
    build_parser,
    default_output_dir,
    evaluate_workflow_case,
    load_workflow_cases,
    summarize_workflow_results,
)
from physioagent.workflow_model import ModelWorkflowPlanner


CASES = Path(__file__).parents[1] / "evaluation" / "workflow_planning_cases_v1.jsonl"


class FakeGenerator:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate_messages(self, messages: list[dict[str, str]]) -> str:
        return self.output


def test_workflow_prompt_version_has_separate_default_output_directory():
    assert default_output_dir("v1").endswith("qwen_zero_shot_planning_v1")
    assert default_output_dir("v2").endswith("qwen_zero_shot_planning_v2")
    args = build_parser().parse_args(["--workflow-prompt-version", "v2"])
    assert args.workflow_prompt_version == "v2"
    assert args.output_dir is None
    assert args.max_new_tokens == 128
    assert args.adapter_path is None


def test_workflow_evaluator_accepts_adapter_path():
    args = build_parser().parse_args(
        ["--adapter-path", "outputs/sft/workflow/final_adapter", "--mode-label", "workflow_dpo_v1"]
    )
    assert args.adapter_path == "outputs/sft/workflow/final_adapter"
    assert args.mode_label == "workflow_dpo_v1"


def test_workflow_evaluator_records_optional_generation_info():
    case = next(case for case in load_workflow_cases(CASES) if case["id"] == "wf_single_001")
    generator = FakeGenerator('{"steps":[{"name":"calculate_heart_rate","arguments":{}}]}')
    generator.last_generation_info = {
        "num_generated_tokens": 42,
        "max_new_tokens": 128,
        "reached_max_new_tokens": False,
    }
    row = evaluate_workflow_case(case, planner=ModelWorkflowPlanner(generator))
    assert row["generation_info"] == generator.last_generation_info


def test_workflow_planning_cases_have_expected_coverage():
    cases = load_workflow_cases(CASES)
    assert len(cases) == 20
    assert Counter(case["category"] for case in cases) == {
        "filter_then_heart_rate": 6,
        "filter_then_peaks": 4,
        "filter_then_statistics": 4,
        "load_then_statistics": 3,
        "single_step": 3,
    }


def test_workflow_planning_dry_run_is_fully_executable():
    results = [evaluate_workflow_case(case, dry_run=True) for case in load_workflow_cases(CASES)]
    summary = summarize_workflow_results(results)
    assert summary["metrics"]["valid_plan_rate"] == 1.0
    assert summary["metrics"]["plan_exact_rate"] == 1.0
    assert summary["metrics"]["end_to_end_success_rate"] == 1.0
    assert summary["failed_ids"] == []


def test_workflow_evaluator_executes_correct_fake_plan():
    case = next(case for case in load_workflow_cases(CASES) if case["id"] == "wf_filter_hr_001")
    output = (
        '{"steps":['
        '{"name":"filter_signal","arguments":{"lowcut":0.5,"highcut":40.0}},'
        '{"name":"calculate_heart_rate","arguments":{}}]}'
    )
    row = evaluate_workflow_case(case, planner=ModelWorkflowPlanner(FakeGenerator(output)))
    assert row["plan_exact"] is True
    assert row["execution_success"] is True
    assert row["reference_check_passed"] is True
    assert row["end_to_end_success"] is True


def test_workflow_evaluator_records_parse_failure():
    case = load_workflow_cases(CASES)[0]
    row = evaluate_workflow_case(
        case,
        planner=ModelWorkflowPlanner(FakeGenerator("not json")),
    )
    assert row["valid_plan"] is False
    assert row["error_stage"] == "parse"
    assert row["end_to_end_success"] is False
