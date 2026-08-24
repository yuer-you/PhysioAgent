from pathlib import Path

import pytest

from physioagent.workflow import (
    parse_workflow_plan,
    parse_workflow_plan_with_recovery,
    workflow_response_to_dict,
)
from physioagent.workflow_model import (
    ModelWorkflowAgent,
    ModelWorkflowPlanner,
    WORKFLOW_SYSTEM_PROMPT_V1,
    WORKFLOW_SYSTEM_PROMPT_V2,
    WORKFLOW_SYSTEM_PROMPT_V3,
)


REAL_ECG_207 = Path(__file__).parents[1] / "data" / "real" / "mitdb" / "207_30s" / "signal.csv"


class FakeGenerator:
    def __init__(self, output: str) -> None:
        self.output = output
        self.messages: list[dict[str, str]] | None = None

    def generate_messages(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.output


def test_parse_strict_two_step_workflow_plan():
    steps = parse_workflow_plan(
        '{"steps":['
        '{"name":"filter_signal","arguments":{"lowcut":0.5,"highcut":40.0}},'
        '{"name":"calculate_heart_rate","arguments":{}}]}'
    )
    assert [step.name for step in steps] == ["filter_signal", "calculate_heart_rate"]
    assert steps[0].arguments == {"lowcut": 0.5, "highcut": 40.0}


@pytest.mark.parametrize(
    "text, message",
    [
        ('```json\n{"steps":[]}\n```', "exactly one JSON"),
        ('{"steps":[],"reason":"done"}', "only the 'steps'"),
        ('{"steps":[]}', "non-empty"),
        ('{"steps":[{"name":"invented","arguments":{}}]}', "Invalid workflow step"),
        (
            '{"steps":[' + ",".join('{"name":"calculate_statistics","arguments":{}}' for _ in range(5)) + "]}",
            "exceeds max_steps=4",
        ),
    ],
)
def test_parse_workflow_plan_rejects_invalid_outputs(text, message):
    with pytest.raises(ValueError, match=message):
        parse_workflow_plan(text)


def test_conservative_recovery_inserts_only_missing_steps_bracket():
    raw = (
        '{"steps":[{"name":"load_signal","arguments":{}},'
        '{"name":"calculate_statistics","arguments":{}}}'
    )
    recovered = parse_workflow_plan_with_recovery(raw)
    assert recovered.recovery_applied is True
    assert recovered.recovery_type == "insert_missing_steps_closing_bracket"
    assert recovered.effective_text.endswith("]}")
    assert [step.name for step in recovered.steps] == ["load_signal", "calculate_statistics"]


def test_conservative_recovery_ignores_brackets_inside_string():
    raw = (
        '{"steps":[{"name":"load_signal","arguments":{"signal_column":"lead[0]"}},'
        '{"name":"calculate_statistics","arguments":{}}}'
    )
    recovered = parse_workflow_plan_with_recovery(raw)
    assert recovered.steps[0].arguments == {"signal_column": "lead[0]"}


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"steps":[]}\n```',
        '{"steps":[{"name":"calculate_statistics","arguments":{}}]',
        '{"steps":[{"name":"calculate_statistics","arguments":{"bad":1}}}',
        '{"steps":[{"name":"calculate_statistics","arguments":{"x":"unfinished}}}',
    ],
)
def test_conservative_recovery_refuses_ambiguous_or_semantic_errors(raw):
    with pytest.raises(ValueError):
        parse_workflow_plan_with_recovery(raw)


def test_model_workflow_planner_includes_profile_but_not_file_path():
    generator = FakeGenerator('{"steps":[{"name":"detect_peaks","arguments":{}}]}')
    planner = ModelWorkflowPlanner(generator)
    steps, _ = planner.generate_plan("检测 R 峰", signal_profile="ecg")
    assert steps[0].name == "detect_peaks"
    assert generator.messages is not None
    assert WORKFLOW_SYSTEM_PROMPT_V1 in generator.messages[0]["content"]
    assert "当前 signal_profile：ecg" in generator.messages[0]["content"]
    assert "data/real" not in generator.messages[0]["content"]


def test_workflow_prompt_v2_adds_default_column_boundary_rule():
    generator = FakeGenerator(
        '{"steps":[{"name":"load_signal","arguments":{}},'
        '{"name":"calculate_statistics","arguments":{}}]}'
    )
    planner = ModelWorkflowPlanner(generator, prompt_version="v2")
    planner.generate_plan("Read the default signal column and report statistics.", "ecg")
    assert generator.messages is not None
    system_prompt = generator.messages[0]["content"]
    assert WORKFLOW_SYSTEM_PROMPT_V2 in system_prompt
    assert "必须省略 signal_column" in system_prompt
    assert '"signal_column":""' in system_prompt


def test_model_workflow_planner_rejects_unknown_prompt_version():
    with pytest.raises(ValueError, match="prompt_version"):
        ModelWorkflowPlanner(FakeGenerator("{}"), prompt_version="v4")


def test_workflow_prompt_v3_adds_json_closure_check():
    generator = FakeGenerator(
        '{"steps":[{"name":"load_signal","arguments":{}},'
        '{"name":"calculate_statistics","arguments":{}}]}'
    )
    planner = ModelWorkflowPlanner(generator, prompt_version="v3")
    planner.generate_plan("读取默认列后计算统计量", "ecg")
    assert generator.messages is not None
    system_prompt = generator.messages[0]["content"]
    assert WORKFLOW_SYSTEM_PROMPT_V3 in system_prompt
    assert "最后两个字符恰好是 ]}" in system_prompt
    assert system_prompt.rstrip().endswith("当前 signal_profile：ecg")


def test_model_workflow_agent_executes_generated_plan():
    generator = FakeGenerator(
        '{"steps":['
        '{"name":"filter_signal","arguments":{"lowcut":0.5,"highcut":40.0}},'
        '{"name":"calculate_heart_rate","arguments":{}}]}'
    )
    response = ModelWorkflowAgent(generator).run(
        "先滤波再计算心率",
        REAL_ECG_207,
        sampling_rate=360,
        signal_profile="ecg",
    )
    assert response.planner == "model_zero_shot"
    assert response.raw_plan == generator.output
    assert response.trace[1].input_source == "step_1_filter_signal_output"
    assert response.final_result["num_peaks"] == 29
    assert response.final_result["mean_heart_rate_bpm"] == pytest.approx(56.84744806842749)


def test_model_workflow_agent_records_conservative_recovery():
    generator = FakeGenerator(
        '{"steps":[{"name":"load_signal","arguments":{}},'
        '{"name":"calculate_statistics","arguments":{}}}'
    )
    response = ModelWorkflowAgent(
        generator,
        prompt_version="v2",
        allow_recovery=True,
    ).run(
        "读取默认列后计算统计量",
        REAL_ECG_207,
        sampling_rate=360,
        signal_profile="ecg",
    )
    assert response.plan_recovery is not None
    assert response.plan_recovery["recovery_applied"] is True
    assert response.trace[-1].tool_name == "calculate_statistics"
    payload = workflow_response_to_dict(response, REAL_ECG_207, 360)
    assert payload["workflow"] == "model_workflow_v1"
    assert payload["plan_recovery"]["recovery_type"] == "insert_missing_steps_closing_bracket"
