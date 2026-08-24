from pathlib import Path

import pytest

from physioagent.agent import QwenAgent, RuleBasedAgent, ToolCall, ToolExecutor, parse_tool_call


DATA = Path(__file__).parents[1] / "data" / "sample_ecg.csv"
REAL_ECG_207 = Path(__file__).parents[1] / "data" / "real" / "mitdb" / "207_30s" / "signal.csv"


def test_agent_routes_heart_rate_question():
    response = RuleBasedAgent().run("这段信号心率是多少？", DATA, sampling_rate=25)
    assert response.tool_name == "calculate_heart_rate"
    assert "75.0 BPM" in response.answer


def test_agent_defaults_to_statistics():
    response = RuleBasedAgent().run("请概述该信号。", DATA, sampling_rate=25)
    assert response.tool_name == "calculate_statistics"


def test_tool_executor_uses_frozen_ecg_profile_on_real_record():
    response = ToolExecutor().run(
        ToolCall("calculate_heart_rate"),
        REAL_ECG_207,
        sampling_rate=360,
        signal_profile="ecg",
    )
    assert response.signal_profile == "ecg"
    assert response.tool_result["detector"] == "ecg_detector_v1"
    assert response.tool_result["num_peaks"] == 29
    assert response.tool_result["mean_heart_rate_bpm"] == pytest.approx(56.84744806842749)


def test_ecg_profile_rejects_model_generated_generic_peak_parameters():
    with pytest.raises(ValueError, match="frozen ecg_detector_v1"):
        ToolExecutor().run(
            ToolCall("detect_peaks", {"prominence": 0.2}),
            REAL_ECG_207,
            sampling_rate=360,
            signal_profile="ecg",
        )


def test_tool_executor_rejects_unknown_signal_profile():
    with pytest.raises(ValueError, match="signal_profile"):
        ToolExecutor().run(
            ToolCall("calculate_statistics"),
            DATA,
            sampling_rate=25,
            signal_profile="ppg",
        )


def test_parse_qwen_json_tool_call():
    call = parse_tool_call('{"name":"filter_signal","arguments":{"lowcut":0.5,"highcut":8.0}}')
    assert call.name == "filter_signal"
    assert call.arguments == {"lowcut": 0.5, "highcut": 8.0}


def test_parse_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        parse_tool_call('{"name":"diagnose_patient","arguments":{}}')


def test_parse_rejects_invented_argument():
    with pytest.raises(ValueError, match="Unexpected arguments"):
        parse_tool_call('{"name":"calculate_statistics","arguments":{"patient_id":1}}')


def test_parse_rejects_markdown_or_trailing_text():
    with pytest.raises(ValueError, match="exactly one JSON object"):
        parse_tool_call('```json\n{"name":"calculate_statistics","arguments":{}}\n```')
    with pytest.raises(ValueError, match="exactly one JSON object"):
        parse_tool_call(
            '{"name":"detect_peaks","arguments":{}} '
            '{"name":"calculate_heart_rate","arguments":{}}'
        )


def test_parse_rejects_top_level_extra_or_missing_fields():
    with pytest.raises(ValueError, match="only name and arguments"):
        parse_tool_call('{"name":"load_signal","signal_column":"ecg"}')
    with pytest.raises(ValueError, match="only name and arguments"):
        parse_tool_call('{"name":"load_signal","arguments":{},"reason":"needed"}')


def test_parse_rejects_arguments_array():
    with pytest.raises(ValueError, match="arguments must be a JSON object"):
        parse_tool_call('{"name":"calculate_statistics","arguments":[]}')


@pytest.mark.parametrize(
    "payload",
    [
        '{"name":"load_signal","arguments":{"signal_column":""}}',
        '{"name":"load_signal","arguments":{"signal_column":"   "}}',
    ],
)
def test_parse_rejects_empty_signal_column(payload):
    with pytest.raises(ValueError, match="must not be empty"):
        parse_tool_call(payload)


def test_prompt_v4_uses_extraction_rules_and_error_examples():
    agent = QwenAgent.__new__(QwenAgent)
    agent.prompt_version = "v4"
    prompt = agent._build_messages("计算心率")[0]["content"]
    assert "原文没有数字或数词" in prompt
    assert "禁止填默认值、猜测值或 null" in prompt
    assert "允许的工具名只有以下五个" in prompt
    assert '正确：{"name":"calculate_heart_rate","arguments":{}}' in prompt
