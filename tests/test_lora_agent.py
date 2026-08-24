from pathlib import Path

import pytest

from physioagent.lora_agent import LoRAAgent
from physioagent.sft_data_v2 import SFT_SYSTEM_PROMPT_V2


DATA = Path(__file__).parents[1] / "data" / "sample_ecg.csv"
REAL_ECG_207 = Path(__file__).parents[1] / "data" / "real" / "mitdb" / "207_30s" / "signal.csv"


class FakeGenerator:
    """不加载模型，只返回预先设定的模型输出。

    Return a predefined model output without loading a model.
    """

    def __init__(self, output: str) -> None:
        self.output = output
        self.messages: list[dict[str, str]] | None = None

    def generate_messages(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.output


def test_lora_agent_executes_model_tool_call():
    generator = FakeGenerator('{"name":"calculate_heart_rate","arguments":{}}')
    response = LoRAAgent(generator=generator).run("计算平均心率", DATA, sampling_rate=25)

    assert generator.messages == [
        {"role": "system", "content": SFT_SYSTEM_PROMPT_V2},
        {"role": "user", "content": "计算平均心率"},
    ]
    assert response.raw_decision == '{"name":"calculate_heart_rate","arguments":{}}'
    assert response.tool_name == "calculate_heart_rate"
    assert response.tool_result["mean_heart_rate_bpm"] == pytest.approx(75.0)
    assert "75.0 BPM" in response.answer


def test_lora_agent_preserves_explicit_arguments():
    generator = FakeGenerator('{"name":"detect_peaks","arguments":{"prominence":0.5}}')
    response = LoRAAgent(generator=generator).run("突出度 0.5", DATA, sampling_rate=25)
    assert response.tool_arguments == {"prominence": 0.5}


def test_lora_agent_executes_ecg_profile_without_giving_profile_to_model():
    generator = FakeGenerator('{"name":"calculate_heart_rate","arguments":{}}')
    response = LoRAAgent(generator=generator).run(
        "计算这段 ECG 的平均心率",
        REAL_ECG_207,
        sampling_rate=360,
        signal_profile="ecg",
    )
    assert generator.messages[-1]["content"] == "计算这段 ECG 的平均心率"
    assert response.signal_profile == "ecg"
    assert response.tool_result["detector"] == "ecg_detector_v1"
    assert response.tool_result["mean_heart_rate_bpm"] == pytest.approx(56.84744806842749)


def test_lora_agent_rejects_invalid_model_output_before_execution():
    generator = FakeGenerator('{"name":"invented_tool","arguments":{}}')
    with pytest.raises(ValueError, match="Unknown tool"):
        LoRAAgent(generator=generator).run("随便分析", DATA, sampling_rate=25)


def test_lora_agent_requires_model_paths_without_generator():
    with pytest.raises(ValueError, match="model_path and adapter_path"):
        LoRAAgent()
