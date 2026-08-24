import json
from pathlib import Path

from physioagent.final_demo import (
    DEFAULT_ADAPTER_PATH,
    build_demo_payload,
    build_parser,
    run_final_workflow,
    write_demo_payload,
)


ROOT = Path(__file__).parents[1]
SIGNAL = ROOT / "data" / "real" / "mitdb" / "207_30s" / "signal.csv"


class FakeGenerator:
    def generate_messages(self, messages):
        assert [message["role"] for message in messages] == ["system", "user"]
        return (
            '{"steps":['
            '{"name":"load_signal","arguments":{"signal_column":"signal"}},'
            '{"name":"filter_signal","arguments":{"lowcut":0.5,"highcut":40.0}},'
            '{"name":"calculate_heart_rate","arguments":{}}]}'
        )


def test_final_demo_defaults_to_dpo_adapter_and_project_output():
    args = build_parser().parse_args([])
    assert args.adapter_path == DEFAULT_ADAPTER_PATH
    assert args.output.startswith("outputs/demo/")
    assert args.max_new_tokens == 128


def test_final_demo_runs_three_step_real_ecg_and_writes_trace(tmp_path):
    response = run_final_workflow(
        FakeGenerator(),
        question="load, filter, then calculate heart rate",
        signal_file=SIGNAL,
        sampling_rate=360.0,
        signal_profile="ecg",
    )
    assert [item.tool_name for item in response.trace] == [
        "load_signal",
        "filter_signal",
        "calculate_heart_rate",
    ]
    assert response.stop_reason == "plan_completed"
    assert response.final_result["num_peaks"] == 29
    payload = build_demo_payload(
        response,
        model_path="model",
        adapter_path="adapter",
        signal_file=SIGNAL,
        sampling_rate=360.0,
    )
    output = write_demo_payload(tmp_path / "demo.json", payload)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["clinical_use"] is False
    assert len(saved["workflow"]["trace"]) == 3
