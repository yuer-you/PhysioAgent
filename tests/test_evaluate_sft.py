import json
import hashlib

import pytest

from physioagent.evaluate_sft import (
    evaluate_cases,
    load_sft_test_cases,
    load_tool_calling_cases,
    summarize_results,
    validate_adapter_directory,
    verify_frozen_final_test,
)


class FakeGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate_messages(self, messages):
        assert [message["role"] for message in messages] == ["system", "user"]
        return next(self.outputs)


def _sft_row():
    return {
        "id": "sft_test_one",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "统计信号"},
        ],
        "completion": [
            {"role": "assistant", "content": '{"name":"calculate_statistics","arguments":{}}'}
        ],
        "metadata": {
            "tool_name": "calculate_statistics",
            "arguments": {},
            "language": "zh",
            "source": "test",
        },
    }


def test_load_sft_test_cases_converts_prompt_completion(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(_sft_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    cases = load_sft_test_cases(path)
    assert cases[0]["expected_name"] == "calculate_statistics"
    assert cases[0]["question"] == "统计信号"


def test_evaluate_cases_preserves_invalid_output():
    case = {
        "id": "one",
        "category": "statistics",
        "question": "统计信号",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "expected_name": "calculate_statistics",
        "expected_arguments": {},
    }
    results = evaluate_cases([case], FakeGenerator(["not json"]), "test")
    assert results[0]["valid_tool_call"] is False
    assert results[0]["exact_match"] is False
    assert "error" in results[0]
    assert summarize_results(results)["failed_ids"] == ["one"]


def test_load_tool_cases_uses_sft_system_prompt(tmp_path):
    path = tmp_path / "cases.jsonl"
    row = {
        "id": "one",
        "category": "statistics",
        "question": "stats",
        "expected_name": "calculate_statistics",
        "expected_arguments": {},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    case = load_tool_calling_cases(path)[0]
    assert case["messages"][0]["role"] == "system"
    assert case["messages"][1]["content"] == "stats"


def test_validate_adapter_directory_requires_weights(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weights"):
        validate_adapter_directory(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    validate_adapter_directory(tmp_path)


def test_verify_frozen_final_test_detects_changes(tmp_path):
    cases = tmp_path / "final.jsonl"
    manifest = tmp_path / "manifest.json"
    cases.write_text('{"id":"one"}\n', encoding="utf-8")
    digest = hashlib.sha256(cases.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps({"status": "frozen", "sha256": digest}), encoding="utf-8"
    )
    assert verify_frozen_final_test(cases, manifest) == digest
    cases.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_frozen_final_test(cases, manifest)
