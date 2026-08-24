import json

import pytest

from physioagent.train_sft import (
    build_parser,
    file_sha256,
    load_jsonl,
    validate_sft_rows,
    validate_training_files,
)


def _row(row_id, question):
    return {
        "id": row_id,
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": question},
        ],
        "completion": [
            {"role": "assistant", "content": '{"name":"calculate_statistics","arguments":{}}'}
        ],
        "metadata": {"tool_name": "calculate_statistics", "arguments": {}},
    }


def test_validate_sft_rows_accepts_prompt_completion():
    assert validate_sft_rows([_row("one", "统计信号")], "train") == {"统计信号"}


def test_validate_sft_rows_accepts_workflow_completion():
    row = {
        "id": "workflow-one",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "先滤波再统计"},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": (
                    '{"steps":[{"name":"filter_signal","arguments":{}},'
                    '{"name":"calculate_statistics","arguments":{}}]}'
                ),
            }
        ],
        "metadata": {
            "task_type": "workflow",
            "expected_steps": [
                {"name": "filter_signal", "arguments": {}},
                {"name": "calculate_statistics", "arguments": {}},
            ],
        },
    }
    assert validate_sft_rows([row], "train") == {"先滤波再统计"}


def test_validate_sft_rows_rejects_bad_completion():
    row = _row("one", "统计信号")
    row["completion"][0]["content"] = '{"name":"invented_tool","arguments":{}}'
    with pytest.raises(ValueError):
        validate_sft_rows([row], "train")


def test_validate_training_files_rejects_leakage(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    text = json.dumps(_row("train-one", "same question"), ensure_ascii=False) + "\n"
    train.write_text(text, encoding="utf-8")
    validation.write_text(
        json.dumps(_row("validation-one", "same question"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leakage"):
        validate_training_files(train, validation)


def test_load_jsonl_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "missing.jsonl")


def test_file_sha256_is_reproducible(tmp_path):
    path = tmp_path / "data.txt"
    path.write_bytes(b"physioagent")
    assert file_sha256(path) == "008c2e8ea771efbd26f245b1eac5ccb3151da9e104a0c0089c86337f4c18bc66"


def test_train_parser_supports_token_length_inspection():
    args = build_parser().parse_args(["--dry-run", "--inspect-token-lengths"])
    assert args.dry_run is True
    assert args.inspect_token_lengths is True
