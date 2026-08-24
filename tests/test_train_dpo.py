import json

import pytest

from physioagent.train_dpo import (
    _validate_cli,
    build_parser,
    load_jsonl,
    validate_dpo_rows,
    validate_dpo_training_files,
)


def _row(row_id: str, question: str) -> dict:
    chosen_steps = [
        {"name": "load_signal", "arguments": {}},
        {"name": "calculate_statistics", "arguments": {}},
    ]
    rejected_steps = [chosen_steps[-1]]
    compact = lambda steps: json.dumps({"steps": steps}, separators=(",", ":"))
    return {
        "id": row_id,
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": question},
        ],
        "chosen": [{"role": "assistant", "content": compact(chosen_steps)}],
        "rejected": [{"role": "assistant", "content": compact(rejected_steps)}],
        "metadata": {
            "task_type": "workflow_preference",
            "split": "train",
            "preference_type": "omit_load",
            "chosen_steps": chosen_steps,
            "rejected_steps": rejected_steps,
        },
    }


def test_validate_dpo_rows_accepts_valid_pair():
    assert validate_dpo_rows([_row("one", "question")], "train") == {"question"}


def test_validate_dpo_rows_rejects_identical_pair():
    row = _row("one", "question")
    row["rejected"] = row["chosen"]
    row["metadata"]["rejected_steps"] = row["metadata"]["chosen_steps"]
    with pytest.raises(ValueError, match="identical"):
        validate_dpo_rows([row], "train")


def test_validate_dpo_training_files_rejects_question_leakage(tmp_path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(json.dumps(_row("train", "same")) + "\n", encoding="utf-8")
    validation_row = _row("validation", "same")
    validation_row["metadata"]["split"] = "validation"
    validation.write_text(json.dumps(validation_row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        validate_dpo_training_files(train, validation)


def test_load_jsonl_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "missing.jsonl")


def test_dpo_parser_defaults_and_length_guard():
    args = build_parser().parse_args(["--dry-run", "--inspect-token-lengths"])
    _validate_cli(args)
    assert args.epochs == 1.0
    assert args.learning_rate == 5e-6
    assert args.beta == 0.1
    assert args.max_prompt_length == 896
    assert args.max_completion_length == 128
    assert args.max_length == 1024

    invalid = build_parser().parse_args(["--max-prompt-length", "900", "--max-completion-length", "128"])
    with pytest.raises(ValueError, match="must not exceed"):
        _validate_cli(invalid)
