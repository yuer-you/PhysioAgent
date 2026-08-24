import json

from physioagent.evaluate_workflow_sft_validation import (
    evaluate_validation_cases,
    load_workflow_validation_cases,
    summarize_validation_results,
)


class FakeGenerator:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.last_generation_info = None

    def generate_messages(self, messages):
        assert [message["role"] for message in messages] == ["system", "user"]
        output = next(self.outputs)
        self.last_generation_info = {
            "num_generated_tokens": 10,
            "max_new_tokens": 128,
            "reached_max_new_tokens": False,
        }
        return output


def _row(row_id="one"):
    steps = [
        {"name": "load_signal", "arguments": {}},
        {"name": "calculate_statistics", "arguments": {}},
    ]
    return {
        "id": row_id,
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "use the default field, then summarize it"},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps({"steps": steps}, separators=(",", ":")),
            }
        ],
        "metadata": {
            "task_type": "workflow",
            "category": "load_then_statistics",
            "step_count": 2,
            "load_policy": "load_default",
            "language": "en",
            "column_kind": "default",
            "paraphrase_split": "validation_only",
            "expected_steps": steps,
        },
    }


def test_load_workflow_validation_cases_checks_completion(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    cases = load_workflow_validation_cases(path)
    assert cases[0]["load_policy"] == "load_default"
    assert cases[0]["step_count"] == 2


def test_generation_evaluation_preserves_errors_and_group_metrics(tmp_path):
    path = tmp_path / "validation.jsonl"
    rows = [_row("correct"), _row("invalid")]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    correct = rows[0]["completion"][0]["content"]
    results = evaluate_validation_cases(
        load_workflow_validation_cases(path),
        FakeGenerator([correct, "not json"]),
    )
    summary = summarize_validation_results(results)
    assert results[0]["plan_exact"] is True
    assert results[0]["generation_info"]["num_generated_tokens"] == 10
    assert results[1]["error_type"] == "invalid_plan"
    assert summary["metrics"] == {"valid_plan_rate": 0.5, "plan_exact_rate": 0.5}
    assert summary["by_load_policy"]["load_default"]["correct"] == 1
    assert summary["error_types"] == {"invalid_plan": 1}


def test_dry_run_uses_expected_completions(tmp_path):
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    results = evaluate_validation_cases(load_workflow_validation_cases(path), dry_run=True)
    assert results[0]["valid_plan"] is True
    assert results[0]["plan_exact"] is True


def test_validation_parser_accepts_dpo_mode_label():
    from physioagent.evaluate_workflow_sft_validation import build_parser

    args = build_parser().parse_args(["--mode-label", "workflow_dpo_v1"])
    assert args.mode_label == "workflow_dpo_v1"
