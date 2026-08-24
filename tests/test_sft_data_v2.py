import json
from collections import Counter
from pathlib import Path

from physioagent.agent import parse_tool_call
from physioagent.sft_data_v2 import (
    SFT_SYSTEM_PROMPT_V2,
    generate_datasets_v2,
    validate_datasets_v2,
)


FINAL_CASES = Path(__file__).parents[1] / "evaluation" / "final_cases_v1.jsonl"


def test_v2_is_balanced_reproducible_and_valid():
    first = generate_datasets_v2()
    second = generate_datasets_v2()
    assert first == second
    assert {split: len(rows) for split, rows in first.items()} == {
        "train": 700,
        "validation": 150,
        "test": 100,
    }
    expected = {"train": 140, "validation": 30, "test": 20}
    for split, rows in first.items():
        assert Counter(row["metadata"]["tool_name"] for row in rows) == {
            name: expected[split]
            for name in (
                "calculate_statistics",
                "load_signal",
                "detect_peaks",
                "calculate_heart_rate",
                "filter_signal",
            )
        }
        assert all(row["prompt"][0]["content"] == SFT_SYSTEM_PROMPT_V2 for row in rows)
        for row in rows:
            parse_tool_call(row["completion"][0]["content"])
    validate_datasets_v2(first)


def test_v2_contains_targeted_schema_and_ordinal_examples():
    rows = generate_datasets_v2()["train"]
    targeted = [row for row in rows if row["metadata"]["source"] == "synthetic_targeted_v2"]
    text = "\n".join(row["prompt"][1]["content"] for row in targeted)
    assert "四阶" in text
    assert "third-order" in text
    assert any(row["metadata"]["arguments"].get("signal_column") for row in targeted)
    assert "cutoff1" not in "\n".join(row["completion"][0]["content"] for row in targeted)


def test_v2_questions_do_not_overlap_frozen_final_test():
    final_questions = {
        json.loads(line)["question"].strip().casefold()
        for line in FINAL_CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    v2_questions = {
        row["prompt"][1]["content"].strip().casefold()
        for rows in generate_datasets_v2().values()
        for row in rows
    }
    assert final_questions.isdisjoint(v2_questions)
