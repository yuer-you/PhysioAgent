import json
from collections import Counter
from pathlib import Path

from physioagent.sft_data_v2_1 import generate_datasets_v2_1, validate_datasets_v2_1


FINAL_CASES = Path(__file__).parents[1] / "evaluation" / "final_cases_v1.jsonl"


def test_v2_1_is_reproducible_and_has_expected_sizes():
    first = generate_datasets_v2_1()
    second = generate_datasets_v2_1()
    assert first == second
    assert {split: len(rows) for split, rows in first.items()} == {
        "train": 760,
        "validation": 170,
        "test": 100,
    }
    validate_datasets_v2_1(first)


def test_v2_1_additions_target_only_the_two_exposed_error_types():
    data = generate_datasets_v2_1()
    additions = {
        split: [row for row in data[split] if row["metadata"]["source"] == "synthetic_targeted_v2_1"]
        for split in ("train", "validation")
    }
    assert Counter(row["metadata"]["tool_name"] for row in additions["train"]) == {
        "calculate_statistics": 30,
        "filter_signal": 30,
    }
    assert Counter(row["metadata"]["tool_name"] for row in additions["validation"]) == {
        "calculate_statistics": 10,
        "filter_signal": 10,
    }
    for rows in additions.values():
        for row in rows:
            if row["metadata"]["tool_name"] == "calculate_statistics":
                assert row["metadata"]["arguments"] == {}
            else:
                assert set(row["metadata"]["arguments"]) == {"order"}


def test_v2_1_does_not_copy_failed_development_questions_or_final_cases():
    data = generate_datasets_v2_1()
    questions = {
        row["prompt"][1]["content"].strip().casefold()
        for rows in data.values()
        for row in rows
    }
    exposed_failures = {
        "Without modifying the waveform, provide its sample count and duration.",
        "Without modifying the waveform, provide its sample count and duration. Leave the original signal unchanged.",
        "Apply a sixth-order band-pass filter.",
    }
    # 这三条仍只存在于继承的开发 test，不得出现在 train/validation。
    train_validation_questions = {
        row["prompt"][1]["content"].strip().casefold()
        for split in ("train", "validation")
        for row in data[split]
    }
    assert train_validation_questions.isdisjoint({question.casefold() for question in exposed_failures})
    final_questions = {
        json.loads(line)["question"].strip().casefold()
        for line in FINAL_CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert questions.isdisjoint(final_questions)
