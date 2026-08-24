import json
from collections import Counter
from pathlib import Path

from physioagent.sft_workflow_data import (
    CATEGORY_COUNTS,
    WORKFLOW_SFT_SYSTEM_PROMPT,
    generate_workflow_sft_datasets,
    validate_workflow_sft_datasets,
    write_workflow_sft_datasets,
)
from physioagent.train_sft import validate_sft_rows
from physioagent.workflow import parse_workflow_plan


ROOT = Path(__file__).parents[1]


def _questions(path: Path) -> set[str]:
    return {
        json.loads(line)["question"].strip().casefold()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_workflow_sft_is_reproducible_valid_and_has_expected_sizes():
    first = generate_workflow_sft_datasets()
    second = generate_workflow_sft_datasets()
    assert first == second
    assert {split: len(rows) for split, rows in first.items()} == {
        "train": 1200,
        "validation": 200,
    }
    for split, rows in first.items():
        assert Counter(row["metadata"]["category"] for row in rows) == CATEGORY_COUNTS[split]
        assert all(row["prompt"][0]["content"] == WORKFLOW_SFT_SYSTEM_PROMPT for row in rows)
        assert all(row["metadata"]["task_type"] == "workflow" for row in rows)
        for row in rows:
            parse_workflow_plan(row["completion"][0]["content"])
        validate_sft_rows(rows, split)
    validate_workflow_sft_datasets(first)


def test_workflow_sft_focuses_on_load_and_three_step_plans():
    rows = generate_workflow_sft_datasets()["train"]
    with_load = [
        row
        for row in rows
        if any(step["name"] == "load_signal" for step in row["metadata"]["expected_steps"])
    ]
    three_step = [row for row in rows if row["metadata"]["step_count"] == 3]
    default_load = [
        row
        for row in with_load
        if any(step["name"] == "load_signal" and step["arguments"] == {} for step in row["metadata"]["expected_steps"])
    ]
    explicit_load = [
        row
        for row in with_load
        if any(step["name"] == "load_signal" and step["arguments"] for step in row["metadata"]["expected_steps"])
    ]
    assert len(with_load) >= 650
    assert len(three_step) == 520
    assert default_load and explicit_load


def test_workflow_sft_does_not_copy_development_or_frozen_questions():
    data_questions = {
        row["prompt"][1]["content"].strip().casefold()
        for rows in generate_workflow_sft_datasets().values()
        for row in rows
    }
    forbidden_questions = _questions(ROOT / "evaluation" / "workflow_planning_cases_v1.jsonl")
    forbidden_questions |= _questions(ROOT / "evaluation" / "workflow_final_cases_v1.jsonl")
    assert data_questions.isdisjoint(forbidden_questions)


def test_workflow_sft_written_bytes_are_cross_platform_frozen(tmp_path):
    paths = write_workflow_sft_datasets(tmp_path)
    expected_hashes = {
        "train": "b1355c025af32ab1b85b9b4cce64de0cdfa74aeaa8d689ab598f0c45e4b08aa0",
        "validation": "9e53d703141456bd059fc31ca67695c5ef9edc51af1c1839e3b0b7ee35e48d29",
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_sha256"] == expected_hashes
    for split, path in paths.items():
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert manifest["file_sha256"][split] == expected_hashes[split]
