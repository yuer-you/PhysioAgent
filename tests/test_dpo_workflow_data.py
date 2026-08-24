import hashlib
import json
from collections import Counter
from pathlib import Path

from physioagent.dpo_workflow_data import (
    COLUMNS,
    PREFERENCE_COUNTS,
    generate_workflow_dpo_datasets,
    validate_workflow_dpo_datasets,
    write_workflow_dpo_datasets,
)
from physioagent.sft_workflow_data import generate_workflow_sft_datasets
from physioagent.sft_workflow_data_v2 import generate_workflow_sft_v2_datasets
from physioagent.train_dpo import validate_dpo_rows
from physioagent.workflow import parse_workflow_plan


ROOT = Path(__file__).parents[1]


def _evaluation_questions(path: Path) -> set[str]:
    return {
        json.loads(line)["question"].strip().casefold()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _sft_questions(datasets: dict) -> set[str]:
    return {
        row["prompt"][1]["content"].strip().casefold()
        for rows in datasets.values()
        for row in rows
    }


def test_workflow_dpo_data_is_reproducible_valid_and_balanced():
    first = generate_workflow_dpo_datasets()
    second = generate_workflow_dpo_datasets()
    assert first == second
    assert {split: len(rows) for split, rows in first.items()} == {"train": 1000, "validation": 200}
    for split, rows in first.items():
        assert Counter(row["metadata"]["preference_type"] for row in rows) == PREFERENCE_COUNTS[split]
        validate_dpo_rows(rows, split)
        for row in rows:
            parse_workflow_plan(row["chosen"][0]["content"])
            parse_workflow_plan(row["rejected"][0]["content"])
            assert row["chosen"][0]["content"] != row["rejected"][0]["content"]
    validate_workflow_dpo_datasets(first)


def test_workflow_dpo_balances_preference_length_direction():
    expected = {
        "train": {"chosen_longer": 380, "equal": 320, "chosen_shorter": 300},
        "validation": {"chosen_longer": 76, "equal": 64, "chosen_shorter": 60},
    }
    for split, rows in generate_workflow_dpo_datasets().items():
        directions = Counter()
        for row in rows:
            chosen_count = row["metadata"]["chosen_step_count"]
            rejected_count = row["metadata"]["rejected_step_count"]
            if chosen_count > rejected_count:
                directions["chosen_longer"] += 1
            elif chosen_count < rejected_count:
                directions["chosen_shorter"] += 1
            else:
                directions["equal"] += 1
        assert directions == expected[split]


def test_workflow_dpo_uses_split_isolated_columns_and_no_frozen_questions():
    assert set(COLUMNS["train"]).isdisjoint(COLUMNS["validation"])
    dpo_questions = _sft_questions(generate_workflow_dpo_datasets())
    forbidden = _sft_questions(generate_workflow_sft_datasets())
    forbidden |= _sft_questions(generate_workflow_sft_v2_datasets())
    for name in (
        "workflow_planning_cases_v1.jsonl",
        "workflow_final_cases_v1.jsonl",
        "workflow_final_cases_v2.jsonl",
        "workflow_final_cases_v3.jsonl",
    ):
        forbidden |= _evaluation_questions(ROOT / "evaluation" / name)
    assert dpo_questions.isdisjoint(forbidden)


def test_workflow_dpo_written_bytes_are_cross_platform_frozen(tmp_path):
    paths = write_workflow_dpo_datasets(tmp_path)
    expected_hashes = {
        "train": "b6c7cd4244943cdfeb92b5dcb7ee805f4ffab8e212754b76ba4b1f3cae59ddd2",
        "validation": "b2a12a322522ccee388e7cbc82c9a7f939ec7212fc9ef90ad81fa738f84f053e",
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_sha256"] == expected_hashes
    assert manifest["frozen_tests_read_by_generator"] is False
    for split, path in paths.items():
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert hashlib.sha256(raw).hexdigest() == expected_hashes[split]
