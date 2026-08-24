import hashlib
import json
from collections import Counter
from pathlib import Path

from physioagent.sft_workflow_data import WORKFLOW_SFT_SYSTEM_PROMPT, generate_workflow_sft_datasets
from physioagent.sft_workflow_data_v2 import (
    CATEGORY_COUNTS_V2,
    COLUMNS_V2,
    LOAD_TEMPLATES_V2,
    generate_workflow_sft_v2_datasets,
    validate_workflow_sft_v2_datasets,
    write_workflow_sft_v2_datasets,
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


def _template_ids(split: str) -> set[str]:
    return {
        template_id
        for language_templates in LOAD_TEMPLATES_V2[split].values()
        for kind_templates in language_templates.values()
        for template_id, _ in kind_templates
    }


def test_workflow_sft_v2_is_reproducible_valid_and_has_expected_sizes():
    first = generate_workflow_sft_v2_datasets()
    second = generate_workflow_sft_v2_datasets()
    assert first == second
    assert {split: len(rows) for split, rows in first.items()} == {
        "train": 1800,
        "validation": 300,
    }
    for split, rows in first.items():
        assert Counter(row["metadata"]["category"] for row in rows) == CATEGORY_COUNTS_V2[split]
        assert all(row["prompt"][0]["content"] == WORKFLOW_SFT_SYSTEM_PROMPT for row in rows)
        assert all(row["metadata"]["task_type"] == "workflow" for row in rows)
        assert all(row["metadata"]["version"] == "workflow_sft_v2" for row in rows)
        for row in rows:
            parse_workflow_plan(row["completion"][0]["content"])
        validate_sft_rows(rows, split)
    validate_workflow_sft_v2_datasets(first)


def test_workflow_sft_v2_isolates_validation_paraphrases_and_columns():
    datasets = generate_workflow_sft_v2_datasets()
    assert _template_ids("train").isdisjoint(_template_ids("validation"))
    assert set(COLUMNS_V2["train"]).isdisjoint(COLUMNS_V2["validation"])

    for split, rows in datasets.items():
        used_template_ids = {
            row["metadata"]["load_template_id"]
            for row in rows
            if row["metadata"]["load_template_id"] is not None
        }
        assert used_template_ids == _template_ids(split)
        assert all(row["metadata"]["paraphrase_split"] == f"{split}_only" for row in rows)
        for row in rows:
            load_steps = [
                step for step in row["metadata"]["expected_steps"] if step["name"] == "load_signal"
            ]
            policy = row["metadata"]["load_policy"]
            if policy == "no_load":
                assert load_steps == []
            elif policy == "load_default":
                assert load_steps == [{"name": "load_signal", "arguments": {}}]
            else:
                assert policy == "load_explicit"
                assert len(load_steps) == 1
                column = load_steps[0]["arguments"]["signal_column"]
                assert column in COLUMNS_V2[split]


def test_workflow_sft_v2_balances_load_and_no_load_contrasts():
    datasets = generate_workflow_sft_v2_datasets()
    expected_step_counts = {
        "train": {1: 240, 2: 720, 3: 840},
        "validation": {1: 40, 2: 120, 3: 140},
    }
    for split, rows in datasets.items():
        assert Counter(row["metadata"]["step_count"] for row in rows) == expected_step_counts[split]
        policies = Counter(row["metadata"]["load_policy"] for row in rows)
        assert policies["load_default"] >= 85
        assert policies["load_explicit"] >= 85
        assert policies["no_load"] >= 120
        assert any(
            marker in row["prompt"][1]["content"].casefold()
            for row in rows
            for marker in ("不要重新读取", "without loading", "skip any read step", "省略加载动作")
        )


def test_workflow_sft_v2_never_teaches_an_incompatible_ecg_filter_chain():
    for rows in generate_workflow_sft_v2_datasets().values():
        for row in rows:
            steps = row["metadata"]["expected_steps"]
            if steps[-1]["name"] not in {"detect_peaks", "calculate_heart_rate"}:
                continue
            filter_steps = [step for step in steps if step["name"] == "filter_signal"]
            if not filter_steps:
                continue
            arguments = filter_steps[0]["arguments"]
            assert arguments
            assert arguments["lowcut"] <= 5.0
            assert arguments["highcut"] >= 15.0


def test_workflow_sft_v2_does_not_copy_previous_data_or_evaluation_questions():
    data_questions = {
        row["prompt"][1]["content"].strip().casefold()
        for rows in generate_workflow_sft_v2_datasets().values()
        for row in rows
    }
    old_sft_questions = {
        row["prompt"][1]["content"].strip().casefold()
        for rows in generate_workflow_sft_datasets().values()
        for row in rows
    }
    forbidden_questions = _questions(ROOT / "evaluation" / "workflow_planning_cases_v1.jsonl")
    forbidden_questions |= _questions(ROOT / "evaluation" / "workflow_final_cases_v1.jsonl")
    forbidden_questions |= _questions(ROOT / "evaluation" / "workflow_final_cases_v2.jsonl")
    assert data_questions.isdisjoint(old_sft_questions)
    assert data_questions.isdisjoint(forbidden_questions)


def test_workflow_sft_v2_written_bytes_are_cross_platform_frozen(tmp_path):
    paths = write_workflow_sft_v2_datasets(tmp_path)
    expected_hashes = {
        "train": "66da58371d627997e13a1bd37b7c59c7a4f062e2d8d84171a7b1e350125e63a8",
        "validation": "aead0dbb6f1069e1ea5f899399578455b6612aaedd955409b9096c0722190a74",
    }
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_sha256"] == expected_hashes
    assert manifest["frozen_tests_read_by_generator"] is False
    for split, path in paths.items():
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert hashlib.sha256(raw).hexdigest() == expected_hashes[split]
