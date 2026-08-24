import hashlib
import json
from collections import Counter
from pathlib import Path

from physioagent.dpo_workflow_data import generate_workflow_dpo_datasets
from physioagent.evaluate_workflow_planner import evaluate_workflow_case, load_workflow_cases
from physioagent.sft_workflow_data import generate_workflow_sft_datasets
from physioagent.sft_workflow_data_v2 import generate_workflow_sft_v2_datasets
from physioagent.workflow_final_v4_data import (
    CATEGORY_COUNTS_V4,
    generate_workflow_final_v4_cases,
)


ROOT = Path(__file__).parents[1]
FINAL_V4 = ROOT / "evaluation" / "workflow_final_cases_v4.jsonl"
MANIFEST = ROOT / "evaluation" / "workflow_final_cases_v4_manifest.json"


def _evaluation_questions(path: Path) -> set[str]:
    return {case["question"].strip().casefold() for case in load_workflow_cases(path)}


def _training_questions(datasets: dict) -> set[str]:
    return {
        row["prompt"][1]["content"].strip().casefold()
        for rows in datasets.values()
        for row in rows
    }


def _stored_case(case: dict) -> dict:
    return {key: value for key, value in case.items() if key not in {"load_policy", "language"}}


def test_workflow_final_v4_is_frozen_balanced_and_unseen():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stored_cases = load_workflow_cases(FINAL_V4)
    generated_cases = generate_workflow_final_v4_cases()
    forbidden = _training_questions(generate_workflow_sft_datasets())
    forbidden |= _training_questions(generate_workflow_sft_v2_datasets())
    forbidden |= _training_questions(generate_workflow_dpo_datasets())
    for name in (
        "workflow_planning_cases_v1.jsonl",
        "workflow_final_cases_v1.jsonl",
        "workflow_final_cases_v2.jsonl",
        "workflow_final_cases_v3.jsonl",
    ):
        forbidden |= _evaluation_questions(ROOT / "evaluation" / name)
    questions = {case["question"].strip().casefold() for case in generated_cases}

    assert manifest["status"] == "frozen-unseen"
    assert manifest["frozen_before_dpo_training"] is True
    assert hashlib.sha256(FINAL_V4.read_bytes()).hexdigest() == manifest["cases_sha256"]
    assert len(stored_cases) == len(generated_cases) == manifest["num_cases"] == 80
    assert stored_cases == [_stored_case(case) for case in generated_cases]
    assert questions.isdisjoint(forbidden)
    assert Counter(case["category"] for case in generated_cases) == CATEGORY_COUNTS_V4
    assert Counter(len(case["expected_steps"]) for case in generated_cases) == {1: 10, 2: 28, 3: 42}
    assert Counter(case["load_policy"] for case in generated_cases) == {
        "load_explicit": 27,
        "load_default": 27,
        "no_load": 26,
    }
    assert Counter(case["language"] for case in generated_cases) == {"zh": 40, "en": 40}


def test_workflow_final_v4_expected_plans_execute_end_to_end():
    rows = [evaluate_workflow_case(case, dry_run=True) for case in load_workflow_cases(FINAL_V4)]
    assert all(row["plan_exact"] for row in rows)
    assert all(row["execution_success"] for row in rows)
    assert all(row["reference_check_passed"] for row in rows)
    assert all(row["answer_grounded"] for row in rows)
    assert all(row["end_to_end_success"] for row in rows)
