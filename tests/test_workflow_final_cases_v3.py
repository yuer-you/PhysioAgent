import hashlib
import json
from collections import Counter
from pathlib import Path

from physioagent.evaluate_workflow_planner import evaluate_workflow_case, load_workflow_cases
from physioagent.sft_workflow_data import generate_workflow_sft_datasets
from physioagent.sft_workflow_data_v2 import generate_workflow_sft_v2_datasets


ROOT = Path(__file__).parents[1]
FINAL_V3 = ROOT / "evaluation" / "workflow_final_cases_v3.jsonl"
MANIFEST = ROOT / "evaluation" / "workflow_final_cases_v3_manifest.json"


def _case_questions(path: Path) -> set[str]:
    return {case["question"].strip().casefold() for case in load_workflow_cases(path)}


def _sft_questions(datasets: dict) -> set[str]:
    return {
        row["prompt"][1]["content"].strip().casefold()
        for rows in datasets.values()
        for row in rows
    }


def _load_policy(case: dict) -> str:
    load_steps = [step for step in case["expected_steps"] if step["name"] == "load_signal"]
    if not load_steps:
        return "no_load"
    return "load_explicit" if load_steps[0]["arguments"] else "load_default"


def test_workflow_final_v3_is_frozen_balanced_and_unseen():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = load_workflow_cases(FINAL_V3)
    actual_hash = hashlib.sha256(FINAL_V3.read_bytes()).hexdigest()
    forbidden = _case_questions(ROOT / "evaluation" / "workflow_planning_cases_v1.jsonl")
    forbidden |= _case_questions(ROOT / "evaluation" / "workflow_final_cases_v1.jsonl")
    forbidden |= _case_questions(ROOT / "evaluation" / "workflow_final_cases_v2.jsonl")
    forbidden |= _sft_questions(generate_workflow_sft_datasets())
    forbidden |= _sft_questions(generate_workflow_sft_v2_datasets())
    questions = {case["question"].strip().casefold() for case in cases}

    assert manifest["status"] == "frozen-unseen"
    assert manifest["frozen_before_adapter_training"] is True
    assert actual_hash == manifest["cases_sha256"]
    assert len(cases) == manifest["num_cases"] == 60
    assert len(questions) == 60
    assert questions.isdisjoint(forbidden)
    assert Counter(case["category"] for case in cases) == manifest["coverage"]["categories"]
    assert Counter(len(case["expected_steps"]) for case in cases) == {1: 8, 2: 26, 3: 26}
    assert Counter(_load_policy(case) for case in cases) == {
        "load_explicit": 18,
        "load_default": 18,
        "no_load": 24,
    }


def test_workflow_final_v3_expected_plans_execute_end_to_end():
    rows = [evaluate_workflow_case(case, dry_run=True) for case in load_workflow_cases(FINAL_V3)]
    assert all(row["plan_exact"] for row in rows)
    assert all(row["execution_success"] for row in rows)
    assert all(row["reference_check_passed"] for row in rows)
    assert all(row["answer_grounded"] for row in rows)
    assert all(row["end_to_end_success"] for row in rows)
