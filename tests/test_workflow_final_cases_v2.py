import hashlib
import json
from collections import Counter
from pathlib import Path

from physioagent.evaluate_workflow_planner import evaluate_workflow_case, load_workflow_cases
from physioagent.sft_workflow_data import generate_workflow_sft_datasets


ROOT = Path(__file__).parents[1]
FINAL_V2 = ROOT / "evaluation" / "workflow_final_cases_v2.jsonl"
MANIFEST = ROOT / "evaluation" / "workflow_final_cases_v2_manifest.json"


def _case_questions(path: Path) -> set[str]:
    return {case["question"].strip().casefold() for case in load_workflow_cases(path)}


def test_workflow_final_v2_is_frozen_balanced_and_unseen():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases = load_workflow_cases(FINAL_V2)
    actual_hash = hashlib.sha256(FINAL_V2.read_bytes()).hexdigest()
    forbidden = _case_questions(ROOT / "evaluation" / "workflow_planning_cases_v1.jsonl")
    forbidden |= _case_questions(ROOT / "evaluation" / "workflow_final_cases_v1.jsonl")
    forbidden |= {
        row["prompt"][1]["content"].strip().casefold()
        for rows in generate_workflow_sft_datasets().values()
        for row in rows
    }
    questions = {case["question"].strip().casefold() for case in cases}

    assert manifest["status"] == "frozen-unseen"
    assert actual_hash == manifest["cases_sha256"]
    assert len(cases) == manifest["num_cases"] == 40
    assert questions.isdisjoint(forbidden)
    assert Counter(case["category"] for case in cases) == manifest["coverage"]["categories"]
    assert Counter(len(case["expected_steps"]) for case in cases) == {1: 6, 2: 16, 3: 18}


def test_workflow_final_v2_expected_plans_execute_end_to_end():
    rows = [evaluate_workflow_case(case, dry_run=True) for case in load_workflow_cases(FINAL_V2)]
    assert all(row["plan_exact"] for row in rows)
    assert all(row["execution_success"] for row in rows)
    assert all(row["reference_check_passed"] for row in rows)
    assert all(row["answer_grounded"] for row in rows)
    assert all(row["end_to_end_success"] for row in rows)
