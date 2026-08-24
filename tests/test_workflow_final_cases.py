import hashlib
import json
from collections import Counter
from pathlib import Path

from physioagent.evaluate_workflow_planner import evaluate_workflow_case, load_workflow_cases


ROOT = Path(__file__).parents[1]
DEVELOPMENT_CASES = ROOT / "evaluation" / "workflow_planning_cases_v1.jsonl"
FINAL_CASES = ROOT / "evaluation" / "workflow_final_cases_v1.jsonl"
MANIFEST = ROOT / "evaluation" / "workflow_final_cases_v1_manifest.json"


def test_frozen_workflow_final_manifest_and_coverage():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(FINAL_CASES.read_bytes()).hexdigest()
    cases = load_workflow_cases(FINAL_CASES)
    development_questions = {
        case["question"] for case in load_workflow_cases(DEVELOPMENT_CASES)
    }

    assert manifest["status"] == "frozen-unseen"
    assert actual_hash == manifest["cases_sha256"]
    assert len(cases) == manifest["num_cases"] == 30
    assert not development_questions.intersection(case["question"] for case in cases)
    assert Counter(case["category"] for case in cases) == manifest["coverage"]["categories"]
    assert {len(case["expected_steps"]) for case in cases} == {1, 2, 3}


def test_frozen_workflow_final_expected_plans_execute_end_to_end():
    rows = [
        evaluate_workflow_case(case, dry_run=True)
        for case in load_workflow_cases(FINAL_CASES)
    ]
    assert all(row["plan_exact"] for row in rows)
    assert all(row["execution_success"] for row in rows)
    assert all(row["reference_check_passed"] for row in rows)
    assert all(row["answer_grounded"] for row in rows)
    assert all(row["end_to_end_success"] for row in rows)
