import copy

import pytest

from physioagent.evaluate_final import build_final_summary, validate_result_rows


def _case(case_id="one"):
    return {
        "id": case_id,
        "category": "statistics",
        "question": "stats",
        "expected_name": "calculate_statistics",
        "expected_arguments": {},
    }


def _result(case_id="one", exact=True):
    return {
        **_case(case_id),
        "valid_tool_call": True,
        "name_correct": exact,
        "requested_arguments_correct": exact,
        "extra_arguments": [],
        "has_extra_arguments": False,
        "arguments_exact": exact,
        "exact_match": exact,
    }


def test_validate_result_rows_rejects_label_changes():
    row = _result()
    validate_result_rows([row], [_case()], "model")
    changed = copy.deepcopy(row)
    changed["expected_name"] = "detect_peaks"
    with pytest.raises(ValueError, match="label changed"):
        validate_result_rows([changed], [_case()], "model")


def test_build_final_summary_preserves_precommitted_selection():
    results = {
        "prompt_v4": [_result(exact=False)],
        "lora_v1": [_result(exact=True)],
        "lora_v2": [_result(exact=True)],
    }
    summary = build_final_summary(
        results,
        frozen_sha256="abc",
        model_path="base",
        v1_adapter="v1",
        v2_adapter="v2",
    )
    assert summary["precommitted_selection"]["selected_model"] == "lora_v2"
    assert summary["precommitted_selection"]["selection_used_final_results"] is False
    assert summary["models"]["prompt_v4"]["metrics"]["exact_match_accuracy"] == 0.0
    assert summary["models"]["lora_v2"]["metrics"]["exact_match_accuracy"] == 1.0
