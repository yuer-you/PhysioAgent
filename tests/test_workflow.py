from pathlib import Path

import pytest

from physioagent.workflow import (
    RuleBasedWorkflowAgent,
    RuleBasedWorkflowPlanner,
    WorkflowExecutor,
    WorkflowStep,
)


REAL_ECG_207 = Path(__file__).parents[1] / "data" / "real" / "mitdb" / "207_30s" / "signal.csv"


def test_workflow_planner_builds_filter_then_heart_rate_plan():
    plan = RuleBasedWorkflowPlanner().plan("先用 0.5 到 40 Hz 滤波，再计算平均心率")
    assert [step.name for step in plan] == ["filter_signal", "calculate_heart_rate"]
    assert plan[0].arguments == {"lowcut": 0.5, "highcut": 40.0}
    assert plan[1].arguments == {}


def test_workflow_planner_extracts_english_range_and_chinese_order():
    plan = RuleBasedWorkflowPlanner().plan("先用三阶 filter between 5 and 15 Hz，再检测 R 峰")
    assert [step.name for step in plan] == ["filter_signal", "detect_peaks"]
    assert plan[0].arguments == {"lowcut": 5.0, "highcut": 15.0, "order": 3}


def test_workflow_planner_extracts_english_order():
    plan = RuleBasedWorkflowPlanner().plan(
        "Filter between 0.5 and 40 Hz using order 4, then report heart rate."
    )
    assert plan[0].arguments == {"lowcut": 0.5, "highcut": 40.0, "order": 4}


def test_workflow_passes_filtered_signal_to_ecg_heart_rate_in_memory():
    response = RuleBasedWorkflowAgent().run(
        "请先用 0.5 到 40 Hz 对 ECG 滤波，再计算平均心率。",
        REAL_ECG_207,
        sampling_rate=360,
        signal_profile="ecg",
    )
    assert [item.tool_name for item in response.trace] == ["filter_signal", "calculate_heart_rate"]
    assert response.trace[0].input_source == "original_signal"
    assert response.trace[1].input_source == "step_1_filter_signal_output"
    assert response.trace[0].result_summary["num_samples"] == 10800
    assert response.final_result["num_peaks"] == 29
    assert response.final_result["mean_heart_rate_bpm"] == pytest.approx(56.84744806842749)
    assert "56.8 BPM" in response.answer
    assert response.stop_reason == "plan_completed"


def test_workflow_can_end_with_peak_detection():
    response = RuleBasedWorkflowAgent().run(
        "先用 5 到 15 Hz 滤波，再找出 R 峰",
        REAL_ECG_207,
        sampling_rate=360,
        signal_profile="ecg",
    )
    assert response.trace[-1].tool_name == "detect_peaks"
    assert response.final_result["num_peaks"] == 29


def test_workflow_planner_rejects_unknown_goal():
    with pytest.raises(ValueError, match="could not identify"):
        RuleBasedWorkflowPlanner().plan("请评价患者是否患病")


def test_workflow_executor_enforces_step_limit():
    plan = [WorkflowStep("calculate_statistics") for _ in range(5)]
    with pytest.raises(ValueError, match="exceeds max_steps=4"):
        WorkflowExecutor(max_steps=4).run(
            "重复统计",
            plan,
            REAL_ECG_207,
            sampling_rate=360,
            signal_profile="ecg",
        )


def test_workflow_blocks_filter_that_removes_required_ecg_band():
    plan = [WorkflowStep("filter_signal"), WorkflowStep("calculate_heart_rate")]
    with pytest.raises(ValueError, match="ecg_detector_v1 requires the full 5-15 Hz"):
        WorkflowExecutor().run(
            "默认滤波后计算心率",
            plan,
            REAL_ECG_207,
            sampling_rate=360,
            signal_profile="ecg",
        )
