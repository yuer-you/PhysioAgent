"""不依赖 Agent 框架的确定性多步工具控制器。

这一版只实现线性计划，不做循环、反思或模型自由规划。每一步都读取上一状态中的
内存信号，并记录输入来源、参数、结果摘要和 grounded answer。

Deterministic multi-step tool controller without an agent framework. This version implements only linear
plans, without loops, reflection, or unconstrained model planning. Every step reads the in-memory signal
from the previous state and records its input source, arguments, result summary, and grounded answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .agent import ToolCall, parse_tool_call
from .ecg import ECG_DETECTOR_V1_CONFIG, calculate_ecg_heart_rate, detect_ecg_r_peaks
from .tools import calculate_heart_rate, calculate_statistics, detect_peaks, filter_signal, load_signal


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStepTrace:
    step: int
    tool_name: str
    arguments: dict[str, Any]
    input_source: str
    result_summary: object
    answer: str


@dataclass
class WorkflowResponse:
    question: str
    signal_profile: str
    plan: list[WorkflowStep]
    trace: list[WorkflowStepTrace]
    final_result: object
    answer: str
    stop_reason: str = "plan_completed"
    planner: str = "rule_based"
    raw_plan: str | None = None
    plan_recovery: dict[str, Any] | None = None


@dataclass(frozen=True)
class RecoveredWorkflowPlan:
    """严格解析或一次保守恢复后的计划，以及完整审计信息。

    A strictly parsed or conservatively recovered plan with complete audit information.
    """

    steps: list[WorkflowStep]
    original_text: str
    effective_text: str
    recovery_applied: bool = False
    recovery_type: str | None = None
    strict_error: str | None = None


def parse_workflow_plan(text: str, max_steps: int = 4) -> list[WorkflowStep]:
    """严格解析模型计划：顶层只能有 steps，且每一步都是合法工具调用。

    Strictly parse a model plan whose top level contains only steps and whose steps are valid tool calls.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as error:
        raise ValueError(f"Workflow output must be exactly one JSON object: {text!r}") from error
    if not isinstance(payload, dict) or set(payload) != {"steps"}:
        raise ValueError("Workflow plan must be an object containing only the 'steps' field.")
    steps_payload = payload["steps"]
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("Workflow steps must be a non-empty JSON array.")
    if len(steps_payload) > max_steps:
        raise ValueError(f"Workflow plan exceeds max_steps={max_steps}.")

    steps: list[WorkflowStep] = []
    for index, step_payload in enumerate(steps_payload, start=1):
        if not isinstance(step_payload, dict):
            raise ValueError(f"Workflow step {index} must be a JSON object.")
        try:
            call = parse_tool_call(json.dumps(step_payload, ensure_ascii=False))
        except ValueError as error:
            raise ValueError(f"Invalid workflow step {index}: {error}") from error
        steps.append(WorkflowStep(call.name, call.arguments))
    return steps


def parse_workflow_plan_with_recovery(
    text: str,
    max_steps: int = 4,
) -> RecoveredWorkflowPlan:
    """先严格解析；仅在唯一可判定的缺失 steps 右括号场景下恢复。

    Parse strictly first and recover only the uniquely identifiable missing steps closing bracket.
    """
    try:
        steps = parse_workflow_plan(text, max_steps=max_steps)
        return RecoveredWorkflowPlan(
            steps=steps,
            original_text=text,
            effective_text=text.strip(),
        )
    except ValueError as strict_error:
        candidate = _repair_missing_steps_closing_bracket(text)
        if candidate is None:
            raise strict_error
        try:
            steps = parse_workflow_plan(candidate, max_steps=max_steps)
        except ValueError:
            # 候选文本仍不满足完整 schema 时，报告原始严格解析错误，不做第二种猜测。
            # If the candidate still fails the full schema, report the original error without another guess.
            raise strict_error
        return RecoveredWorkflowPlan(
            steps=steps,
            original_text=text,
            effective_text=candidate,
            recovery_applied=True,
            recovery_type="insert_missing_steps_closing_bracket",
            strict_error=str(strict_error),
        )


def _repair_missing_steps_closing_bracket(text: str) -> str | None:
    """当最外层 steps 数组唯一缺少 `]` 时构造一个候选文本。

    Build one candidate when the outermost steps array is missing only its closing `]`.
    """
    stripped = text.strip()
    if not re.match(r'^\{\s*"steps"\s*:\s*\[', stripped) or not stripped.endswith("}"):
        return None

    # 暂时移除最后的顶层 `}`。若剩余文本恰好只有顶层对象和 steps 数组未闭合，
    # 就能唯一确定应当先补 `]`，再放回这个 `}`。
    # Temporarily remove the final top-level `}`. If only the top object and steps array remain open,
    # the unique repair is to insert `]` before restoring that `}`.
    open_delimiters = _open_json_delimiters(stripped[:-1])
    if open_delimiters != ["{", "["]:
        return None
    return stripped[:-1] + "]}"


def _open_json_delimiters(text: str) -> list[str] | None:
    """返回 JSON 文本外部尚未闭合的括号；字符串内部的括号不参与计算。

    Return unclosed delimiters outside JSON strings; delimiters inside strings do not count.
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or stack[-1] != pairs[character]:
                return None
            stack.pop()
    if in_string or escaped:
        return None
    return stack


class RuleBasedWorkflowPlanner:
    """把少量明确的组合问法转换为线性工具计划。

    Convert a small set of explicit compound requests into linear tool plans.
    """

    def plan(self, question: str) -> list[WorkflowStep]:
        text = question.strip()
        if not text:
            raise ValueError("question must not be empty.")
        lowered = text.lower()
        steps: list[WorkflowStep] = []

        asks_for_filter = any(token in lowered for token in ("滤波", "filter", "hz", "赫兹"))
        if asks_for_filter:
            steps.append(WorkflowStep("filter_signal", self._extract_filter_arguments(text)))

        # 心率是最终目标时，即使问题也提到 R 峰，也应以心率工具结束。
        # When heart rate is the final goal, finish with the HR tool even if the question mentions R peaks.
        if any(token in lowered for token in ("心率", "bpm", "heart rate", "每分钟", "跳多少")):
            steps.append(WorkflowStep("calculate_heart_rate"))
        elif any(token in lowered for token in ("r 峰", "r峰", "峰值", "peaks", "peak")):
            steps.append(WorkflowStep("detect_peaks"))
        elif any(token in lowered for token in ("统计", "均值", "标准差", "statistics", "mean", "时长")):
            steps.append(WorkflowStep("calculate_statistics"))

        if not steps:
            raise ValueError("The rule-based workflow planner could not identify a supported goal.")
        return [self._validate_step(step) for step in steps]

    @staticmethod
    def _extract_filter_arguments(question: str) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        range_patterns = (
            r"(\d+(?:\.\d+)?)\s*(?:到|至|[-~])\s*(\d+(?:\.\d+)?)\s*(?:hz|赫兹)",
            r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*hz",
        )
        for pattern in range_patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                arguments["lowcut"] = float(match.group(1))
                arguments["highcut"] = float(match.group(2))
                break

        order_match = re.search(r"(\d+)\s*阶", question)
        if not order_match:
            order_match = re.search(r"order\s*(\d+)", question, flags=re.IGNORECASE)
        if not order_match:
            order_match = re.search(r"([一二三四五六七八九十])阶", question)
            if order_match:
                chinese_numbers = {
                    "一": 1,
                    "二": 2,
                    "三": 3,
                    "四": 4,
                    "五": 5,
                    "六": 6,
                    "七": 7,
                    "八": 8,
                    "九": 9,
                    "十": 10,
                }
                arguments["order"] = chinese_numbers[order_match.group(1)]
        else:
            arguments["order"] = int(order_match.group(1))
        return arguments

    @staticmethod
    def _validate_step(step: WorkflowStep) -> WorkflowStep:
        call = parse_tool_call(
            json.dumps({"name": step.name, "arguments": step.arguments}, ensure_ascii=False)
        )
        return WorkflowStep(call.name, call.arguments)


class WorkflowExecutor:
    """顺序执行计划，并在内存中传递最新信号。

    Execute a plan sequentially while passing the latest signal in memory.
    """

    def __init__(self, max_steps: int = 4) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self.max_steps = max_steps

    def run(
        self,
        question: str,
        plan: list[WorkflowStep],
        file_path: str | Path,
        sampling_rate: float,
        signal_profile: str = "generic",
    ) -> WorkflowResponse:
        if signal_profile not in {"generic", "ecg"}:
            raise ValueError("signal_profile must be 'generic' or 'ecg'.")
        if not plan:
            raise ValueError("workflow plan must contain at least one step.")
        if len(plan) > self.max_steps:
            raise ValueError(f"workflow plan exceeds max_steps={self.max_steps}.")
        self._validate_plan_compatibility(plan, signal_profile)

        current_signal = load_signal(file_path)
        current_source = "original_signal"
        trace: list[WorkflowStepTrace] = []
        final_result: object = current_signal

        for index, step in enumerate(plan, start=1):
            call = RuleBasedWorkflowPlanner._validate_step(step)
            input_source = current_source
            result, answer = self._run_on_signal(
                call,
                current_signal,
                sampling_rate,
                signal_profile,
                file_path,
            )
            if call.name in {"load_signal", "filter_signal"}:
                current_signal = np.asarray(result, dtype=float)
                current_source = f"step_{index}_{call.name}_output"
            final_result = result
            trace.append(
                WorkflowStepTrace(
                    step=index,
                    tool_name=call.name,
                    arguments=dict(call.arguments),
                    input_source=input_source,
                    result_summary=summarize_workflow_result(result),
                    answer=answer,
                )
            )

        final_answer = f"已按顺序执行 {len(trace)} 个工具。{trace[-1].answer}"
        return WorkflowResponse(
            question=question,
            signal_profile=signal_profile,
            plan=plan,
            trace=trace,
            final_result=final_result,
            answer=final_answer,
        )

    @staticmethod
    def _validate_plan_compatibility(plan: list[WorkflowStep], signal_profile: str) -> None:
        """阻止前置滤波删除下游 ECG 检测器必需的频率信息。

        Prevent upstream filtering from removing frequencies required by the downstream ECG detector.
        """
        if signal_profile != "ecg":
            return
        active_filter_bands: list[tuple[float, float]] = []
        for step in plan:
            if step.name == "load_signal":
                # 重新加载原始信号会清除之前所有滤波的影响。
                # Reloading the original signal clears the effect of every previous filter.
                active_filter_bands = []
            elif step.name == "filter_signal":
                active_filter_bands.append(
                    (
                        float(step.arguments.get("lowcut", 0.5)),
                        float(step.arguments.get("highcut", 8.0)),
                    )
                )
            elif step.name in {"detect_peaks", "calculate_heart_rate"}:
                required_low = ECG_DETECTOR_V1_CONFIG.lowcut_hz
                required_high = ECG_DETECTOR_V1_CONFIG.highcut_hz
                for lowcut, highcut in active_filter_bands:
                    if lowcut > required_low or highcut < required_high:
                        raise ValueError(
                            "Incompatible ECG workflow: an upstream filter keeps "
                            f"{lowcut:g}-{highcut:g} Hz, but ecg_detector_v1 requires the full "
                            f"{required_low:g}-{required_high:g} Hz band. Use a wider filter or "
                            "remove the filter step."
                        )

    @staticmethod
    def _run_on_signal(
        call: WorkflowStep,
        signal: np.ndarray,
        sampling_rate: float,
        signal_profile: str,
        file_path: str | Path,
    ) -> tuple[object, str]:
        args = dict(call.arguments)
        if call.name == "load_signal":
            result = load_signal(file_path, **args)
            return result, f"已读取信号，共 {len(result)} 个采样点。"
        if call.name == "filter_signal":
            result = filter_signal(signal, sampling_rate, **args)
            return result, f"已完成带通滤波，得到 {len(result)} 个采样点。"
        if call.name == "calculate_statistics":
            result = calculate_statistics(signal, sampling_rate)
            answer = (
                f"信号共 {result['num_samples']} 点、时长 {result['duration_seconds']:.2f} 秒，"
                f"均值 {result['mean']:.3f}。"
            )
            return result, answer
        if call.name in {"detect_peaks", "calculate_heart_rate"} and signal_profile == "ecg":
            if args:
                raise ValueError(
                    f"{call.name} with signal_profile='ecg' uses frozen ecg_detector_v1 "
                    "and accepts no workflow-generated peak parameters."
                )
            if call.name == "detect_peaks":
                result = detect_ecg_r_peaks(signal, sampling_rate)
                return result, f"检测到 {result['num_peaks']} 个 ECG R 峰，位置为 {result['peak_indices']}。"
            result = calculate_ecg_heart_rate(signal, sampling_rate)
            return result, (
                f"估计平均心率为 {result['mean_heart_rate_bpm']:.1f} BPM"
                f"（检测到 {result['num_peaks']} 个峰）。"
            )
        if call.name == "detect_peaks":
            result = detect_peaks(signal, sampling_rate, **args)
            return result, f"检测到 {result['num_peaks']} 个峰，位置为 {result['peak_indices']}。"
        if call.name == "calculate_heart_rate":
            result = calculate_heart_rate(signal, sampling_rate, **args)
            return result, (
                f"估计平均心率为 {result['mean_heart_rate_bpm']:.1f} BPM"
                f"（检测到 {result['num_peaks']} 个峰）。"
            )
        raise ValueError(f"Unsupported workflow tool: {call.name}")


class RuleBasedWorkflowAgent:
    """组合规则规划器和内存执行器的最小多步 Agent。

    Minimal multi-step agent combining the rule planner and in-memory executor.
    """

    def __init__(self) -> None:
        self.planner = RuleBasedWorkflowPlanner()
        self.executor = WorkflowExecutor()

    def run(
        self,
        question: str,
        file_path: str | Path,
        sampling_rate: float,
        signal_profile: str = "generic",
    ) -> WorkflowResponse:
        plan = self.planner.plan(question)
        return self.executor.run(question, plan, file_path, sampling_rate, signal_profile)


def summarize_workflow_result(result: object) -> object:
    if isinstance(result, np.ndarray):
        return {
            "type": "ndarray",
            "num_samples": int(result.size),
            "mean": float(np.mean(result)),
            "std": float(np.std(result)),
            "min": float(np.min(result)),
            "max": float(np.max(result)),
        }
    return result


def workflow_response_to_dict(
    response: WorkflowResponse,
    signal_file: str | Path,
    sampling_rate: float,
) -> dict[str, Any]:
    """把 workflow 轨迹转换成紧凑、可写 JSON 的结构。

    Convert a workflow trace into a compact JSON-serializable structure.
    """
    return {
        "workflow": (
            "model_workflow_v1"
            if response.planner.startswith("model_zero_shot")
            else "rule_based_workflow_v1"
        ),
        "planner": response.planner,
        "raw_plan": response.raw_plan,
        "plan_recovery": response.plan_recovery,
        "question": response.question,
        "signal_file": str(signal_file),
        "sampling_rate_hz": sampling_rate,
        "signal_profile": response.signal_profile,
        "plan": [{"name": step.name, "arguments": step.arguments} for step in response.plan],
        "trace": [
            {
                "step": item.step,
                "tool_name": item.tool_name,
                "arguments": item.arguments,
                "input_source": item.input_source,
                "result_summary": item.result_summary,
                "answer": item.answer,
            }
            for item in response.trace
        ],
        "final_result_summary": summarize_workflow_result(response.final_result),
        "stop_reason": response.stop_reason,
        "answer": response.answer,
    }
