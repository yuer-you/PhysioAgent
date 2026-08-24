"""让本地 Qwen 生成严格多步计划，再交给确定性 WorkflowExecutor。"""

from __future__ import annotations

from pathlib import Path

from .lora_model import MessageGenerator
from .workflow import (
    RecoveredWorkflowPlan,
    WorkflowExecutor,
    WorkflowResponse,
    WorkflowStep,
    parse_workflow_plan,
    parse_workflow_plan_with_recovery,
)


WORKFLOW_SYSTEM_PROMPT_V1 = """你是生理时序多步工具规划器。根据用户问题生成一个有限、线性的工具计划。

输出协议：
1. 只输出一个 JSON 对象，顶层只能包含 steps。
2. steps 必须是包含 1 到 4 个对象的数组。
3. 每一步必须且只能包含 name 和 arguments：
   {"name":"工具名","arguments":{}}
4. 禁止输出解释、Markdown、最终答案或任何前后缀。

允许的工具名只有：
load_signal | calculate_statistics | detect_peaks | calculate_heart_rate | filter_signal

规划规则：
1. 按用户明确要求的先后顺序排列工具。
2. 只有确实需要处理前一步输出时才生成多步；简单问题只生成一步。
3. arguments 只填写用户明确说出的参数，不得补默认值。
4. 文件路径、采样率、信号数组和 signal_profile 由程序提供，禁止写入 arguments。
5. ECG profile 使用冻结检测器，因此 detect_peaks 和 calculate_heart_rate 的 arguments 必须是 {}。
6. 用户要求“先滤波再计算心率”时，第一步 filter_signal，第二步 calculate_heart_rate。
7. 用户要求“先读取某列再统计”时，第一步 load_signal，第二步 calculate_statistics。

示例：
用户：先保留 1 到 30 Hz，再计算平均心率。
输出：{"steps":[{"name":"filter_signal","arguments":{"lowcut":1.0,"highcut":30.0}},{"name":"calculate_heart_rate","arguments":{}}]}

用户：检测 ECG R 峰。
输出：{"steps":[{"name":"detect_peaks","arguments":{}}]}

用户：先读取 ecg 列，再给出统计量。
输出：{"steps":[{"name":"load_signal","arguments":{"signal_column":"ecg"}},{"name":"calculate_statistics","arguments":{}}]}
"""


# v2 只针对 v1 暴露出的唯一错误增加边界规则，不改变工具集合或规划任务。
# 保留 v1 文本可以让两次实验的差异清晰、可复现。
WORKFLOW_SYSTEM_PROMPT_V2 = WORKFLOW_SYSTEM_PROMPT_V1 + """

v2 参数边界规则（优先遵守）：
1. 任何字符串参数都不能是空字符串、纯空白字符串或 null。
2. signal_column 只有在用户明确说出一个非空列名时才能填写。
3. 用户说“默认列”“default signal column”或没有说列名时，必须省略 signal_column，不能填写空字符串或 "default"。

边界示例：
用户：Read the default signal column and then report descriptive statistics.
错误输出：{"steps":[{"name":"load_signal","arguments":{"signal_column":""}},{"name":"calculate_statistics","arguments":{}}]}
正确输出：{"steps":[{"name":"load_signal","arguments":{}},{"name":"calculate_statistics","arguments":{}}]}
"""


# v3 保留 v2 的参数语义修复，并针对真实输出中遗漏 steps 数组右括号的问题，
# 增加纯格式自检。仍然不引入自动 JSON 修复，保证评测能暴露模型错误。
WORKFLOW_SYSTEM_PROMPT_V3 = WORKFLOW_SYSTEM_PROMPT_V2 + """

v3 JSON 闭合检查（只在内部检查，不要输出检查过程）：
1. 完整输出必须以 {"steps":[ 开始，并以 ]} 结束。
2. 每个左方括号 [ 必须有对应的右方括号 ]；每个左花括号 { 必须有对应的右花括号 }。
3. 完成最后一个 step 对象后，必须先用 ] 关闭 steps 数组，再用 } 关闭顶层对象。
4. 输出前再次确认最后两个字符恰好是 ]}。

合法的两步计划模板：
{"steps":[{"name":"load_signal","arguments":{}},{"name":"calculate_statistics","arguments":{}}]}
"""


WORKFLOW_SYSTEM_PROMPTS = {
    "v1": WORKFLOW_SYSTEM_PROMPT_V1,
    "v2": WORKFLOW_SYSTEM_PROMPT_V2,
    "v3": WORKFLOW_SYSTEM_PROMPT_V3,
}


class ModelWorkflowPlanner:
    """使用 MessageGenerator 生成并严格解析 workflow JSON。"""

    def __init__(
        self,
        generator: MessageGenerator,
        max_steps: int = 4,
        prompt_version: str = "v1",
        allow_recovery: bool = False,
    ) -> None:
        if prompt_version not in WORKFLOW_SYSTEM_PROMPTS:
            choices = ", ".join(sorted(WORKFLOW_SYSTEM_PROMPTS))
            raise ValueError(f"prompt_version must be one of: {choices}.")
        self.generator = generator
        self.max_steps = max_steps
        self.prompt_version = prompt_version
        self.allow_recovery = allow_recovery
        self.last_recovery: RecoveredWorkflowPlan | None = None

    def build_messages(self, question: str, signal_profile: str) -> list[dict[str, str]]:
        if signal_profile not in {"generic", "ecg"}:
            raise ValueError("signal_profile must be 'generic' or 'ecg'.")
        return [
            {
                "role": "system",
                "content": WORKFLOW_SYSTEM_PROMPTS[self.prompt_version]
                + f"\n当前 signal_profile：{signal_profile}",
            },
            {"role": "user", "content": question},
        ]

    def generate_plan(self, question: str, signal_profile: str) -> tuple[list[WorkflowStep], str]:
        raw_plan = self.generator.generate_messages(self.build_messages(question, signal_profile))
        if self.allow_recovery:
            recovered = parse_workflow_plan_with_recovery(raw_plan, self.max_steps)
            self.last_recovery = recovered
            return recovered.steps, raw_plan
        self.last_recovery = None
        return parse_workflow_plan(raw_plan, self.max_steps), raw_plan


class ModelWorkflowAgent:
    """模型只负责规划；已有确定性执行器负责状态与工具运行。"""

    def __init__(
        self,
        generator: MessageGenerator,
        max_steps: int = 4,
        prompt_version: str = "v1",
        allow_recovery: bool = False,
    ) -> None:
        self.planner = ModelWorkflowPlanner(
            generator,
            max_steps=max_steps,
            prompt_version=prompt_version,
            allow_recovery=allow_recovery,
        )
        self.executor = WorkflowExecutor(max_steps=max_steps)

    def run(
        self,
        question: str,
        file_path: str | Path,
        sampling_rate: float,
        signal_profile: str = "generic",
    ) -> WorkflowResponse:
        plan, raw_plan = self.planner.generate_plan(question, signal_profile)
        response = self.executor.run(question, plan, file_path, sampling_rate, signal_profile)
        response.planner = (
            "model_zero_shot"
            if self.planner.prompt_version == "v1"
            else f"model_zero_shot_{self.planner.prompt_version}"
        )
        response.raw_plan = raw_plan
        recovered = self.planner.last_recovery
        if recovered is not None:
            response.plan_recovery = {
                "recovery_applied": recovered.recovery_applied,
                "recovery_type": recovered.recovery_type,
                "strict_error": recovered.strict_error,
                "effective_text": recovered.effective_text,
            }
        return response
