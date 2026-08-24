"""Agent 决策和工具执行。

RuleBasedAgent 与 QwenAgent 共用同一执行器。二者唯一的区别是如何产生工具调用，
这样后续比较规则、原始模型和微调模型时是公平的。

Agent decisions and tool execution. RuleBasedAgent and QwenAgent share the same executor; their only
difference is how they produce tool calls. This keeps later comparisons among rules, the base model,
and fine-tuned models fair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ecg import calculate_ecg_heart_rate, detect_ecg_r_peaks
from .schemas import TOOL_NAMES, TOOL_SCHEMAS
from .tools import calculate_heart_rate, calculate_statistics, detect_peaks, filter_signal, load_signal


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    tool_name: str
    tool_result: object
    answer: str
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    raw_decision: str | None = None
    signal_profile: str = "generic"


def parse_tool_call(text: str) -> ToolCall:
    """严格验证模型输出：只能包含一个完整的 JSON 工具调用对象。

    Strictly validate that model output contains exactly one complete JSON tool-call object.
    """
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as error:
        raise ValueError(f"Model output must be exactly one JSON object: {text!r}") from error

    if not isinstance(payload, dict):
        raise ValueError("Tool call must be a JSON object.")
    expected_keys = {"name", "arguments"}
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"Tool call must contain only name and arguments; missing={missing}, extra={extra}.")
    name = payload.get("name")
    arguments = payload["arguments"]
    if name not in TOOL_NAMES:
        raise ValueError(f"Unknown tool name: {name!r}")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    _validate_arguments(name, arguments)
    return ToolCall(name=name, arguments=arguments)


def _validate_arguments(name: str, arguments: dict[str, Any]) -> None:
    """根据 schema 检查参数名和最基础的 JSON 类型。

    Validate argument names and basic JSON types against the schema.
    """
    schema = next(item["function"] for item in TOOL_SCHEMAS if item["function"]["name"] == name)
    properties = schema["parameters"]["properties"]
    unexpected = set(arguments) - set(properties)
    if unexpected:
        raise ValueError(f"Unexpected arguments for {name}: {sorted(unexpected)}")

    python_types = {"string": str, "number": (int, float), "integer": int}
    for key, value in arguments.items():
        expected = python_types[properties[key]["type"]]
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ValueError(f"Argument {key!r} has the wrong type.")
        if properties[key].get("minLength") is not None and len(value.strip()) < properties[key]["minLength"]:
            raise ValueError(f"Argument {key!r} must not be empty or whitespace-only.")


class ToolExecutor:
    """把已经验证的工具调用映射到普通 Python 函数。

    Map a validated tool call to an ordinary Python function.
    """

    def run(
        self,
        call: ToolCall,
        file_path: str | Path,
        sampling_rate: float,
        raw_decision: str | None = None,
        signal_profile: str = "generic",
    ) -> AgentResponse:
        if signal_profile not in {"generic", "ecg"}:
            raise ValueError("signal_profile must be 'generic' or 'ecg'.")
        args = dict(call.arguments)
        if call.name not in TOOL_NAMES:
            raise ValueError(f"Unknown tool name: {call.name!r}")
        _validate_arguments(call.name, args)

        if call.name == "load_signal":
            signal = load_signal(file_path, **args)
            result: object = signal
            answer = f"已读取信号，共 {len(signal)} 个采样点。"
        else:
            signal = load_signal(file_path)
            if call.name == "calculate_heart_rate":
                if signal_profile == "ecg":
                    _require_frozen_ecg_arguments(call.name, args)
                    result = calculate_ecg_heart_rate(signal, sampling_rate)
                else:
                    result = calculate_heart_rate(signal, sampling_rate, **args)
                answer = f"估计平均心率为 {result['mean_heart_rate_bpm']:.1f} BPM（检测到 {result['num_peaks']} 个峰）。"
            elif call.name == "detect_peaks":
                if signal_profile == "ecg":
                    _require_frozen_ecg_arguments(call.name, args)
                    result = detect_ecg_r_peaks(signal, sampling_rate)
                    answer = f"检测到 {result['num_peaks']} 个 ECG R 峰，位置为 {result['peak_indices']}。"
                else:
                    result = detect_peaks(signal, sampling_rate, **args)
                    answer = f"检测到 {result['num_peaks']} 个峰，位置为 {result['peak_indices']}。"
            elif call.name == "filter_signal":
                result = filter_signal(signal, sampling_rate, **args)
                answer = f"已完成带通滤波，得到 {len(result)} 个采样点。"
            elif call.name == "calculate_statistics":
                result = calculate_statistics(signal, sampling_rate)
                answer = f"信号共 {result['num_samples']} 点、时长 {result['duration_seconds']:.2f} 秒，均值 {result['mean']:.3f}。"
            else:  # parse_tool_call 已拦截未知名称；这里防止手动构造非法 ToolCall。
                # parse_tool_call rejects unknown names; this branch guards manually constructed invalid ToolCalls.
                raise ValueError(f"Unknown tool name: {call.name!r}")

        return AgentResponse(
            tool_name=call.name,
            tool_result=result,
            answer=answer,
            tool_arguments=args,
            raw_decision=raw_decision,
            signal_profile=signal_profile,
        )


def _require_frozen_ecg_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    """ECG v1 配置已冻结，不能让模型输出覆盖其检测参数。

    The ECG v1 configuration is frozen, so model output cannot override detector parameters.
    """
    if arguments:
        raise ValueError(
            f"{tool_name} with signal_profile='ecg' uses the frozen ecg_detector_v1 configuration "
            "and does not accept model-generated peak parameters."
        )


class RuleBasedAgent:
    """用关键词选择工具的对照基线。

    Keyword-based tool-selection control baseline.
    """

    def __init__(self) -> None:
        self.executor = ToolExecutor()

    def run(
        self,
        question: str,
        file_path: str | Path,
        sampling_rate: float,
        signal_profile: str = "generic",
    ) -> AgentResponse:
        call = ToolCall(name=self._decide(question))
        return self.executor.run(call, file_path, sampling_rate, signal_profile=signal_profile)

    @staticmethod
    def _decide(question: str) -> str:
        text = question.lower()
        if any(word in text for word in ("heart rate", "心率", "bpm", "hr")):
            return "calculate_heart_rate"
        if any(word in text for word in ("peak", "peaks", "峰值", "峰")):
            return "detect_peaks"
        if any(word in text for word in ("filter", "滤波", "噪声")):
            return "filter_signal"
        if any(word in text for word in ("load", "read", "加载", "读取")):
            return "load_signal"
        return "calculate_statistics"


class QwenAgent:
    """从本地目录加载 Qwen，并用它生成结构化工具调用。

    Load Qwen from a local directory and use it to generate structured tool calls.
    """

    def __init__(
        self,
        model_path: str | Path,
        prompt_version: str = "v4",
        max_new_tokens: int = 128,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if prompt_version not in {"v1", "v2", "v3", "v4"}:
            raise ValueError("prompt_version must be 'v1', 'v2', 'v3', or 'v4'.")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer.")
        self.model_path = str(model_path)
        self.prompt_version = prompt_version
        self.max_new_tokens = int(max_new_tokens)
        self.last_generation_info: dict[str, Any] | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
        self.model.eval()
        # Qwen 模型自带的 generation_config 含采样参数。基线使用确定性解码，
        # 因而关闭采样并清空只对采样有效的参数，避免无意义的警告。
        # Qwen's generation_config includes sampling parameters. The baseline uses deterministic decoding,
        # so sampling is disabled and sampling-only fields are cleared to avoid irrelevant warnings.
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self._torch = torch
        self.executor = ToolExecutor()

    def _build_messages(self, question: str) -> list[dict[str, str]]:
        """构造可复现的 v1-v4 提示词。

        Build reproducible v1-v4 prompts.
        """
        schemas = json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2)
        if self.prompt_version == "v1":
            system_prompt = (
                "你是生理时序工具选择器。根据信号分析问题选择且只选择一个工具。\n"
                "只输出一个 JSON 对象，格式必须是："
                '{"name":"工具名","arguments":{}}。不要输出解释或 Markdown。\n'
                "信号文件和采样率由程序提供，不要把它们写入 arguments。\n"
                f"可用工具：\n{schemas}"
            )
        elif self.prompt_version == "v2":
            system_prompt = (
                "你是生理时序工具调用生成器。根据用户当前的问题选择且只选择一个工具。\n"
                "只输出一个合法 JSON 对象，格式为："
                '{"name":"工具名","arguments":{}}。不要输出解释、Markdown 或其他文字。\n\n'
                "参数规则（必须严格遵守）：\n"
                "1. arguments 只填写用户在当前问题中明确给出的参数和值。\n"
                "2. schema 描述中的默认值只是程序说明，用户未明确给出时必须省略。\n"
                '3. 用户没有明确指定任何参数，或要求“使用默认设置”时，arguments 必须是 {}。\n'
                "4. 信号数组、文件路径和采样率由程序提供，绝不能写入 arguments。\n"
                "5. 不得猜测、补全或创造参数。可以进行明确的单位换算，例如 500 毫秒等于 0.5 秒。\n\n"
                "示例：\n"
                '用户：找出信号中的峰。\n输出：{"name":"detect_peaks","arguments":{}}\n'
                '用户：检测峰，突出度至少为 0.6。\n输出：{"name":"detect_peaks","arguments":{"prominence":0.6}}\n'
                '用户：按默认配置进行滤波。\n输出：{"name":"filter_signal","arguments":{}}\n'
                '用户：保留 2 到 9 Hz。\n输出：{"name":"filter_signal","arguments":{"lowcut":2.0,"highcut":9.0}}\n\n'
                f"可用工具：\n{schemas}"
            )
        elif self.prompt_version == "v3":
            # v3 使用精简的模型侧工具说明，不展示默认数值，降低模型复制默认参数的倾向。
            # v3 uses a compact model-facing tool guide without defaults to reduce copying of default arguments.
            tool_guide = (
                "- calculate_statistics: 统计样本数、时长、均值、标准差、最小值、最大值；arguments 永远为 {}。\n"
                "- load_signal: 读取信号；可选参数 signal_column(string)。\n"
                "- detect_peaks: 检测峰；可选参数 min_distance_seconds(number)、prominence(number)。\n"
                "- calculate_heart_rate: 根据峰间隔计算 BPM；可选参数 min_distance_seconds(number)、prominence(number)。\n"
                "- filter_signal: 带通滤波；可选参数 lowcut(number)、highcut(number)、order(integer)。"
            )
            system_prompt = (
                "你是生理时序工具调用生成器。只选择一个最能完成用户最终目标的工具。\n\n"
                "输出协议：\n"
                '1. 输出必须且只能是一个 JSON 对象：{"name":"工具名","arguments":{}}。\n'
                '2. 顶层必须恰好包含 name 和 arguments 两个字段；arguments 必须是对象 {}，绝不能是数组 []。\n'
                "3. 不得输出解释、Markdown、候选方案、第二个 JSON 或任何前后缀文字。\n\n"
                "参数规则：\n"
                "1. 只填写用户当前问题中明确说出的参数；未说出的参数必须省略。\n"
                '2. 用户未给参数、要求默认设置，或工具没有参数时，arguments 必须是 {}。\n'
                "3. 不得猜测或补充默认值；文件路径、信号数组、采样率由程序提供。\n"
                "4. 参数和值按语义绑定：秒/毫秒间隔对应 min_distance_seconds；突出度对应 prominence；"
                "低/高截止频率对应 lowcut/highcut；阶数对应 order；列名对应 signal_column。\n"
                "5. 可以做明确单位换算，例如 500 毫秒写成 0.5 秒。\n"
                "6. 若问题提到峰只是为了最终计算 BPM/心率，选择 calculate_heart_rate，不选择 detect_peaks。\n\n"
                "示例：\n"
                '用户：报告信号均值和时长。\n输出：{"name":"calculate_statistics","arguments":{}}\n'
                '用户：按默认设置检测峰。\n输出：{"name":"detect_peaks","arguments":{}}\n'
                '用户：检测突出度至少 0.35 的峰。\n输出：{"name":"detect_peaks","arguments":{"prominence":0.35}}\n'
                '用户：用突出度至少 0.5 的峰计算 BPM。\n输出：{"name":"calculate_heart_rate","arguments":{"prominence":0.5}}\n'
                '用户：使用六阶带通滤波器。\n输出：{"name":"filter_signal","arguments":{"order":6}}\n'
                '用户：保留 0.5 到 8 Hz。\n输出：{"name":"filter_signal","arguments":{"lowcut":0.5,"highcut":8.0}}\n'
                '用户：读取默认信号列。\n输出：{"name":"load_signal","arguments":{}}\n\n'
                f"工具说明：\n{tool_guide}"
            )
        else:
            # v4 将参数生成改为“从用户原文抽取”，并加入针对 v3 错误的反例。
            # v4 reframes argument generation as extraction from user text and adds counterexamples for v3 errors.
            system_prompt = (
                "你是一个严格的生理时序工具调用编译器。先在内部判断用户的最终目标，"
                "再从用户原文抽取参数，最后只输出工具调用 JSON。\n\n"
                "允许的工具名只有以下五个，必须逐字复制，禁止创造其他名称：\n"
                "calculate_statistics | load_signal | detect_peaks | calculate_heart_rate | filter_signal\n\n"
                "工具选择：\n"
                "- 样本数、时长、均值、标准差、范围、描述性概括 -> calculate_statistics\n"
                "- 读取或加载 CSV/信号 -> load_signal\n"
                "- 最终目标是找峰或标峰 -> detect_peaks\n"
                "- 最终目标是心率或 BPM，即使提到峰 -> calculate_heart_rate\n"
                "- 最终目标是滤波或保留频段 -> filter_signal\n\n"
                "参数抽取规则：\n"
                "1. 输出中的每个数值必须来自用户原文中的数字或数词。只有明确的单位换算可以改变表示，"
                "例如 500 毫秒变为 0.5 秒。原文没有数字或数词，就不得输出任何数值参数。\n"
                "2. min_distance_seconds 只来自明确的最小峰间隔；prominence 只来自明确的突出度；"
                "lowcut/highcut 只来自明确的低/高截止频率或保留频段；order 只来自明确阶数。不得互换。\n"
                "3. signal_column 只在用户明确说“X 列”“列名 X”“column named X”或“use X as the signal column”时填写。"
                "“生理时序”“波形”“ECG 信号”等普通对象名称不能自动当作列名。\n"
                "4. 用户没明确给出的参数必须完全省略。禁止填默认值、猜测值或 null。"
                "要求默认设置时 arguments 必须是 {}。\n"
                "5. calculate_statistics 的 arguments 永远是 {}。文件路径、信号数组、采样率由程序提供。\n\n"
                "输出协议：\n"
                '- 必须且只能输出一个 JSON 对象：{"name":"工具名","arguments":{}}\n'
                "- 顶层恰好包含 name 和 arguments；arguments 必须是对象 {}，绝不能是 [] 或 null。\n"
                "- 禁止解释、Markdown、候选调用、第二个 JSON 和任何前后缀。\n\n"
                "纠错示例：\n"
                "用户：这段信号的平均心率是多少？\n"
                '错误：{"name":"calculate_heart_rate","arguments":{"min_distance_seconds":0.5}}\n'
                '正确：{"name":"calculate_heart_rate","arguments":{}}\n'
                "原因：用户没有给任何数值参数。\n\n"
                "用户：请对这段信号进行带通滤波。\n"
                '错误：{"name":"filter_signal","arguments":{"lowcut":0.5,"highcut":8.0}}\n'
                '正确：{"name":"filter_signal","arguments":{}}\n'
                "原因：用户没有给截止频率。\n\n"
                "用户：读取默认信号列。\n"
                '错误：{"name":"load_signal","arguments":{"signal_column":null}}\n'
                '正确：{"name":"load_signal","arguments":{}}\n'
                "原因：未指定列名时省略参数。\n\n"
                "用户：给出波形的描述性概括。\n"
                '正确：{"name":"calculate_statistics","arguments":{}}\n\n'
                "用户：用突出度至少 0.37 的峰计算 BPM。\n"
                '正确：{"name":"calculate_heart_rate","arguments":{"prominence":0.37}}\n\n'
                "输出前在内部检查：工具名在白名单中；每个参数都有原文依据；没有默认值；arguments 是对象；"
                "最终答案仍只能是 JSON。"
            )
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": question},
        ]

    def generate_decision(self, question: str) -> str:
        """只生成模型原文；评测时需要保留无法解析的错误输出。

        Generate raw model text only so unparsable failures remain available during evaluation.
        """
        return self.generate_messages(self._build_messages(question))

    def generate_messages(self, messages: list[dict[str, str]]) -> str:
        """对任意 messages 做确定性生成，供单工具和多步规划共用。

        Deterministically generate from arbitrary messages for both single-tool and multi-step planning.
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        num_generated_tokens = int(generated.shape[0])
        self.last_generation_info = {
            "num_generated_tokens": num_generated_tokens,
            "max_new_tokens": self.max_new_tokens,
            "reached_max_new_tokens": num_generated_tokens >= self.max_new_tokens,
        }
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def decide(self, question: str) -> tuple[ToolCall, str]:
        raw_decision = self.generate_decision(question)
        return parse_tool_call(raw_decision), raw_decision

    def run(
        self,
        question: str,
        file_path: str | Path,
        sampling_rate: float,
        signal_profile: str = "generic",
    ) -> AgentResponse:
        call, raw_decision = self.decide(question)
        return self.executor.run(
            call,
            file_path,
            sampling_rate,
            raw_decision=raw_decision,
            signal_profile=signal_profile,
        )
