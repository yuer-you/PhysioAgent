"""生成冻结 final v4 案例；不读取 SFT、DPO 或旧评测数据。

Generate frozen final-v4 cases without reading SFT, DPO, or previous evaluation data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FINAL_V4_SEED = 20260822

CATEGORY_COUNTS_V4 = {
    "single_step": 10,
    "filter_then_heart_rate": 6,
    "filter_then_peaks": 6,
    "filter_then_statistics": 6,
    "load_then_statistics": 10,
    "load_filter_heart_rate": 14,
    "load_filter_peaks": 14,
    "load_filter_statistics": 14,
}

RECORDS = ("100", "101", "200", "207")
SAFE_BANDS = [
    (0.52, 38.5),
    (0.82, 34.0),
    (1.3, 28.0),
    (1.65, 26.0),
    (2.35, 22.8),
    (3.15, 20.2),
    (4.25, 16.2),
    (4.95, 15.1),
]
STATISTICS_BANDS = [
    (0.25, 6.5),
    (0.6, 9.2),
    (1.25, 12.4),
    (2.15, 14.2),
    (3.05, 11.6),
    (4.35, 13.5),
    (0.95, 8.7),
]

LOAD_PHRASES = {
    "zh": {
        "explicit": [
            "先把存放在 signal 键下的数值纳入流程",
            "先从键 signal 抽取输入波形",
            "先将数据来源锁定到 signal",
            "先以 signal 这个键初始化输入",
            "先从 signal 对应的数据项开始",
            "先把 signal 中的序列接入工作流",
            "先指定数据键 signal 作为来源",
        ],
        "default": [
            "先让数据读取阶段自行挑选通道",
            "先从自动解析出的波形列起步",
            "先采用无需手工指定的来源",
            "先让输入模块决定使用哪一列",
            "先从软件自动发现的信号字段开始",
            "先接受文件给出的首选数据源",
            "先由读取过程确定当前通道",
        ],
    },
    "en": {
        "explicit": [
            "first materialize the trace held by key signal",
            "first pull input values from key signal",
            "first bind the workflow source to signal",
            "first initialize input with the signal key",
            "first begin from the data item named signal",
            "first connect the series under signal to the workflow",
            "first specify data key signal as the source",
        ],
        "default": [
            "first let data ingestion choose the channel",
            "first begin with the automatically resolved waveform column",
            "first use a source requiring no manual field name",
            "first let the input module decide which column to use",
            "first start from the signal field discovered by the software",
            "first accept the file-preferred data source",
            "first let the reading stage determine the active channel",
        ],
    },
}

NO_LOAD_PHRASES = {
    "zh": [
        "当前波形已在工作区，不初始化数据来源",
        "沿用已激活的序列，不执行通道读取",
        "不要选择输入键，直接处理现有信号",
        "信号已经准备完毕，省略来源步骤",
        "保持内存中的当前波形，不接入新字段",
        "从眼前序列继续，不调用数据读取",
    ],
    "en": [
        "the trace is already active, so do not initialize a source",
        "continue with the selected series without reading a channel",
        "do not choose an input key; process the existing signal directly",
        "the signal is ready, so omit the source stage",
        "keep the waveform in memory without attaching a new field",
        "continue from the trace at hand without invoking data reading",
    ],
}

ANALYSIS_PHRASES = {
    "calculate_heart_rate": {
        "zh": ["推算平均心率", "给出每分钟心搏数", "计算最终 BPM"],
        "en": ["derive mean heart rate", "give beats per minute", "calculate the final BPM"],
    },
    "detect_peaks": {
        "zh": ["定位 R 波峰", "列出心搏峰索引", "识别 ECG 峰值"],
        "en": ["locate R waves", "list heartbeat peak indices", "identify ECG peaks"],
    },
    "calculate_statistics": {
        "zh": ["总结数值统计", "报告点数、时长和均值", "概述结果序列"],
        "en": ["summarize numerical statistics", "report count, duration, and mean", "characterize the resulting series"],
    },
}

FRAMES = {
    "zh": [
        "请为这段 ECG 安排操作：{task}。",
        "本轮要求按次序完成：{task}。",
        "生成计划以实现：{task}。",
        "处理当前记录时，{task}，随后停止。",
        "只执行明确要求：{task}。",
    ],
    "en": [
        "Plan operations for this ECG to {task}.",
        "This run must complete in order: {task}.",
        "Generate a workflow to {task}.",
        "While handling the current record, {task}, then stop.",
        "Perform only the stated request: {task}.",
    ],
}


def _language(index: int) -> str:
    return "zh" if index % 2 == 0 else "en"


def _analysis_tool(category: str, index: int) -> str:
    if category.endswith("heart_rate"):
        return "calculate_heart_rate"
    if category.endswith("peaks"):
        return "detect_peaks"
    if category.endswith("statistics"):
        return "calculate_statistics"
    if category == "single_step":
        return ("calculate_statistics", "detect_peaks", "calculate_heart_rate", "filter_signal", "load_signal")[
            (index // 2) % 5
        ]
    raise ValueError(f"Unknown category: {category}")


def _analysis_phrase(tool: str, language: str, index: int) -> str:
    choices = ANALYSIS_PHRASES[tool][language]
    return choices[(index // 2) % len(choices)]


def _load_kind(category: str, index: int) -> str:
    if category == "single_step":
        return "explicit" if _language(index) == "zh" else "default"
    offsets = {
        "load_then_statistics": 1,
        "load_filter_heart_rate": 0,
        "load_filter_peaks": 1,
        "load_filter_statistics": 0,
    }
    return "default" if ((index // 2) + offsets[category]) % 2 == 0 else "explicit"


def _load_step_and_phrase(category: str, language: str, index: int) -> tuple[dict[str, Any], str, str]:
    kind = _load_kind(category, index)
    phrase = LOAD_PHRASES[language][kind][(index // 2) % len(LOAD_PHRASES[language][kind])]
    arguments = {} if kind == "default" else {"signal_column": "signal"}
    return {"name": "load_signal", "arguments": arguments}, phrase, kind


def _filter_step_and_phrase(
    language: str,
    index: int,
    *,
    statistics: bool,
) -> tuple[dict[str, Any], str]:
    if statistics and index % 7 == 0:
        return {"name": "filter_signal", "arguments": {}}, (
            "采用默认带通处理" if language == "zh" else "apply the default band-pass processing"
        )
    bands = STATISTICS_BANDS if statistics else SAFE_BANDS
    lowcut, highcut = bands[(index // 2) % len(bands)]
    arguments: dict[str, Any] = {"lowcut": lowcut, "highcut": highcut}
    if index % 4 == 0:
        arguments["order"] = 2 + ((index // 4) % 4)
    order = arguments.get("order")
    if language == "zh":
        prefix = f"用{order}阶滤波器" if order else ""
        phrase = f"{prefix}保留 {lowcut:g} 到 {highcut:g} Hz"
    else:
        suffix = f" through an order {order} filter" if order else ""
        phrase = f"retain {lowcut:g} to {highcut:g} Hz{suffix}"
    return {"name": "filter_signal", "arguments": arguments}, phrase


def _build_case(category: str, local_index: int, global_index: int) -> dict[str, Any]:
    language = _language(local_index)
    tool = _analysis_tool(category, local_index)
    analysis_step = {"name": tool, "arguments": {}}
    analysis_phrase = None if tool in {"filter_signal", "load_signal"} else _analysis_phrase(tool, language, local_index)
    load_kind = "no_load"

    if category == "single_step":
        if tool == "load_signal":
            load_step, load_phrase, load_kind = _load_step_and_phrase(category, language, local_index)
            steps = [load_step]
            task = load_phrase + ("并立即停止" if language == "zh" else " and stop immediately")
        elif tool == "filter_signal":
            filter_step, filter_phrase = _filter_step_and_phrase(language, local_index, statistics=True)
            steps = [filter_step]
            context = NO_LOAD_PHRASES[language][(local_index // 2) % len(NO_LOAD_PHRASES[language])]
            task = f"{context}，只{filter_phrase}" if language == "zh" else f"{context}; only {filter_phrase}"
        else:
            steps = [analysis_step]
            context = NO_LOAD_PHRASES[language][(local_index // 2) % len(NO_LOAD_PHRASES[language])]
            task = f"{context}，只{analysis_phrase}" if language == "zh" else f"{context}; only {analysis_phrase}"
    elif category.startswith("filter_then_"):
        filter_step, filter_phrase = _filter_step_and_phrase(
            language, local_index, statistics=tool == "calculate_statistics"
        )
        steps = [filter_step, analysis_step]
        context = NO_LOAD_PHRASES[language][(local_index // 2) % len(NO_LOAD_PHRASES[language])]
        task = (
            f"{context}，先{filter_phrase}，最后{analysis_phrase}"
            if language == "zh"
            else f"{context}; first {filter_phrase}, and finally {analysis_phrase}"
        )
    elif category == "load_then_statistics":
        load_step, load_phrase, load_kind = _load_step_and_phrase(category, language, local_index)
        steps = [load_step, analysis_step]
        task = f"{load_phrase}，然后{analysis_phrase}" if language == "zh" else f"{load_phrase}, then {analysis_phrase}"
    elif category.startswith("load_filter_"):
        load_step, load_phrase, load_kind = _load_step_and_phrase(category, language, local_index)
        filter_step, filter_phrase = _filter_step_and_phrase(
            language, local_index, statistics=tool == "calculate_statistics"
        )
        steps = [load_step, filter_step, analysis_step]
        task = (
            f"{load_phrase}，接下来{filter_phrase}，最后{analysis_phrase}"
            if language == "zh"
            else f"{load_phrase}, next {filter_phrase}, and finally {analysis_phrase}"
        )
    else:
        raise ValueError(f"Unknown category: {category}")

    frame = FRAMES[language][((local_index // 2) + list(CATEGORY_COUNTS_V4).index(category)) % len(FRAMES[language])]
    record = RECORDS[global_index % len(RECORDS)]
    prefix = f"wf_final_v4_{category.replace('calculate_', '').replace('filter_then_', 'filter_')}_{local_index + 1:03d}"
    return {
        "id": prefix,
        "category": category,
        "record": record,
        "signal_file": f"data/real/mitdb/{record}_30s/signal.csv",
        "reference_file": f"data/real/mitdb/{record}_30s/reference.json",
        "signal_profile": "ecg",
        "question": frame.format(task=task),
        "expected_steps": steps,
        "load_policy": f"load_{load_kind}" if load_kind != "no_load" else "no_load",
        "language": language,
    }


def generate_workflow_final_v4_cases() -> list[dict[str, Any]]:
    cases = []
    global_index = 0
    for category, count in CATEGORY_COUNTS_V4.items():
        for local_index in range(count):
            cases.append(_build_case(category, local_index, global_index))
            global_index += 1
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate final v4 case id.")
    if len({case["question"].strip().casefold() for case in cases}) != len(cases):
        raise ValueError("Duplicate final v4 question.")
    return cases


def write_workflow_final_v4_cases(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for case in generate_workflow_final_v4_cases():
            # 评测器不需要审计字段；策略/语言由冻结测试单元测试从标签与生成器核对。
            # Evaluators do not need audit fields; frozen-test tests verify strategy/language from labels and generators.
            stored = {key: value for key, value in case.items() if key not in {"load_policy", "language"}}
            file.write(json.dumps(stored, ensure_ascii=False, separators=(",", ":")) + "\n")
    return destination
