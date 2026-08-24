"""生成多步 Workflow SFT v1 数据；不读取任何冻结工作流测试集。

Generate multi-step Workflow SFT v1 data without reading any frozen workflow test set.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .workflow import parse_workflow_plan
from .workflow_model import WORKFLOW_SYSTEM_PROMPT_V2


WORKFLOW_SFT_SEED = 20260821
WORKFLOW_SFT_SYSTEM_PROMPT = WORKFLOW_SYSTEM_PROMPT_V2 + "\n当前 signal_profile：ecg"

CATEGORY_COUNTS = {
    "train": {
        "single_step": 200,
        "filter_then_heart_rate": 120,
        "filter_then_peaks": 120,
        "filter_then_statistics": 120,
        "load_then_statistics": 120,
        "load_filter_heart_rate": 160,
        "load_filter_peaks": 120,
        "load_filter_statistics": 240,
    },
    "validation": {
        "single_step": 40,
        "filter_then_heart_rate": 20,
        "filter_then_peaks": 20,
        "filter_then_statistics": 20,
        "load_then_statistics": 20,
        "load_filter_heart_rate": 25,
        "load_filter_peaks": 15,
        "load_filter_statistics": 40,
    },
}

FILTER_BANDS = [
    (0.4, 40.0),
    (0.8, 32.0),
    (1.0, 30.0),
    (1.5, 28.0),
    (2.0, 25.0),
    (3.0, 22.0),
    (4.0, 20.0),
    (5.0, 16.0),
]
STATISTICS_BANDS = [(0.3, 8.0), (0.5, 12.0), (1.0, 18.0), (2.0, 30.0), (4.0, 45.0)]
COLUMNS = [
    "signal",
    "MLII",
    "ECG_I",
    "lead_II",
    "wave_raw",
    "channel_A",
    "trace_main",
    "ecg_mv",
    "primary_signal",
    "sensor_ecg",
    "lead_V1",
    "recording",
]
ZH_ORDERS = {2: "二阶", 3: "三阶", 4: "四阶", 5: "五阶"}
EN_ORDERS = {2: "second-order", 3: "third-order", 4: "fourth-order", 5: "fifth-order"}

CONTEXT = {
    "train": {
        "zh": {
            "prefix": [
                "针对当前记录，",
                "在本次分析中，",
                "对于这段波形，",
                "按用户给定流程，",
                "处理当前生理序列时，",
                "面对这份输入，",
                "在保持步骤顺序的情况下，",
                "对现有信号，",
            ],
            "frame": [
                "请完成以下任务：{task}",
                "严格依次执行：{task}",
                "需要按顺序做到：{task}",
                "把操作组织为：{task}",
                "请生成能够实现这个目标的计划：{task}",
            ],
            "suffix": ["。", "，完成最后一步后停止。", "，不要添加题目之外的操作。"],
        },
        "en": {
            "prefix": [
                "For the current record, ",
                "In this analysis, ",
                "For this waveform, ",
                "Following the requested procedure, ",
                "While handling the present series, ",
                "Given this input, ",
                "Keeping the stated order, ",
                "For the available signal, ",
            ],
            "frame": [
                "complete this task: {task}",
                "perform these operations in order: {task}",
                "build a plan that will {task}",
                "the required procedure is to {task}",
                "carry out exactly this workflow: {task}",
            ],
            "suffix": [".", ", then stop after the final operation.", ", without adding another operation."],
        },
    },
    "validation": {
        "zh": {
            "prefix": ["请为这份数据规划操作：", "当前请求是：", "对输入波形，", "本轮只需要做到："],
            "frame": ["{task}", "按次序完成{task}", "生成计划以便{task}", "依照要求{task}", "不要跳步地完成{task}"],
            "suffix": ["。", "，结束后不要继续调用工具。"],
        },
        "en": {
            "prefix": ["Plan operations for this data: ", "The current request is to ", "For the input trace, ", "This run only needs to ",],
            "frame": ["{task}", "complete in sequence: {task}", "produce a plan to {task}", "follow the request and {task}", "without skipping a step, {task}"],
            "suffix": [".", ", and make no further tool call."],
        },
    },
}


def _compact_plan(steps: list[dict[str, Any]]) -> str:
    return json.dumps({"steps": steps}, ensure_ascii=False, separators=(",", ":"))


def _decorate(task: str, split: str, language: str, index: int) -> str:
    choices = CONTEXT[split][language]
    local_index = index // 2
    prefixes = choices["prefix"]
    frames = choices["frame"]
    suffixes = choices["suffix"]
    prefix = prefixes[local_index % len(prefixes)]
    frame = frames[(local_index // len(prefixes)) % len(frames)]
    suffix = suffixes[(local_index // (len(prefixes) * len(frames))) % len(suffixes)]
    return prefix + frame.format(task=task) + suffix


def _filter_arguments(index: int, *, statistics: bool = False) -> dict[str, Any]:
    if statistics and index % 6 == 0:
        return {}
    bands = STATISTICS_BANDS if statistics else FILTER_BANDS
    lowcut, highcut = bands[(index // 2) % len(bands)]
    arguments: dict[str, Any] = {"lowcut": lowcut, "highcut": highcut}
    if index % 3 == 0:
        arguments["order"] = 2 + ((index // 3) % 4)
    return arguments


def _filter_phrase(arguments: dict[str, Any], language: str) -> str:
    if not arguments:
        return "按默认设置进行带通滤波" if language == "zh" else "apply the default band-pass filter"
    lowcut = arguments["lowcut"]
    highcut = arguments["highcut"]
    order = arguments.get("order")
    if language == "zh":
        order_text = f"用{ZH_ORDERS[order]}" if order else ""
        return f"{order_text}保留 {lowcut:g} 到 {highcut:g} Hz"
    order_text = f" with a {EN_ORDERS[order]} design" if order else ""
    return f"retain {lowcut:g} to {highcut:g} Hz{order_text}"


def _load_arguments(index: int) -> dict[str, Any]:
    if index % 3 == 0:
        return {}
    return {"signal_column": COLUMNS[(index // 2) % len(COLUMNS)]}


def _load_phrase(arguments: dict[str, Any], language: str) -> str:
    column = arguments.get("signal_column")
    if language == "zh":
        return f"读取名为 {column} 的列" if column else "读取默认信号列"
    return f"load the column named {column}" if column else "load the default signal column"


def _build_steps_and_task(
    category: str,
    index: int,
    language: str,
) -> tuple[list[dict[str, Any]], str, str]:
    if category == "single_step":
        tools = [
            "calculate_statistics",
            "load_signal",
            "filter_signal",
            "detect_peaks",
            "calculate_heart_rate",
        ]
        tool = tools[(index // 2) % len(tools)]
        if tool == "calculate_statistics":
            steps = [{"name": tool, "arguments": {}}]
            task = "只汇总样本数、时长和描述性统计" if language == "zh" else "report only sample count, duration, and descriptive statistics"
        elif tool == "load_signal":
            arguments = _load_arguments(index)
            steps = [{"name": tool, "arguments": arguments}]
            task = (_load_phrase(arguments, language) + ("并停止" if language == "zh" else " and stop"))
        elif tool == "filter_signal":
            arguments = _filter_arguments(index, statistics=True)
            steps = [{"name": tool, "arguments": arguments}]
            task = (_filter_phrase(arguments, language) + ("，不做后续分析" if language == "zh" else " without a later analysis"))
        elif tool == "detect_peaks":
            steps = [{"name": tool, "arguments": {}}]
            task = "只定位 ECG R 峰" if language == "zh" else "only locate the ECG R peaks"
        else:
            steps = [{"name": tool, "arguments": {}}]
            task = "只计算平均心率" if language == "zh" else "only calculate mean heart rate"
        return steps, task, tool

    load_arguments = _load_arguments(index)
    filter_arguments = _filter_arguments(index, statistics="statistics" in category)
    load_phrase = _load_phrase(load_arguments, language)
    filter_phrase = _filter_phrase(filter_arguments, language)
    load_step = {"name": "load_signal", "arguments": load_arguments}
    filter_step = {"name": "filter_signal", "arguments": filter_arguments}
    stats_step = {"name": "calculate_statistics", "arguments": {}}
    peaks_step = {"name": "detect_peaks", "arguments": {}}
    heart_step = {"name": "calculate_heart_rate", "arguments": {}}

    if category == "load_then_statistics":
        steps = [load_step, stats_step]
        task = f"{load_phrase}，再计算其统计量" if language == "zh" else f"{load_phrase}, then summarize its statistics"
    elif category == "filter_then_statistics":
        steps = [filter_step, stats_step]
        task = f"先{filter_phrase}，再统计处理后的数据" if language == "zh" else f"first {filter_phrase}, then summarize the processed data"
    elif category == "filter_then_peaks":
        steps = [filter_step, peaks_step]
        task = f"先{filter_phrase}，再检测 R 峰" if language == "zh" else f"first {filter_phrase}, then detect R peaks"
    elif category == "filter_then_heart_rate":
        steps = [filter_step, heart_step]
        task = f"先{filter_phrase}，再估计平均心率" if language == "zh" else f"first {filter_phrase}, then estimate mean heart rate"
    elif category == "load_filter_statistics":
        steps = [load_step, filter_step, stats_step]
        task = f"先{load_phrase}，随后{filter_phrase}，最后统计处理结果" if language == "zh" else f"first {load_phrase}, next {filter_phrase}, and finally summarize the result"
    elif category == "load_filter_peaks":
        steps = [load_step, filter_step, peaks_step]
        task = f"先{load_phrase}，随后{filter_phrase}，最后找出 R 峰" if language == "zh" else f"first {load_phrase}, next {filter_phrase}, and finally locate R peaks"
    elif category == "load_filter_heart_rate":
        steps = [load_step, filter_step, heart_step]
        task = f"先{load_phrase}，随后{filter_phrase}，最后计算心率" if language == "zh" else f"first {load_phrase}, next {filter_phrase}, and finally calculate heart rate"
    else:
        raise ValueError(f"Unknown workflow SFT category: {category}")
    return steps, task, steps[-1]["name"]


def _make_rows(split: str, category: str, count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        language = "zh" if index % 2 == 0 else "en"
        steps, task, final_tool = _build_steps_and_task(category, index, language)
        completion = _compact_plan(steps)
        parsed = parse_workflow_plan(completion)
        if len(parsed) != len(steps):
            raise ValueError("Generated workflow plan lost a step during validation.")
        rows.append(
            {
                "id": f"workflow_sft_v1_{split}_{category}_{index + 1:04d}",
                "prompt": [
                    {"role": "system", "content": WORKFLOW_SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": _decorate(task, split, language, index)},
                ],
                "completion": [{"role": "assistant", "content": completion}],
                "metadata": {
                    "task_type": "workflow",
                    "split": split,
                    "category": category,
                    "step_count": len(steps),
                    "final_tool": final_tool,
                    "language": language,
                    "signal_profile": "ecg",
                    "expected_steps": steps,
                    "source": "synthetic_workflow_templates_v1",
                    "version": "workflow_sft_v1",
                },
            }
        )
    return rows


def generate_workflow_sft_datasets(seed: int = WORKFLOW_SFT_SEED) -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(seed)
    for split, counts in CATEGORY_COUNTS.items():
        rows = []
        for category, count in counts.items():
            rows.extend(_make_rows(split, category, count))
        rng.shuffle(rows)
        datasets[split] = rows
    validate_workflow_sft_datasets(datasets)
    return datasets


def validate_workflow_sft_datasets(datasets: dict[str, list[dict[str, Any]]]) -> None:
    if set(datasets) != {"train", "validation"}:
        raise ValueError("Workflow SFT must contain only train and validation splits.")
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    expected_totals = {split: sum(counts.values()) for split, counts in CATEGORY_COUNTS.items()}
    for split, rows in datasets.items():
        if len(rows) != expected_totals[split]:
            raise ValueError(f"{split}: expected {expected_totals[split]} rows, got {len(rows)}.")
        category_counts = Counter(row["metadata"]["category"] for row in rows)
        if category_counts != Counter(CATEGORY_COUNTS[split]):
            raise ValueError(f"{split}: wrong category distribution: {category_counts}")
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen_ids.add(row["id"])
            question = row["prompt"][1]["content"].strip().casefold()
            if question in seen_questions:
                raise ValueError(f"Question leakage across workflow SFT splits: {question}")
            seen_questions.add(question)
            if row["prompt"][0]["content"] != WORKFLOW_SFT_SYSTEM_PROMPT:
                raise ValueError(f"Wrong workflow system prompt: {row['id']}")
            parsed = parse_workflow_plan(row["completion"][0]["content"])
            parsed_steps = [{"name": step.name, "arguments": step.arguments} for step in parsed]
            if parsed_steps != row["metadata"]["expected_steps"]:
                raise ValueError(f"Completion and metadata disagree: {row['id']}")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_workflow_sft_datasets(
    output_dir: str | Path,
    seed: int = WORKFLOW_SFT_SEED,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = generate_workflow_sft_datasets(seed=seed)
    paths: dict[str, Path] = {}
    for split, rows in datasets.items():
        path = output / f"{split}.jsonl"
        # 固定使用 LF，确保 Windows 本地与 Linux 服务器生成完全相同的字节和 SHA-256。
        # Force LF so Windows and Linux generate identical bytes and SHA-256 hashes.
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split] = path
    manifest = {
        "version": "workflow_sft_v1",
        "seed": seed,
        "system_prompt": "workflow_prompt_v2_with_ecg_profile",
        "split_sizes": {split: len(rows) for split, rows in datasets.items()},
        "category_counts": CATEGORY_COUNTS,
        "step_counts": {
            split: dict(sorted(Counter(row["metadata"]["step_count"] for row in rows).items()))
            for split, rows in datasets.items()
        },
        "file_sha256": {split: _file_sha256(path) for split, path in paths.items()},
        "frozen_tests_read_by_generator": False,
        "forbidden_training_sources": [
            "evaluation/workflow_final_cases_v1.jsonl",
            "evaluation/workflow_planning_cases_v1.jsonl",
        ],
        "test_split_created": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths
