"""生成 Workflow SFT v2 数据，重点训练 load_signal 的释义泛化。

设计原则：
1. 训练集和验证集使用不同的加载表达模板；
2. 同时覆盖显式列名、默认列和“不需要加载”的对照任务；
3. 不读取开发集或冻结测试集，只记录它们为禁止复制的数据源。

Generate Workflow SFT v2 data with emphasis on paraphrase generalization for load_signal.
Design principles: use different loading-expression templates for training and validation; cover explicit
columns, default columns, and control tasks that need no loading; and never read development or frozen test
sets, recording them only as prohibited copy sources.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .sft_workflow_data import WORKFLOW_SFT_SYSTEM_PROMPT
from .workflow import parse_workflow_plan


WORKFLOW_SFT_V2_SEED = 20260822

CATEGORY_COUNTS_V2 = {
    "train": {
        "single_step": 240,
        "filter_then_heart_rate": 180,
        "filter_then_peaks": 180,
        "filter_then_statistics": 180,
        "load_then_statistics": 180,
        "load_filter_heart_rate": 240,
        "load_filter_peaks": 240,
        "load_filter_statistics": 360,
    },
    "validation": {
        "single_step": 40,
        "filter_then_heart_rate": 30,
        "filter_then_peaks": 30,
        "filter_then_statistics": 30,
        "load_then_statistics": 30,
        "load_filter_heart_rate": 40,
        "load_filter_peaks": 40,
        "load_filter_statistics": 60,
    },
}

# 列名也按 split 隔离，验证模型能否复制未在训练集中出现的合法列名。
# Isolate column names by split to test copying of valid names unseen during training.
COLUMNS_V2 = {
    "train": [
        "signal",
        "MLII",
        "ECG_I",
        "lead_II",
        "wave_raw",
        "channel_A",
        "trace_main",
        "ecg_mv",
    ],
    "validation": ["lead_V1", "recording", "primary_signal", "sensor_ecg"],
}

FILTER_BANDS_V2 = {
    "train": [
        (0.4, 40.0),
        (0.8, 32.0),
        (1.0, 30.0),
        (1.5, 28.0),
        (2.0, 25.0),
        (3.0, 22.0),
        (4.0, 20.0),
        (5.0, 16.0),
    ],
    "validation": [
        (0.6, 38.0),
        (0.9, 33.0),
        (1.2, 27.0),
        (2.5, 19.0),
        (3.2, 23.0),
        (4.5, 15.0),
    ],
}

# 每个模板都有稳定 id，便于测试 train/validation 的表达族确实隔离。
# Give every template a stable ID so tests can verify train/validation expression-family isolation.
LOAD_TEMPLATES_V2 = {
    "train": {
        "zh": {
            "explicit": [
                ("train_zh_explicit_read", "从 {column} 列读取波形"),
                ("train_zh_explicit_load", "把 {column} 字段中的序列加载进来"),
                ("train_zh_explicit_import", "读入通道 {column} 的数据"),
                ("train_zh_explicit_fetch", "导入字段 {column} 的数值"),
                ("train_zh_explicit_retrieve", "取出 {column} 列作为当前信号"),
            ],
            "default": [
                ("train_zh_default_read", "读取程序默认选择的信号列"),
                ("train_zh_default_load", "按默认列加载波形"),
                ("train_zh_default_import", "读入标准信号通道"),
                ("train_zh_default_fetch", "导入系统预设通道"),
                ("train_zh_default_retrieve", "取出默认字段中的序列"),
            ],
        },
        "en": {
            "explicit": [
                ("train_en_explicit_read", "read the waveform from column {column}"),
                ("train_en_explicit_load", "load the series stored in field {column}"),
                ("train_en_explicit_import", "import channel {column}"),
                ("train_en_explicit_fetch", "fetch the values from column {column}"),
                ("train_en_explicit_retrieve", "retrieve field {column} as the active signal"),
            ],
            "default": [
                ("train_en_default_read", "read the program-selected signal column"),
                ("train_en_default_load", "load the waveform from the default column"),
                ("train_en_default_import", "import the standard signal channel"),
                ("train_en_default_fetch", "fetch the system-preset channel"),
                ("train_en_default_retrieve", "retrieve the series from the default field"),
            ],
        },
    },
    "validation": {
        "zh": {
            "explicit": [
                ("validation_zh_explicit_open", "打开 {column} 这一列的波形"),
                ("validation_zh_explicit_obtain", "取得字段 {column} 中的序列"),
                ("validation_zh_explicit_bring", "调入 {column} 通道的数据"),
                ("validation_zh_explicit_access", "访问 {column} 列并把它作为输入"),
            ],
            "default": [
                ("validation_zh_default_open", "打开通常使用的信号列"),
                ("validation_zh_default_obtain", "取得系统选择的通道"),
                ("validation_zh_default_bring", "调入常规输入字段"),
                ("validation_zh_default_access", "访问预设列中的波形"),
            ],
        },
        "en": {
            "explicit": [
                ("validation_en_explicit_open", "open the waveform held in column {column}"),
                ("validation_en_explicit_obtain", "obtain the series from field {column}"),
                ("validation_en_explicit_bring", "bring in channel {column}"),
                ("validation_en_explicit_access", "access column {column} as the input trace"),
            ],
            "default": [
                ("validation_en_default_open", "open the usually selected signal column"),
                ("validation_en_default_obtain", "obtain the system-selected channel"),
                ("validation_en_default_bring", "bring in the regular input field"),
                ("validation_en_default_access", "access the preset column waveform"),
            ],
        },
    },
}

NO_LOAD_CONTEXT_V2 = {
    "train": {
        "zh": [
            "直接处理当前已经提供的波形",
            "不要重新读取任何列，使用现有信号",
            "基于内存中的当前序列",
            "无需加载字段，直接对输入波形操作",
        ],
        "en": [
            "work directly on the waveform already provided",
            "do not reload a column; use the current signal",
            "use the series currently in memory",
            "without loading a field, operate on the input trace",
        ],
    },
    "validation": {
        "zh": [
            "保持现有输入，不执行读取步骤",
            "从当前可用序列直接开始",
            "省略加载动作，只处理眼前波形",
        ],
        "en": [
            "keep the present input and skip any read step",
            "start directly from the currently available series",
            "omit loading and process only the trace at hand",
        ],
    },
}

DEDUP_SUFFIXES_V2 = {
    "zh": [
        "严格保持上述顺序",
        "不要省略任何明确动作",
        "只返回所需工具计划",
        "完成指定目标后立即停止",
        "不要增加计划外步骤",
        "确保每项要求都出现在计划中",
    ],
    "en": [
        "preserving that exact order",
        "without omitting an explicit operation",
        "returning only the required tool plan",
        "stopping immediately after the stated goal",
        "without adding an unrequested step",
        "ensuring every stated operation appears in the plan",
    ],
}

ZH_ORDERS = {2: "二阶", 3: "三阶", 4: "四阶", 5: "五阶"}
EN_ORDERS = {2: "second-order", 3: "third-order", 4: "fourth-order", 5: "fifth-order"}


def _compact_plan(steps: list[dict[str, Any]]) -> str:
    return json.dumps({"steps": steps}, ensure_ascii=False, separators=(",", ":"))


def _language(index: int) -> str:
    return "zh" if index % 2 == 0 else "en"


def _load_spec(split: str, language: str, index: int) -> tuple[dict[str, Any], str, str, str]:
    # 相邻的中英文样例共享 default/explicit 类型，避免语言与参数类型相关。
    # Adjacent Chinese/English examples share default/explicit types to avoid language-argument correlation.
    load_kind = "default" if (index // 2) % 2 == 0 else "explicit"
    templates = LOAD_TEMPLATES_V2[split][language][load_kind]
    template_id, template = templates[(index // 4) % len(templates)]
    if load_kind == "default":
        arguments: dict[str, Any] = {}
        phrase = template
    else:
        column = COLUMNS_V2[split][(index // 4) % len(COLUMNS_V2[split])]
        arguments = {"signal_column": column}
        phrase = template.format(column=column)
    return arguments, phrase, template_id, load_kind


def _filter_spec(
    split: str,
    language: str,
    index: int,
    *,
    allow_default: bool,
) -> tuple[dict[str, Any], str]:
    # 默认滤波器是 0.5-8 Hz，不能接需要完整 5-15 Hz 的冻结 ECG 峰检测器。
    # 因此默认参数只用于“滤波后统计”或单独滤波，不生成无效的峰值/心率链路。
    # The 0.5-8 Hz default filter cannot precede the frozen ECG detector, which needs the full 5-15 Hz band.
    # Therefore defaults are used only for post-filter statistics or standalone filtering, not invalid peak/HR chains.
    if allow_default and index % 7 == 0:
        arguments: dict[str, Any] = {}
        phrase = "采用默认带通设置" if language == "zh" else "apply the default band-pass settings"
        return arguments, phrase
    lowcut, highcut = FILTER_BANDS_V2[split][(index // 2) % len(FILTER_BANDS_V2[split])]
    arguments = {"lowcut": lowcut, "highcut": highcut}
    order = 2 + ((index // 3) % 4) if index % 3 == 0 else None
    if order is not None:
        arguments["order"] = order
    if language == "zh":
        order_text = f"使用{ZH_ORDERS[order]}滤波器，" if order else ""
        phrase = f"{order_text}保留 {lowcut:g} 到 {highcut:g} Hz"
    else:
        order_text = f" through a {EN_ORDERS[order]} filter" if order else ""
        phrase = f"retain {lowcut:g} to {highcut:g} Hz{order_text}"
    return arguments, phrase


def _no_load_context(split: str, language: str, index: int) -> str:
    choices = NO_LOAD_CONTEXT_V2[split][language]
    return choices[(index // 2) % len(choices)]


def _decorate(task: str, split: str, language: str, category: str, index: int) -> str:
    if split == "train":
        zh_frames = [
            "针对当前记录，{task}。",
            "请按以下要求生成计划：{task}。",
            "本轮操作需要依次做到：{task}。",
            "处理这份 ECG 时，{task}，完成后停止。",
        ]
        en_frames = [
            "For the current record, {task}.",
            "Build a plan to {task}.",
            "The required operations are to {task}.",
            "While processing this ECG, {task}, then stop.",
        ]
    else:
        zh_frames = [
            "为这段数据规划操作：{task}。",
            "当前请求是{task}。",
            "不要跳过明确步骤：{task}。",
        ]
        en_frames = [
            "Plan the operations needed to {task}.",
            "The current request is to {task}.",
            "Do not skip an explicit step: {task}.",
        ]
    frames = zh_frames if language == "zh" else en_frames
    category_offset = list(CATEGORY_COUNTS_V2[split]).index(category)
    return frames[((index // 2) + category_offset) % len(frames)].format(task=task)


def _analysis_phrase(tool: str, language: str, split: str, index: int) -> str:
    train_phrases = {
        "calculate_statistics": {
            "zh": [
                "汇总样本数、时长和描述性统计",
                "报告序列长度、持续时间与均值",
                "计算处理结果的基础统计量",
                "给出观测点数量、记录时长和统计摘要",
                "概括输出序列的规模与数值分布",
                "统计最终波形的长度、时间和均值",
                "返回处理后数据的描述性指标",
            ],
            "en": [
                "summarize sample count, duration, and descriptive statistics",
                "report sequence length, elapsed time, and mean",
                "calculate basic statistics for the processed result",
                "give observation count, record duration, and a statistical summary",
                "describe the output series size and value distribution",
                "measure the final waveform length, time span, and mean",
                "return descriptive indicators for the resulting data",
            ],
        },
        "detect_peaks": {
            "zh": [
                "定位 ECG R 峰",
                "找出心搏峰的位置",
                "检测处理结果中的 R 波峰",
                "返回 ECG 峰值索引",
                "标记波形中的心搏峰",
                "识别最终序列的 R 峰",
                "列出检测到的峰位置",
            ],
            "en": [
                "locate the ECG R peaks",
                "find the heartbeat peak positions",
                "detect R waves in the processed result",
                "return the ECG peak indices",
                "mark heartbeat peaks in the waveform",
                "identify R peaks in the final series",
                "list the detected peak locations",
            ],
        },
        "calculate_heart_rate": {
            "zh": [
                "估计平均心率",
                "计算每分钟平均心搏数",
                "报告处理结果的 BPM",
                "得到这段 ECG 的平均心率",
                "返回最终波形的心率估计",
                "计算平均每分钟节律次数",
                "给出基于峰间隔的心率",
            ],
            "en": [
                "estimate mean heart rate",
                "calculate average beats per minute",
                "report BPM for the processed result",
                "derive the mean heart rate of this ECG",
                "return a heart-rate estimate for the final waveform",
                "compute the average rhythm count per minute",
                "give heart rate based on peak intervals",
            ],
        },
    }
    validation_phrases = {
        "calculate_statistics": {
            "zh": ["概述最终数据的长度与统计特征", "量化输出的点数、时长及均值", "为结果生成描述性摘要", "报告处理后序列的基本指标", "总结所得波形"],
            "en": ["characterize the final data length and statistics", "quantify output count, duration, and mean", "produce a descriptive summary of the result", "report basic indicators of the processed series", "summarize the resulting waveform"],
        },
        "detect_peaks": {
            "zh": ["指出最终 ECG 的 R 峰", "确定心搏峰所在位置", "在结果中寻找峰值索引", "识别处理波形的 R 波", "输出检测出的 ECG 峰"],
            "en": ["point out R peaks in the final ECG", "determine heartbeat peak positions", "find peak indices in the result", "recognize R waves in the processed trace", "output the detected ECG peaks"],
        },
        "calculate_heart_rate": {
            "zh": ["推算最终平均 BPM", "求得处理后 ECG 的心率", "报告平均每分钟心搏数", "估算结果波形的心率", "返回平均节律速度"],
            "en": ["derive the final average BPM", "obtain heart rate for the processed ECG", "report average beats per minute", "estimate heart rate from the resulting trace", "return the mean rhythm rate"],
        },
    }
    phrases = train_phrases if split == "train" else validation_phrases
    choices = phrases[tool][language]
    return choices[(index // 8) % len(choices)]


def _build_example(
    split: str,
    category: str,
    index: int,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    language = _language(index)
    load_arguments, load_phrase, load_template_id, load_kind = _load_spec(split, language, index)
    filter_arguments, filter_phrase = _filter_spec(
        split,
        language,
        index,
        allow_default=category == "single_step" or category.endswith("statistics"),
    )
    load_step = {"name": "load_signal", "arguments": load_arguments}
    filter_step = {"name": "filter_signal", "arguments": filter_arguments}
    tool_steps = {
        "calculate_statistics": {"name": "calculate_statistics", "arguments": {}},
        "detect_peaks": {"name": "detect_peaks", "arguments": {}},
        "calculate_heart_rate": {"name": "calculate_heart_rate", "arguments": {}},
    }
    extra = {
        "language": language,
        "load_template_id": None,
        "load_policy": "no_load",
        "column_kind": "not_applicable",
    }

    if category == "single_step":
        tools = [
            "calculate_statistics",
            "load_signal",
            "filter_signal",
            "detect_peaks",
            "calculate_heart_rate",
        ]
        tool = tools[(index // 2) % len(tools)]
        if tool == "load_signal":
            steps = [load_step]
            task = load_phrase + ("并停止" if language == "zh" else " and stop")
            extra.update(
                load_template_id=load_template_id,
                load_policy=f"load_{load_kind}",
                column_kind=load_kind,
            )
        elif tool == "filter_signal":
            steps = [filter_step]
            context = _no_load_context(split, language, index)
            task = f"{context}，{filter_phrase}" if language == "zh" else f"{context} and {filter_phrase}"
        else:
            steps = [tool_steps[tool]]
            context = _no_load_context(split, language, index)
            analysis = _analysis_phrase(tool, language, split, index)
            task = f"{context}，只{analysis}" if language == "zh" else f"{context} and only {analysis}"
        return steps, task, extra

    if category.startswith("load_"):
        extra.update(
            load_template_id=load_template_id,
            load_policy=f"load_{load_kind}",
            column_kind=load_kind,
        )
        prefix = load_phrase
    else:
        prefix = _no_load_context(split, language, index)

    if category.endswith("statistics"):
        final_tool = "calculate_statistics"
    elif category.endswith("peaks"):
        final_tool = "detect_peaks"
    elif category.endswith("heart_rate"):
        final_tool = "calculate_heart_rate"
    else:
        raise ValueError(f"Unknown workflow category: {category}")
    final_step = tool_steps[final_tool]
    analysis = _analysis_phrase(final_tool, language, split, index)

    if category == "load_then_statistics":
        steps = [load_step, final_step]
        task = f"{prefix}，随后{analysis}" if language == "zh" else f"{prefix}, then {analysis}"
    elif category.startswith("load_filter_"):
        steps = [load_step, filter_step, final_step]
        task = (
            f"{prefix}，接着{filter_phrase}，最后{analysis}"
            if language == "zh"
            else f"{prefix}, next {filter_phrase}, and finally {analysis}"
        )
    elif category.startswith("filter_then_"):
        steps = [filter_step, final_step]
        task = (
            f"{prefix}，先{filter_phrase}，再{analysis}"
            if language == "zh"
            else f"{prefix}; first {filter_phrase}, then {analysis}"
        )
    else:
        raise ValueError(f"Unknown workflow category: {category}")
    return steps, task, extra


def _make_rows(split: str, category: str, count: int) -> list[dict[str, Any]]:
    rows = []
    question_occurrences: Counter[str] = Counter()
    for index in range(count):
        steps, task, extra = _build_example(split, category, index)
        completion = _compact_plan(steps)
        parsed = parse_workflow_plan(completion)
        parsed_steps = [{"name": step.name, "arguments": step.arguments} for step in parsed]
        if parsed_steps != steps:
            raise ValueError("Generated Workflow SFT v2 completion does not match its plan.")
        question = _decorate(task, split, extra["language"], category, index)
        normalized = question.strip().casefold()
        occurrence = question_occurrences[normalized]
        question_occurrences[normalized] += 1
        if occurrence:
            suffixes = DEDUP_SUFFIXES_V2[extra["language"]]
            if occurrence > len(suffixes):
                raise ValueError(f"Too many repeated natural-language templates in {split}/{category}.")
            suffix = suffixes[occurrence - 1]
            if extra["language"] == "zh":
                question = question.rstrip("。") + f"，{suffix}。"
            else:
                question = question.rstrip(".") + f", {suffix}."
        rows.append(
            {
                "id": f"workflow_sft_v2_{split}_{category}_{index + 1:04d}",
                "prompt": [
                    {"role": "system", "content": WORKFLOW_SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "completion": [{"role": "assistant", "content": completion}],
                "metadata": {
                    "task_type": "workflow",
                    "split": split,
                    "category": category,
                    "step_count": len(steps),
                    "final_tool": steps[-1]["name"],
                    "language": extra["language"],
                    "signal_profile": "ecg",
                    "expected_steps": steps,
                    "load_policy": extra["load_policy"],
                    "load_template_id": extra["load_template_id"],
                    "column_kind": extra["column_kind"],
                    "paraphrase_split": f"{split}_only",
                    "dedup_variant": occurrence,
                    "source": "synthetic_workflow_templates_v2",
                    "version": "workflow_sft_v2",
                },
            }
        )
    return rows


def generate_workflow_sft_v2_datasets(
    seed: int = WORKFLOW_SFT_V2_SEED,
) -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(seed)
    for split, counts in CATEGORY_COUNTS_V2.items():
        rows = []
        for category, count in counts.items():
            rows.extend(_make_rows(split, category, count))
        rng.shuffle(rows)
        datasets[split] = rows
    validate_workflow_sft_v2_datasets(datasets)
    return datasets


def validate_workflow_sft_v2_datasets(datasets: dict[str, list[dict[str, Any]]]) -> None:
    if set(datasets) != {"train", "validation"}:
        raise ValueError("Workflow SFT v2 must contain only train and validation splits.")
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    template_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split, rows in datasets.items():
        expected_total = sum(CATEGORY_COUNTS_V2[split].values())
        if len(rows) != expected_total:
            raise ValueError(f"{split}: expected {expected_total} rows, got {len(rows)}.")
        if Counter(row["metadata"]["category"] for row in rows) != Counter(CATEGORY_COUNTS_V2[split]):
            raise ValueError(f"{split}: wrong Workflow SFT v2 category distribution.")
        for row in rows:
            row_id = row["id"]
            if row_id in seen_ids:
                raise ValueError(f"Duplicate id: {row_id}")
            seen_ids.add(row_id)
            question = row["prompt"][1]["content"].strip().casefold()
            if question in seen_questions:
                raise ValueError(f"Question leakage across Workflow SFT v2 splits: {question}")
            seen_questions.add(question)
            metadata = row["metadata"]
            if metadata["paraphrase_split"] != f"{split}_only":
                raise ValueError(f"Wrong paraphrase split: {row_id}")
            template_id = metadata["load_template_id"]
            if template_id is not None:
                template_ids[split].add(template_id)
            parsed = parse_workflow_plan(row["completion"][0]["content"])
            parsed_steps = [{"name": step.name, "arguments": step.arguments} for step in parsed]
            if parsed_steps != metadata["expected_steps"]:
                raise ValueError(f"Completion and metadata disagree: {row_id}")
            has_load = any(step["name"] == "load_signal" for step in parsed_steps)
            if has_load != metadata["load_policy"].startswith("load_"):
                raise ValueError(f"load_policy and plan disagree: {row_id}")
    if template_ids["train"] & template_ids["validation"]:
        raise ValueError("Train and validation load template ids overlap.")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_workflow_sft_v2_datasets(
    output_dir: str | Path,
    seed: int = WORKFLOW_SFT_V2_SEED,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = generate_workflow_sft_v2_datasets(seed=seed)
    paths: dict[str, Path] = {}
    for split, rows in datasets.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split] = path

    manifest = {
        "version": "workflow_sft_v2",
        "seed": seed,
        "system_prompt": "workflow_prompt_v2_with_ecg_profile",
        "design_goal": "load_signal paraphrase generalization with split-isolated phrase families",
        "split_sizes": {split: len(rows) for split, rows in datasets.items()},
        "category_counts": CATEGORY_COUNTS_V2,
        "step_counts": {
            split: dict(sorted(Counter(row["metadata"]["step_count"] for row in rows).items()))
            for split, rows in datasets.items()
        },
        "load_policy_counts": {
            split: dict(sorted(Counter(row["metadata"]["load_policy"] for row in rows).items()))
            for split, rows in datasets.items()
        },
        "columns": COLUMNS_V2,
        "paraphrase_policy": {
            "train": "train-only load templates",
            "validation": "validation-only held-out load templates",
        },
        "file_sha256": {split: _file_sha256(path) for split, path in paths.items()},
        "frozen_tests_read_by_generator": False,
        "diagnostic_inspiration": "final v2 error taxonomy only; no final question is read or copied",
        "forbidden_training_sources": [
            "evaluation/workflow_planning_cases_v1.jsonl",
            "evaluation/workflow_final_cases_v1.jsonl",
            "evaluation/workflow_final_cases_v2.jsonl",
        ],
        "test_split_created": False,
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return paths
