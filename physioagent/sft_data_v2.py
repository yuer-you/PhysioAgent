"""生成 SFT v2：复用 v1 能力，并定向增强规范参数名与自然语言参数表达。

Generate SFT v2 by retaining v1 capabilities while targeting canonical argument names and natural phrasing.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .agent import parse_tool_call
from .sft_data import TOOL_NAMES, generate_datasets


SFT_SYSTEM_PROMPT_V2 = (
    "你是生理时序工具调用助手。只输出一个 JSON 对象，顶层恰好包含 name 和 arguments。"
    "合法工具与参数如下：calculate_statistics 只能用 {}；load_signal 只能用 signal_column；"
    "detect_peaks 和 calculate_heart_rate 只能用 min_distance_seconds、prominence；"
    "filter_signal 只能用 lowcut、highcut、order。arguments 必须是对象，只填写用户明确指定的参数；"
    "未指定时省略，禁止创造 column、cutoff1、cutoff2、target_band 等替代参数名。"
    "中文数词或英文序数词表示的滤波阶数也必须写入 order。文件路径、信号数组和采样率由程序提供。"
)

BASE_SEED = 20260812
V2_SEED = 20260813
# 每个工具的新增样本数 / Number of new examples per tool.
TARGETED_SIZES = {"train": 40, "validation": 10}


def _compact_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))


def _variation_suffix(split: str, language: str, index: int) -> str:
    """为同一语义模板增加自然的上下文变化，避免机械重复。

    Add natural contextual variation to one semantic template to avoid mechanical repetition.
    """
    suffixes = {
        "train": {
            "zh": ["。", "，不要执行其他操作。", "，保留原始数据。", "，完成后停止。"],
            "en": ["", " Do not perform another operation.", " Leave the raw data unchanged.", " Stop after this step."],
        },
        "validation": {
            "zh": ["。", "，仅完成这一项任务。"],
            "en": ["", " Complete only this task."],
        },
    }
    bucket_size = 10 if split == "train" else 5
    choices = suffixes[split][language]
    return choices[(index // bucket_size) % len(choices)]


def _stats_case(split: str, index: int) -> tuple[str, dict[str, Any], str]:
    language = "zh" if index % 2 == 0 else "en"
    rates = [100, 125, 200, 250, 360, 500]
    rate = rates[(index * 5 + 1) % len(rates)]
    templates = {
        "train": {
            "zh": [
                "采样率虽然是 {rate} Hz，但只需汇总样本数、时长和统计范围",
                "不要采用 {rate} Hz 相关滤波设置，仅报告原序列的均值与标准差",
                "记录来自 {rate} Hz 采集；本次任务只是生成描述性统计摘要",
                "先忽略峰检测参数，概括这份 {rate} Hz 数据的数值分布",
                "无需修改波形，请统计 {rate} Hz 记录的长度、均值和极值",
            ],
            "en": [
                "The sampling rate is {rate} Hz, but only summarize count, duration, and range.",
                "Do not use {rate} Hz as a filter setting; report mean and standard deviation only.",
                "This was acquired at {rate} Hz; produce descriptive statistics for the raw series.",
                "Ignore peak settings and summarize the numerical distribution of this {rate} Hz trace.",
                "Without altering the waveform, describe the length, average, and extrema of the {rate} Hz record.",
            ],
        },
        "validation": {
            "zh": [
                "已知频率为 {rate} Hz，只计算整体统计量，不做信号处理",
                "对这份 {rate} Hz 序列进行只读的统计概括",
                "不要把 {rate} 当成工具参数，直接汇报数据分布",
            ],
            "en": [
                "Given a {rate} Hz acquisition, compute a read-only statistical overview.",
                "Treat {rate} as context, not a tool argument, and summarize the trace.",
                "Only describe this {rate} Hz record numerically; perform no processing.",
            ],
        },
    }
    choices = templates[split][language]
    question = choices[(index // 2) % len(choices)].format(rate=rate)
    question += _variation_suffix(split, language, index)
    return question, {}, language


def _load_case(split: str, index: int) -> tuple[str, dict[str, Any], str]:
    language = "zh" if index % 2 == 0 else "en"
    columns = [
        "ECG_I", "ppg_green", "lead_aVF", "wave_raw", "channel_B", "resp_trace", "sensor_03", "pulse_clean",
        "MLII", "value_norm", "ecg_filtered", "pleth_raw", "lead_V2", "optical", "signal_aux", "trace_main",
        "ECG_AVR", "ppg_red", "channel_07", "waveform_x", "lead_II_clean", "pulse_ir", "sensor_C", "recording",
        "ecg_mv", "plethysmogram", "lead_V5", "signal_02", "raw_value", "primary_trace", "ECG_III", "ppg_blue",
        "channel_Z", "wave_clean", "lead_aVL", "pulse_wave", "sensor_9", "main_signal", "ECG_V1", "trace_y",
    ]
    explicit = index % 5 != 0
    templates = {
        "train": {
            "zh_explicit": [
                "请从表格中读取 {column} 这一列作为信号",
                "输入通道的字段名是 {column}，只执行加载",
                "把列 {column} 映射为待分析序列并读入",
                "载入名为 {column} 的波形字段，暂不处理",
            ],
            "en_explicit": [
                "Read {column} as the signal column and do nothing else.",
                "The input field is named {column}; load that series.",
                "Map column {column} to the waveform input before analysis.",
                "Import the waveform field called {column} without processing it.",
            ],
            "zh_default": ["只把默认信号列加载进来", "读取当前文件，但不指定任何列名"],
            "en_default": ["Load only the default signal column.", "Read the current file without naming a column."],
        },
        "validation": {
            "zh_explicit": ["选择 {column} 字段作为输入波形", "仅读取数据列 {column}"],
            "en_explicit": ["Select the field {column} as input.", "Load data from the column named {column}."],
            "zh_default": ["载入默认通道，先不要分析"],
            "en_default": ["Bring in the default channel without analysis."],
        },
    }
    kind = f"{language}_{'explicit' if explicit else 'default'}"
    choices = templates[split][kind]
    question = choices[(index // 2) % len(choices)].format(column=columns[index])
    question += _variation_suffix(split, language, index)
    arguments = {"signal_column": columns[index]} if explicit else {}
    return question, arguments, language


def _peak_case(split: str, index: int, *, heart_rate: bool) -> tuple[str, dict[str, Any], str]:
    language = "zh" if index % 2 == 0 else "en"
    pattern = (index // 2) % 4
    distances = [0.25, 0.3, 0.35, 0.42, 0.48, 0.5, 0.55, 0.62, 0.7, 0.75]
    prominences = [0.11, 0.14, 0.17, 0.19, 0.22, 0.26, 0.29, 0.32, 0.36, 0.41]
    distance = distances[(index * 3) % len(distances)]
    prominence = prominences[(index * 7) % len(prominences)]
    arguments: dict[str, Any] = {}
    if pattern in {1, 3}:
        arguments["min_distance_seconds"] = distance
    if pattern in {2, 3}:
        arguments["prominence"] = prominence

    if language == "zh":
        if not arguments:
            params = "，其他检测选项保持默认"
        else:
            parts = []
            if "min_distance_seconds" in arguments:
                parts.append(f"最短峰间距 {distance} 秒")
            if "prominence" in arguments:
                parts.append(f"突出度门限 {prominence}")
            params = "，参数为" + "、".join(parts)
    else:
        if not arguments:
            params = " with all detection options left at default"
        else:
            parts = []
            if "min_distance_seconds" in arguments:
                parts.append(f"minimum peak spacing {distance} seconds")
            if "prominence" in arguments:
                parts.append(f"prominence threshold {prominence}")
            params = " using " + " and ".join(parts)

    task = "heart" if heart_rate else "peaks"
    templates = {
        "train": {
            "zh_peaks": ["返回峰索引{params}", "定位局部极大点而不计算 BPM{params}", "执行波形标峰{params}"],
            "en_peaks": ["Return peak indices{params}.", "Locate local maxima without computing BPM{params}.", "Mark waveform peaks{params}."],
            "zh_heart": ["输出最终平均心率{params}", "把心搏间隔换算成 BPM{params}", "估计每分钟脉搏数{params}"],
            "en_heart": ["Return final average heart rate{params}.", "Convert beat intervals to BPM{params}.", "Estimate pulses per minute{params}."],
        },
        "validation": {
            "zh_peaks": ["只找峰，不报告心率{params}", "给出峰的位置{params}"],
            "en_peaks": ["Find peaks but do not report heart rate{params}.", "Give the peak locations{params}."],
            "zh_heart": ["只报告 BPM，不列峰位置{params}", "计算该记录的脉率{params}"],
            "en_heart": ["Report BPM rather than peak locations{params}.", "Calculate pulse rate for this record{params}."],
        },
    }
    choices = templates[split][f"{language}_{task}"]
    question = choices[(index // 8) % len(choices)].format(params=params)
    question += _variation_suffix(split, language, index)
    return question, arguments, language


ZH_ORDINALS = {2: "二阶", 3: "三阶", 4: "四阶", 5: "五阶", 6: "六阶"}
EN_ORDINALS = {2: "second-order", 3: "third-order", 4: "fourth-order", 5: "fifth-order", 6: "sixth-order"}


def _filter_case(split: str, index: int) -> tuple[str, dict[str, Any], str]:
    language = "zh" if index % 2 == 0 else "en"
    pattern = (index // 2) % 6
    orders = [2, 3, 4, 5, 6]
    lows = [0.35, 0.5, 0.65, 0.8, 1.0, 1.25]
    highs = [7.5, 8.2, 9.0, 9.8, 10.5, 12.0]
    order = orders[(index * 3) % len(orders)]
    low = lows[(index * 5) % len(lows)]
    high = highs[(index * 7) % len(highs)]
    arguments: dict[str, Any] = {}
    if pattern in {1, 4}:
        arguments["order"] = order
    elif pattern == 2:
        arguments.update(lowcut=low, highcut=high)
    elif pattern == 3:
        arguments.update(lowcut=low, highcut=high, order=order)
    elif pattern == 5:
        # 单边截止频率也是合法的显式参数，另一侧继续使用工具默认值。
        # A one-sided cutoff is also explicit and valid; the other side keeps the tool default.
        if (index // 2) % 2:
            arguments["lowcut"] = low
        else:
            arguments["highcut"] = high

    if language == "zh":
        if not arguments:
            detail = "，全部参数使用默认值"
        elif set(arguments) == {"order"}:
            detail = f"，采用{ZH_ORDINALS[order]}滤波器"
        elif set(arguments) == {"lowcut", "highcut"}:
            detail = f"，通带设为 {low} 到 {high} Hz"
        elif set(arguments) == {"lowcut", "highcut", "order"}:
            detail = f"，用{ZH_ORDINALS[order]}结构保留 {low} 至 {high} Hz"
        elif "lowcut" in arguments:
            detail = f"，只把低截止频率改为 {low} Hz"
        else:
            detail = f"，仅将高截止频率设成 {high} Hz"
    else:
        if not arguments:
            detail = " with every option left at default"
        elif set(arguments) == {"order"}:
            detail = f" with a {EN_ORDINALS[order]} design"
        elif set(arguments) == {"lowcut", "highcut"}:
            detail = f" with passband boundaries {low} and {high} Hz"
        elif set(arguments) == {"lowcut", "highcut", "order"}:
            detail = f" as a {EN_ORDINALS[order]} design from {low} through {high} Hz"
        elif "lowcut" in arguments:
            detail = f" while changing only the low cutoff to {low} Hz"
        else:
            detail = f" while setting only the high cutoff to {high} Hz"

    templates = {
        "train": {
            "zh": ["执行带通滤波{detail}", "清除通带之外的成分{detail}", "对当前波形做频带限制{detail}"],
            "en": ["Apply band-pass filtering{detail}.", "Remove out-of-band content{detail}.", "Frequency-limit the current waveform{detail}."],
        },
        "validation": {
            "zh": ["处理这段信号的频带{detail}", "使用带通方式清理记录{detail}"],
            "en": ["Process the signal band{detail}.", "Clean the recording with a band pass{detail}."],
        },
    }
    choices = templates[split][language]
    question = choices[(index // 12) % len(choices)].format(detail=detail)
    question += _variation_suffix(split, language, index)
    return question, arguments, language


GENERATORS: dict[str, Callable[[str, int], tuple[str, dict[str, Any], str]]] = {
    "calculate_statistics": _stats_case,
    "load_signal": _load_case,
    "detect_peaks": lambda split, index: _peak_case(split, index, heart_rate=False),
    "calculate_heart_rate": lambda split, index: _peak_case(split, index, heart_rate=True),
    "filter_signal": _filter_case,
}


def _reprompt_base_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    copied["id"] = copied["id"].replace(f"sft_{split}_", f"sft_v2_base_{split}_", 1)
    copied["prompt"][0]["content"] = SFT_SYSTEM_PROMPT_V2
    copied["metadata"]["source"] = "synthetic_template_v1_reprompted_v2"
    copied["metadata"]["version"] = "v2"
    return copied


def _targeted_rows(split: str) -> list[dict[str, Any]]:
    rows = []
    for tool_name in TOOL_NAMES:
        for index in range(TARGETED_SIZES[split]):
            question, arguments, language = GENERATORS[tool_name](split, index)
            completion = _compact_call(tool_name, arguments)
            parse_tool_call(completion)
            rows.append(
                {
                    "id": f"sft_v2_targeted_{split}_{tool_name}_{index + 1:04d}",
                    "prompt": [
                        {"role": "system", "content": SFT_SYSTEM_PROMPT_V2},
                        {"role": "user", "content": question},
                    ],
                    "completion": [{"role": "assistant", "content": completion}],
                    "metadata": {
                        "split": split,
                        "tool_name": tool_name,
                        "language": language,
                        "arguments": arguments,
                        "source": "synthetic_targeted_v2",
                        "version": "v2",
                    },
                }
            )
    return rows


def generate_datasets_v2(seed: int = V2_SEED) -> dict[str, list[dict[str, Any]]]:
    """生成 700/150/100；函数不读取冻结的 final_cases_v1。

    Generate 700/150/100 examples without reading the frozen final_cases_v1.
    """
    base = generate_datasets(seed=BASE_SEED)
    result = {
        "train": [_reprompt_base_row(row, "train") for row in base["train"]] + _targeted_rows("train"),
        "validation": [_reprompt_base_row(row, "validation") for row in base["validation"]]
        + _targeted_rows("validation"),
        # v1 test 已经被检查过，因此在 v2 中明确称为开发测试，不冒充最终测试。
        # Because the v1 test was already inspected, v2 labels it as development rather than final testing.
        "test": [_reprompt_base_row(row, "test") for row in base["test"]],
    }
    rng = random.Random(seed)
    rng.shuffle(result["train"])
    rng.shuffle(result["validation"])
    rng.shuffle(result["test"])
    validate_datasets_v2(result)
    return result


def validate_datasets_v2(datasets: dict[str, list[dict[str, Any]]]) -> None:
    expected_per_tool = {"train": 140, "validation": 30, "test": 20}
    expected_totals = {split: count * len(TOOL_NAMES) for split, count in expected_per_tool.items()}
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for split, rows in datasets.items():
        if len(rows) != expected_totals[split]:
            raise ValueError(f"{split}: expected {expected_totals[split]} rows, got {len(rows)}.")
        counts = Counter(row["metadata"]["tool_name"] for row in rows)
        if counts != Counter({name: expected_per_tool[split] for name in TOOL_NAMES}):
            raise ValueError(f"{split}: unbalanced tools: {counts}")
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen_ids.add(row["id"])
            question = row["prompt"][1]["content"].strip().casefold()
            if question in seen_questions:
                raise ValueError(f"Question leakage across v2 splits: {question}")
            seen_questions.add(question)
            if row["prompt"][0]["content"] != SFT_SYSTEM_PROMPT_V2:
                raise ValueError(f"Wrong v2 system prompt: {row['id']}")
            call = parse_tool_call(row["completion"][0]["content"])
            if call.name != row["metadata"]["tool_name"] or call.arguments != row["metadata"]["arguments"]:
                raise ValueError(f"Label mismatch: {row['id']}")


def write_datasets_v2(output_dir: str | Path, seed: int = V2_SEED) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets_v2(seed=seed)
    paths = {}
    for split, rows in datasets.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split] = path
    manifest = {
        "version": "synthetic_targeted_v2",
        "seed": seed,
        "base_seed": BASE_SEED,
        "system_prompt_version": "v2_canonical_parameter_names",
        "split_sizes": {split: len(rows) for split, rows in datasets.items()},
        "per_tool": {"train": 140, "validation": 30, "test": 20},
        "targeted_additions_per_tool": TARGETED_SIZES,
        "test_status": "development_test_previously_inspected",
        "frozen_final_test": "evaluation/final_cases_v1.jsonl",
        "final_test_used_by_generator": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths
