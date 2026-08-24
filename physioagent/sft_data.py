"""可复现的工具调用 SFT 数据生成器。

这批数据训练“理解问题并输出工具 JSON”，不包含真实信号数值。真实信号用于验证
工具算法；文本 SFT 数据用于训练模型的工具选择和参数抽取，两者职责不同。
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .agent import parse_tool_call


SFT_SYSTEM_PROMPT = (
    "你是生理时序工具调用助手。只输出一个 JSON 对象，顶层恰好包含 name 和 arguments。"
    "工具名只能是 calculate_statistics、load_signal、detect_peaks、calculate_heart_rate、"
    "filter_signal。arguments 必须是对象；只填写用户明确指定的参数，未指定参数必须省略。"
    "文件路径、信号数组和采样率由程序提供。"
)

SPLIT_SIZES = {"train": 100, "validation": 20, "test": 20}  # 每个工具的样本数
TOOL_NAMES = (
    "calculate_statistics",
    "load_signal",
    "detect_peaks",
    "calculate_heart_rate",
    "filter_signal",
)


def _compact_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))


def _sample_language(rng: random.Random) -> str:
    return rng.choice(("zh", "en"))


def _stats_case(rng: random.Random, split: str) -> tuple[str, dict[str, Any], str]:
    phrases = {
        "train": {
            "zh": ["计算这段信号的{focus}", "请报告当前记录的{focus}", "我想了解波形的{focus}", "帮我汇总信号的{focus}"],
            "en": ["Calculate the signal's {focus}.", "Report the recording's {focus}.", "Summarize the waveform's {focus}.", "I need the signal's {focus}."],
        },
        "validation": {
            "zh": ["给我看看该序列的{focus}", "概述一下这份记录的{focus}"],
            "en": ["Give me an overview of the {focus}.", "Describe this series using its {focus}."],
        },
        "test": {
            "zh": ["无需处理波形，只整理它的{focus}", "先做一份关于{focus}的摘要"],
            "en": ["Without modifying the waveform, provide its {focus}.", "Prepare a short summary covering the {focus}."],
        },
    }
    focuses = {
        "zh": ["均值与标准差", "样本数和持续时间", "最小值、最大值与波动程度", "基础描述统计", "数值范围和平均水平"],
        "en": ["mean and standard deviation", "sample count and duration", "minimum, maximum, and variability", "basic descriptive statistics", "range and average level"],
    }
    suffixes = {
        "train": {
            "zh": ["。", "，请直接给结果。", "，无需滤波。", "，不要进行峰检测。"],
            "en": ["", " Return only the result.", " No filtering is needed.", " Do not detect peaks."],
        },
        "validation": {
            "zh": ["。", "，只需统计摘要。"],
            "en": ["", " I only need the statistical summary."],
        },
        "test": {
            "zh": ["。", "，不要修改原信号。"],
            "en": ["", " Leave the original signal unchanged."],
        },
    }
    language = _sample_language(rng)
    question = rng.choice(phrases[split][language]).format(focus=rng.choice(focuses[language]))
    question += rng.choice(suffixes[split][language])
    return question, {}, language


def _load_case(rng: random.Random, split: str) -> tuple[str, dict[str, Any], str]:
    templates = {
        "train": {
            "zh_default": ["读取当前信号文件", "加载这份 CSV 生理信号", "使用默认信号列读取数据"],
            "zh_column": ["读取 CSV 的 {column} 列", "加载名为 {column} 的信号列", "使用 {column} 作为信号列"],
            "en_default": ["Load the current signal file.", "Read this physiological CSV.", "Use the default signal column."],
            "en_column": ["Read the {column} column from the CSV.", "Load the signal column named {column}.", "Use {column} as the signal column."],
        },
        "validation": {
            "zh_default": ["打开当前时序记录", "从默认列载入波形"],
            "zh_column": ["从列 {column} 中取得信号", "把 {column} 列载入为波形"],
            "en_default": ["Open the current time-series recording.", "Import the waveform from its default column."],
            "en_column": ["Take the samples from column {column}.", "Import column {column} as the waveform."],
        },
        "test": {
            "zh_default": ["先把这份生理序列读进来", "载入文件中默认的波形数据"],
            "zh_column": ["目标信号位于 {column} 列，请读取它", "请将列名 {column} 指定为输入信号"],
            "en_default": ["Bring the current physiological series into memory.", "Load the file's default waveform data."],
            "en_column": ["The target signal is in column {column}; read it.", "Select the column called {column} as input."],
        },
    }
    columns = ["ecg", "ECG_II", "ppg", "waveform", "lead_ii", "channel_A", "value", "signal_raw"]
    language = _sample_language(rng)
    explicit = rng.random() < 0.6
    key = f"{language}_{'column' if explicit else 'default'}"
    column = rng.choice(columns)
    question = rng.choice(templates[split][key]).format(column=column)
    suffixes = {
        "train": {"zh": ["。", "，然后等待下一步。", "，暂不分析。"], "en": ["", " Then wait for the next step.", " Do not analyze it yet."]},
        "validation": {"zh": ["。", "，仅执行读取。"], "en": ["", " Only perform loading."]},
        "test": {"zh": ["。", "，先不要计算统计量。"], "en": ["", " Do not compute statistics yet."]},
    }
    question += rng.choice(suffixes[split][language])
    return question, ({"signal_column": column} if explicit else {}), language


def _peak_like_case(
    rng: random.Random, split: str, *, heart_rate: bool
) -> tuple[str, dict[str, Any], str]:
    language = _sample_language(rng)
    task = "heart" if heart_rate else "peaks"
    templates = {
        "train": {
            "zh_peaks": ["检测这段信号的峰值{params}", "找出当前波形中的局部峰{params}", "标出信号峰{params}"],
            "en_peaks": ["Detect peaks in this signal{params}.", "Find the waveform's local peaks{params}.", "Locate signal peaks{params}."],
            "zh_heart": ["估算这段信号的平均心率{params}", "计算当前记录的 BPM{params}", "根据心搏峰间隔报告心率{params}"],
            "en_heart": ["Estimate mean heart rate{params}.", "Calculate BPM for this recording{params}.", "Report heart rate from beat intervals{params}."],
        },
        "validation": {
            "zh_peaks": ["识别波形的峰{params}", "定位所有局部峰{params}"],
            "en_peaks": ["Identify waveform peaks{params}.", "Mark all local maxima{params}."],
            "zh_heart": ["给出该序列的心率估计{params}", "求平均每分钟心搏数{params}"],
            "en_heart": ["Give a heart-rate estimate{params}.", "Find the average beats per minute{params}."],
        },
        "test": {
            "zh_peaks": ["在不计算心率的情况下找峰{params}", "返回该记录的峰位置{params}"],
            "en_peaks": ["Find peaks without calculating BPM{params}.", "Return the peak locations for this trace{params}."],
            "zh_heart": ["我只需要最终的 BPM{params}", "利用心搏间期推算每分钟心率{params}"],
            "en_heart": ["I only need the final BPM{params}.", "Infer beats per minute from the beat spacing{params}."],
        },
    }
    pattern = rng.choices(("none", "distance", "prominence", "both"), weights=(4, 2, 2, 2), k=1)[0]
    distances = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
    prominences = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    arguments: dict[str, Any] = {}
    if pattern in {"distance", "both"}:
        arguments["min_distance_seconds"] = rng.choice(distances)
    if pattern in {"prominence", "both"}:
        arguments["prominence"] = rng.choice(prominences)

    if language == "zh":
        parts = []
        if "min_distance_seconds" in arguments:
            parts.append(f"，最小峰间隔为 {arguments['min_distance_seconds']} 秒")
        if "prominence" in arguments:
            parts.append(f"，最小突出度为 {arguments['prominence']}")
        params = "".join(parts)
    else:
        parts = []
        if "min_distance_seconds" in arguments:
            parts.append(f" with minimum peak distance {arguments['min_distance_seconds']} seconds")
        if "prominence" in arguments:
            connector = " and" if parts else " with"
            parts.append(f"{connector} minimum prominence {arguments['prominence']}")
        params = "".join(parts)
    question = rng.choice(templates[split][f"{language}_{task}"]).format(params=params)
    return question, arguments, language


def _filter_case(rng: random.Random, split: str) -> tuple[str, dict[str, Any], str]:
    language = _sample_language(rng)
    templates = {
        "train": {
            "zh": ["对当前信号进行带通滤波{params}", "请滤除带外成分{params}", "处理这段波形的频带{params}"],
            "en": ["Band-pass filter the signal{params}.", "Remove out-of-band components{params}.", "Process the waveform frequency band{params}."],
        },
        "validation": {
            "zh": ["清理当前波形中的带外噪声{params}", "应用带通处理{params}"],
            "en": ["Clean out-of-band noise from the trace{params}.", "Apply band-pass processing{params}."],
        },
        "test": {
            "zh": ["仅留下目标频带{params}", "为这份记录执行频带限制{params}"],
            "en": ["Retain only the target frequency band{params}.", "Frequency-limit this recording{params}."],
        },
    }
    pattern = rng.choices(("none", "band", "order", "all"), weights=(3, 3, 2, 2), k=1)[0]
    bands = [(0.3, 7.0), (0.5, 8.0), (0.7, 9.0), (1.0, 10.0), (1.5, 11.0), (2.0, 12.0)]
    orders = [2, 3, 4, 5, 6]
    arguments: dict[str, Any] = {}
    if pattern in {"band", "all"}:
        low, high = rng.choice(bands)
        arguments.update(lowcut=low, highcut=high)
    if pattern in {"order", "all"}:
        arguments["order"] = rng.choice(orders)

    if language == "zh":
        parts = []
        if "lowcut" in arguments:
            parts.append(f"，保留 {arguments['lowcut']} 到 {arguments['highcut']} Hz")
        if "order" in arguments:
            parts.append(f"，阶数设为 {arguments['order']}")
        params = "".join(parts)
    else:
        parts = []
        if "lowcut" in arguments:
            parts.append(f" from {arguments['lowcut']} to {arguments['highcut']} Hz")
        if "order" in arguments:
            parts.append(f" using order {arguments['order']}")
        params = "".join(parts)
    question = rng.choice(templates[split][language]).format(params=params)
    return question, arguments, language


GENERATORS: dict[str, Callable[[random.Random, str], tuple[str, dict[str, Any], str]]] = {
    "calculate_statistics": _stats_case,
    "load_signal": _load_case,
    "detect_peaks": lambda rng, split: _peak_like_case(rng, split, heart_rate=False),
    "calculate_heart_rate": lambda rng, split: _peak_like_case(rng, split, heart_rate=True),
    "filter_signal": _filter_case,
}


def generate_datasets(seed: int = 20260812) -> dict[str, list[dict[str, Any]]]:
    """生成平衡且问题文本互不重复的 train/validation/test 数据。"""
    rng = random.Random(seed)
    datasets: dict[str, list[dict[str, Any]]] = {}
    all_questions: set[str] = set()

    for split, per_tool in SPLIT_SIZES.items():
        rows: list[dict[str, Any]] = []
        for tool_name in TOOL_NAMES:
            generated = 0
            attempts = 0
            while generated < per_tool:
                attempts += 1
                if attempts > per_tool * 500:
                    raise RuntimeError(f"Could not generate enough unique {split}/{tool_name} cases.")
                question, arguments, language = GENERATORS[tool_name](rng, split)
                normalized = question.strip().lower()
                if normalized in all_questions:
                    continue
                assistant = _compact_call(tool_name, arguments)
                parse_tool_call(assistant)
                row = {
                    "id": f"sft_{split}_{tool_name}_{generated + 1:04d}",
                    "prompt": [
                        {"role": "system", "content": SFT_SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    "completion": [{"role": "assistant", "content": assistant}],
                    "metadata": {
                        "split": split,
                        "tool_name": tool_name,
                        "language": language,
                        "arguments": arguments,
                        "source": "synthetic_template_v1",
                    },
                }
                rows.append(row)
                all_questions.add(normalized)
                generated += 1
        rng.shuffle(rows)
        datasets[split] = rows
    validate_datasets(datasets)
    return datasets


def validate_datasets(datasets: dict[str, list[dict[str, Any]]]) -> None:
    """在写盘前检查数量、平衡性、格式、标签和跨集合泄漏。"""
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for split, rows in datasets.items():
        expected_total = SPLIT_SIZES[split] * len(TOOL_NAMES)
        if len(rows) != expected_total:
            raise ValueError(f"{split} should contain {expected_total} rows, got {len(rows)}.")
        counts = Counter(row["metadata"]["tool_name"] for row in rows)
        if counts != Counter({name: SPLIT_SIZES[split] for name in TOOL_NAMES}):
            raise ValueError(f"Unbalanced {split} split: {counts}")

        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen_ids.add(row["id"])
            question = row["prompt"][-1]["content"].strip().lower()
            if question in seen_questions:
                raise ValueError(f"Question leakage across splits: {question}")
            seen_questions.add(question)
            call = parse_tool_call(row["completion"][0]["content"])
            if call.name != row["metadata"]["tool_name"] or call.arguments != row["metadata"]["arguments"]:
                raise ValueError(f"Label and metadata disagree: {row['id']}")


def write_datasets(output_dir: str | Path, seed: int = 20260812) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets(seed=seed)
    paths: dict[str, Path] = {}
    for split, rows in datasets.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split] = path
    manifest = {
        "version": "synthetic_template_v1",
        "seed": seed,
        "format": "conversational_prompt_completion",
        "split_sizes": {split: len(rows) for split, rows in datasets.items()},
        "per_tool": SPLIT_SIZES,
        "note": "Synthetic held-out test uses disjoint templates; a manually authored final test is still required.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths
