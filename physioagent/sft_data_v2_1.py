"""生成 SFT v2.1：只修复 v2 开发评测中剩余的两类局部错误。"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .agent import parse_tool_call
from .sft_data import TOOL_NAMES
from .sft_data_v2 import SFT_SYSTEM_PROMPT_V2, V2_SEED, generate_datasets_v2


V2_1_SEED = 20260814
TARGETED_COUNTS = {
    "train": {"calculate_statistics": 30, "filter_signal": 30},
    "validation": {"calculate_statistics": 10, "filter_signal": 10},
}

ZH_ORDINALS = {2: "二阶", 3: "三阶", 4: "四阶", 5: "五阶", 6: "六阶"}
EN_ORDINALS = {2: "second-order", 3: "third-order", 4: "fourth-order", 5: "fifth-order", 6: "sixth-order"}


def _compact_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, separators=(",", ":"))


def _make_row(
    split: str,
    tool_name: str,
    index: int,
    question: str,
    arguments: dict[str, Any],
    language: str,
) -> dict[str, Any]:
    completion = _compact_call(tool_name, arguments)
    parse_tool_call(completion)
    return {
        "id": f"sft_v2_1_targeted_{split}_{tool_name}_{index + 1:04d}",
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
            "source": "synthetic_targeted_v2_1",
            "version": "v2.1",
            "targeted_issue": (
                "statistics_tool_name_stability"
                if tool_name == "calculate_statistics"
                else "order_only_without_invented_cutoffs"
            ),
        },
    }


def _statistics_rows(split: str) -> list[dict[str, Any]]:
    count = TARGETED_COUNTS[split]["calculate_statistics"]
    train_templates = {
        "zh": [
            "在保持原始序列不变的前提下，汇总采样点总数和记录时长",
            "不要处理波形，只报告它包含多少点以及持续多久",
            "数据已经载入；现在给出样本规模、时间长度和描述性统计",
            "本次不找峰也不滤波，请计算序列长度与持续时间",
            "对当前记录做只读统计，包括点数和时长",
        ],
        "en": [
            "Keep the original series untouched and summarize its observation count and recording length",
            "Do not process the trace; report how many points it contains and how long it lasts",
            "The data is loaded; now give sample size, time span, and descriptive statistics",
            "Skip peak detection and filtering, and calculate sequence length plus duration",
            "Perform a read-only statistical summary including point count and elapsed time",
        ],
    }
    train_suffixes = {
        "zh": ["。", "，同时报告均值。", "，再补充取值范围。"],
        "en": [".", ", together with the mean.", ", and add the numerical range."],
    }
    validation_templates = {
        "zh": [
            "不改变这段数据，请统计总点数与时间跨度。",
            "只做描述性分析：记录多长、包含多少个观测？",
            "保持波形原样，给出样本量、持续时长和均值。",
            "无需创建新工具，直接汇总当前序列的长度和时长。",
            "请提供点数及记录时间，禁止执行滤波。",
        ],
        "en": [
            "Leave the data unchanged and summarize total observations and elapsed time.",
            "Descriptive analysis only: how large and how long is this record?",
            "Preserve the waveform and give sample size, duration, and average.",
            "Use the statistics capability to summarize sequence length and time span.",
            "Provide point count and recording duration without filtering.",
        ],
    }

    rows = []
    for index in range(count):
        language = "zh" if index % 2 == 0 else "en"
        local_index = index // 2
        if split == "train":
            templates = train_templates[language]
            suffixes = train_suffixes[language]
            question = templates[local_index % len(templates)] + suffixes[local_index // len(templates)]
        else:
            question = validation_templates[language][local_index]
        rows.append(_make_row(split, "calculate_statistics", index, question, {}, language))
    return rows


def _filter_rows(split: str) -> list[dict[str, Any]]:
    count = TARGETED_COUNTS[split]["filter_signal"]
    train_templates = {
        "zh": [
            "请用{ordinal}结构进行带通处理，其余参数保持默认。",
            "滤波器阶数选择{digit}，不要设置通带边界。",
            "只调整滤波阶数为{digit}，低、高截止频率均沿用默认值。",
        ],
        "en": [
            "Run band-pass processing with a {ordinal} design; leave both cutoffs unspecified.",
            "Set only the filter order to {digit}; retain the default frequency bounds.",
            "Use order {digit} and do not supply low or high cutoff values.",
        ],
    }
    validation_templates = {
        "zh": "带通部分采用{ordinal}，频率上下限不要改。",
        "en": "Make the band-pass stage {ordinal}; do not alter either cutoff.",
    }
    orders = [2, 3, 4, 5, 6]
    rows = []
    for index in range(count):
        language = "zh" if index % 2 == 0 else "en"
        local_index = index // 2
        order = orders[local_index % len(orders)]
        ordinal = ZH_ORDINALS[order] if language == "zh" else EN_ORDINALS[order]
        if split == "train":
            template = train_templates[language][local_index // len(orders)]
        else:
            template = validation_templates[language]
        question = template.format(ordinal=ordinal, digit=order)
        rows.append(_make_row(split, "filter_signal", index, question, {"order": order}, language))
    return rows


def generate_datasets_v2_1(seed: int = V2_1_SEED) -> dict[str, list[dict[str, Any]]]:
    """生成 760/170/100；不读取冻结最终测试或其标签。"""
    base = generate_datasets_v2(seed=V2_SEED)
    result = {split: copy.deepcopy(rows) for split, rows in base.items()}
    for split in ("train", "validation"):
        result[split].extend(_statistics_rows(split))
        result[split].extend(_filter_rows(split))
    rng = random.Random(seed)
    for rows in result.values():
        rng.shuffle(rows)
    validate_datasets_v2_1(result)
    return result


def validate_datasets_v2_1(datasets: dict[str, list[dict[str, Any]]]) -> None:
    expected_totals = {"train": 760, "validation": 170, "test": 100}
    expected_counts = {
        "train": {
            "calculate_statistics": 170,
            "load_signal": 140,
            "detect_peaks": 140,
            "calculate_heart_rate": 140,
            "filter_signal": 170,
        },
        "validation": {
            "calculate_statistics": 40,
            "load_signal": 30,
            "detect_peaks": 30,
            "calculate_heart_rate": 30,
            "filter_signal": 40,
        },
        "test": {name: 20 for name in TOOL_NAMES},
    }
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for split, rows in datasets.items():
        if len(rows) != expected_totals[split]:
            raise ValueError(f"{split}: expected {expected_totals[split]} rows, got {len(rows)}.")
        counts = Counter(row["metadata"]["tool_name"] for row in rows)
        if counts != Counter(expected_counts[split]):
            raise ValueError(f"{split}: unexpected tool counts: {counts}")
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen_ids.add(row["id"])
            question = row["prompt"][1]["content"].strip().casefold()
            if question in seen_questions:
                raise ValueError(f"Question leakage across v2.1 splits: {question}")
            seen_questions.add(question)
            if row["prompt"][0]["content"] != SFT_SYSTEM_PROMPT_V2:
                raise ValueError(f"Wrong system prompt: {row['id']}")
            call = parse_tool_call(row["completion"][0]["content"])
            if call.name != row["metadata"]["tool_name"] or call.arguments != row["metadata"]["arguments"]:
                raise ValueError(f"Label mismatch: {row['id']}")


def write_datasets_v2_1(output_dir: str | Path, seed: int = V2_1_SEED) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = generate_datasets_v2_1(seed=seed)
    paths = {}
    for split, rows in datasets.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split] = path
    manifest = {
        "version": "synthetic_targeted_v2_1",
        "seed": seed,
        "base_version": "synthetic_targeted_v2",
        "base_seed": V2_SEED,
        "split_sizes": {split: len(rows) for split, rows in datasets.items()},
        "targeted_additions": TARGETED_COUNTS,
        "targeted_issues": [
            "statistics_tool_name_stability",
            "order_only_without_invented_cutoffs",
        ],
        "test_status": "development_test_previously_inspected",
        "frozen_final_test": "evaluation/final_cases_v1.jsonl",
        "final_test_used_by_generator": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths
