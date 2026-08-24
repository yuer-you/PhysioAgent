"""生成 Workflow DPO v1 偏好数据；不读取任何冻结测试集。

chosen 和 rejected 都是合法工作流 JSON。偏好只针对语义差异，避免模型把
“能解析”误当成“计划正确”。训练集与验证集使用不同问题模板和列名。

Generate Workflow DPO v1 preference data without reading frozen test sets. Both chosen and rejected
completions are valid workflow JSON; preferences target semantic differences so the model cannot confuse
parseability with planning correctness. Training and validation use different question templates and columns.
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


WORKFLOW_DPO_SEED = 20260822

PREFERENCE_COUNTS = {
    "train": {
        "omit_load": 300,
        "default_invent_column": 160,
        "explicit_drop_column": 160,
        "extra_unrequested_load": 180,
        "omit_final_analysis": 80,
        "duplicate_final_analysis": 120,
    },
    "validation": {
        "omit_load": 60,
        "default_invent_column": 32,
        "explicit_drop_column": 32,
        "extra_unrequested_load": 36,
        "omit_final_analysis": 16,
        "duplicate_final_analysis": 24,
    },
}

COLUMNS = {
    "train": ["signal", "MLII", "ECG_I", "lead_II", "wave_raw", "channel_A"],
    "validation": ["lead_V1", "recording", "primary_signal", "sensor_ecg"],
}

INVENTED_COLUMNS = ["waveform", "ecg", "input_field", "preset"]

FILTER_BANDS = {
    "train": [(0.45, 39.0), (0.9, 31.0), (1.4, 27.0), (2.2, 24.0), (3.4, 19.0), (4.6, 15.5)],
    "validation": [(0.65, 37.0), (1.1, 29.0), (1.8, 25.0), (2.7, 21.0), (3.8, 17.0), (4.9, 15.2)],
}

LOAD_PHRASES = {
    "train": {
        "zh": {
            "default": ["先采用自动选择的输入列", "先让程序决定源字段", "从默认来源通道开始", "先使用未指定名称的标准列"],
            "explicit": ["先从 {column} 列建立输入", "先把字段 {column} 作为源序列", "先读取通道 {column}", "先选定 {column} 为输入字段"],
        },
        "en": {
            "default": ["first use the automatically selected input column", "first let the program determine the source field", "start from the default source channel", "first use the unnamed standard column"],
            "explicit": ["first establish input from column {column}", "first use field {column} as the source series", "first read channel {column}", "first select {column} as the input field"],
        },
    },
    "validation": {
        "zh": {
            "default": ["先从系统首选字段取得输入", "先沿用加载器推荐的通道", "先由软件确定波形来源", "先取得无需点名的常规列"],
            "explicit": ["先以 {column} 列为数据起点", "先将 {column} 指定为来源", "先取得字段 {column} 的序列", "先把 {column} 通道设为输入"],
        },
        "en": {
            "default": ["first obtain input from the system-preferred field", "first follow the loader-recommended channel", "first let the software resolve the waveform source", "first obtain the ordinary column without naming it"],
            "explicit": ["first take column {column} as the data origin", "first designate {column} as the source", "first obtain the series in field {column}", "first set channel {column} as input"],
        },
    },
}

NO_LOAD_PHRASES = {
    "train": {
        "zh": ["直接使用已经激活的波形", "不要执行来源选择，处理当前信号", "从内存中的现有序列开始"],
        "en": ["use the already active waveform directly", "skip source selection and process the current signal", "start from the existing series in memory"],
    },
    "validation": {
        "zh": ["保持当前输入，不调用加载工具", "仅处理眼前已经就绪的序列", "省略数据来源步骤并继续"],
        "en": ["keep the current input without calling the loader", "process only the ready series at hand", "omit the data-source step and continue"],
    },
}

ANALYSIS_PHRASES = {
    "calculate_heart_rate": {
        "zh": ["计算平均心率", "报告每分钟心搏数", "得到平均 BPM"],
        "en": ["calculate mean heart rate", "report beats per minute", "derive average BPM"],
    },
    "detect_peaks": {
        "zh": ["检测 R 峰", "返回心搏峰位置", "标出 ECG 峰值"],
        "en": ["detect R peaks", "return heartbeat peak positions", "mark ECG peaks"],
    },
    "calculate_statistics": {
        "zh": ["汇总描述性统计", "报告长度、时长和均值", "概括最终序列"],
        "en": ["summarize descriptive statistics", "report length, duration, and mean", "characterize the final series"],
    },
}

DEDUP_SUFFIXES = {
    "zh": ["严格保持这个次序", "不要增减明确步骤", "只返回对应计划", "完成目标后停止", "确保所有动作都出现"],
    "en": ["preserving this order", "without adding or dropping an explicit step", "returning only the corresponding plan", "stopping after the goal", "ensuring every operation appears"],
}


def _compact(steps: list[dict[str, Any]]) -> str:
    return json.dumps({"steps": steps}, ensure_ascii=False, separators=(",", ":"))


def _language(index: int) -> str:
    return "zh" if index % 2 == 0 else "en"


def _analysis_tool(index: int) -> str:
    return ("calculate_heart_rate", "detect_peaks", "calculate_statistics")[(index // 2) % 3]


def _analysis_phrase(tool: str, language: str, index: int) -> str:
    phrases = ANALYSIS_PHRASES[tool][language]
    return phrases[(index // 6) % len(phrases)]


def _filter_step(split: str, language: str, index: int) -> tuple[dict[str, Any], str]:
    lowcut, highcut = FILTER_BANDS[split][(index // 2) % len(FILTER_BANDS[split])]
    arguments: dict[str, Any] = {"lowcut": lowcut, "highcut": highcut}
    if index % 4 == 0:
        arguments["order"] = 2 + ((index // 4) % 4)
    order = arguments.get("order")
    if language == "zh":
        order_text = f"用{order}阶滤波器" if order else ""
        phrase = f"{order_text}保留 {lowcut:g} 到 {highcut:g} Hz"
    else:
        order_text = f" with an order {order} filter" if order else ""
        phrase = f"retain {lowcut:g} to {highcut:g} Hz{order_text}"
    return {"name": "filter_signal", "arguments": arguments}, phrase


def _load_spec(split: str, language: str, index: int, kind: str) -> tuple[dict[str, Any], str]:
    templates = LOAD_PHRASES[split][language][kind]
    template = templates[(index // 2) % len(templates)]
    if kind == "default":
        return {"name": "load_signal", "arguments": {}}, template
    column = COLUMNS[split][(index // 2) % len(COLUMNS[split])]
    return {"name": "load_signal", "arguments": {"signal_column": column}}, template.format(column=column)


def _analysis_step(tool: str) -> dict[str, Any]:
    return {"name": tool, "arguments": {}}


def _category_suffix(tool: str) -> str:
    return {
        "calculate_heart_rate": "heart_rate",
        "detect_peaks": "peaks",
        "calculate_statistics": "statistics",
    }[tool]


def _decorate(task: str, split: str, language: str, preference_type: str, index: int) -> str:
    if split == "train":
        frames = {
            "zh": ["针对当前 ECG，{task}。", "请规划以下流程：{task}。", "本轮需要{task}。", "按顺序完成：{task}。"],
            "en": ["For the current ECG, {task}.", "Plan this workflow: {task}.", "This run should {task}.", "Complete in order: {task}."],
        }
    else:
        frames = {
            "zh": ["为输入波形生成计划：{task}。", "要求依次做到：{task}。", "当前目标是{task}。"],
            "en": ["Generate a plan for the input waveform to {task}.", "The required sequence is to {task}.", "The current goal is to {task}."],
        }
    offset = list(PREFERENCE_COUNTS[split]).index(preference_type)
    return frames[language][((index // 2) + offset) % len(frames[language])].format(task=task)


def _build_pair(
    split: str,
    preference_type: str,
    index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any]]:
    language = _language(index)
    tool = _analysis_tool(index)
    analysis_step = _analysis_step(tool)
    analysis_phrase = _analysis_phrase(tool, language, index)
    filter_step, filter_phrase = _filter_step(split, language, index)

    if preference_type in {"omit_load", "default_invent_column", "explicit_drop_column"}:
        kind = "default" if preference_type == "default_invent_column" else "explicit"
        if preference_type == "omit_load":
            kind = "default" if (index // 2) % 2 == 0 else "explicit"
        load_step, load_phrase = _load_spec(split, language, index, kind)
        use_three_steps = preference_type != "default_invent_column" or (index // 2) % 3 != 0
        if use_three_steps:
            chosen = [load_step, filter_step, analysis_step]
            task = (
                f"{load_phrase}，接着{filter_phrase}，最后{analysis_phrase}"
                if language == "zh"
                else f"{load_phrase}, next {filter_phrase}, and finally {analysis_phrase}"
            )
        else:
            chosen = [load_step, analysis_step]
            task = f"{load_phrase}，随后{analysis_phrase}" if language == "zh" else f"{load_phrase}, then {analysis_phrase}"
        if preference_type == "omit_load":
            rejected = chosen[1:]
        elif preference_type == "default_invent_column":
            rejected = json.loads(json.dumps(chosen))
            rejected[0]["arguments"] = {"signal_column": INVENTED_COLUMNS[(index // 2) % len(INVENTED_COLUMNS)]}
        else:
            rejected = json.loads(json.dumps(chosen))
            rejected[0]["arguments"] = {}
        metadata = {
            "load_policy": f"load_{kind}",
            "category": (
                "load_filter_" + _category_suffix(tool)
                if use_three_steps
                else "load_then_" + _category_suffix(tool)
            ),
        }
    elif preference_type == "extra_unrequested_load":
        context = NO_LOAD_PHRASES[split][language][(index // 2) % len(NO_LOAD_PHRASES[split][language])]
        chosen = [filter_step, analysis_step]
        rejected = [{"name": "load_signal", "arguments": {}}, *chosen]
        task = (
            f"{context}，先{filter_phrase}，再{analysis_phrase}"
            if language == "zh"
            else f"{context}; first {filter_phrase}, then {analysis_phrase}"
        )
        metadata = {"load_policy": "no_load", "category": "filter_then_" + _category_suffix(tool)}
    elif preference_type == "omit_final_analysis":
        context = NO_LOAD_PHRASES[split][language][(index // 2) % len(NO_LOAD_PHRASES[split][language])]
        chosen = [filter_step, analysis_step]
        rejected = chosen[:-1]
        task = (
            f"{context}，先{filter_phrase}，最后必须{analysis_phrase}"
            if language == "zh"
            else f"{context}; first {filter_phrase}, and finally {analysis_phrase}"
        )
        metadata = {"load_policy": "no_load", "category": "filter_then_" + _category_suffix(tool)}
    elif preference_type == "duplicate_final_analysis":
        kind = "default" if (index // 2) % 2 == 0 else "explicit"
        load_step, load_phrase = _load_spec(split, language, index, kind)
        chosen = [load_step, analysis_step]
        rejected = [*chosen, json.loads(json.dumps(analysis_step))]
        task = (
            f"{load_phrase}，然后只执行一次{analysis_phrase}并停止"
            if language == "zh"
            else f"{load_phrase}, then {analysis_phrase} once and stop"
        )
        metadata = {
            "load_policy": f"load_{kind}",
            "category": "load_then_" + _category_suffix(tool),
        }
    else:
        raise ValueError(f"Unknown preference type: {preference_type}")

    metadata.update(
        language=language,
        chosen_steps=chosen,
        rejected_steps=rejected,
        chosen_step_count=len(chosen),
        rejected_step_count=len(rejected),
        final_tool=chosen[-1]["name"],
    )
    return chosen, rejected, task, metadata


def _make_rows(split: str, preference_type: str, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for index in range(count):
        chosen, rejected, task, metadata = _build_pair(split, preference_type, index)
        question = _decorate(task, split, metadata["language"], preference_type, index)
        normalized = question.strip().casefold()
        occurrence = seen[normalized]
        seen[normalized] += 1
        if occurrence:
            suffixes = DEDUP_SUFFIXES[metadata["language"]]
            if occurrence > len(suffixes):
                raise ValueError(f"Too many repeated DPO templates in {split}/{preference_type}.")
            suffix = suffixes[occurrence - 1]
            question = (
                question.rstrip("。") + f"，{suffix}。"
                if metadata["language"] == "zh"
                else question.rstrip(".") + f", {suffix}."
            )
        rows.append(
            {
                "id": f"workflow_dpo_v1_{split}_{preference_type}_{index + 1:04d}",
                "prompt": [
                    {"role": "system", "content": WORKFLOW_SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "chosen": [{"role": "assistant", "content": _compact(chosen)}],
                "rejected": [{"role": "assistant", "content": _compact(rejected)}],
                "metadata": {
                    "task_type": "workflow_preference",
                    "split": split,
                    "preference_type": preference_type,
                    "source": "synthetic_workflow_preferences_v1",
                    "version": "workflow_dpo_v1",
                    **metadata,
                },
            }
        )
    return rows


def generate_workflow_dpo_datasets(seed: int = WORKFLOW_DPO_SEED) -> dict[str, list[dict[str, Any]]]:
    datasets: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(seed)
    for split, counts in PREFERENCE_COUNTS.items():
        rows: list[dict[str, Any]] = []
        for preference_type, count in counts.items():
            rows.extend(_make_rows(split, preference_type, count))
        rng.shuffle(rows)
        datasets[split] = rows
    validate_workflow_dpo_datasets(datasets)
    return datasets


def _steps(raw: str) -> list[dict[str, Any]]:
    return [{"name": step.name, "arguments": step.arguments} for step in parse_workflow_plan(raw)]


def _validate_pair(row: dict[str, Any]) -> None:
    metadata = row["metadata"]
    chosen = _steps(row["chosen"][0]["content"])
    rejected = _steps(row["rejected"][0]["content"])
    if chosen != metadata["chosen_steps"] or rejected != metadata["rejected_steps"]:
        raise ValueError(f"Preference text and metadata disagree: {row['id']}")
    if chosen == rejected:
        raise ValueError(f"Chosen and rejected are identical: {row['id']}")
    preference_type = metadata["preference_type"]
    if preference_type == "omit_load" and not (chosen[0]["name"] == "load_signal" and rejected == chosen[1:]):
        raise ValueError(f"Invalid omit_load pair: {row['id']}")
    if preference_type == "default_invent_column":
        if chosen[0]["arguments"] != {} or not rejected[0]["arguments"].get("signal_column"):
            raise ValueError(f"Invalid default_invent_column pair: {row['id']}")
    if preference_type == "explicit_drop_column":
        if not chosen[0]["arguments"].get("signal_column") or rejected[0]["arguments"] != {}:
            raise ValueError(f"Invalid explicit_drop_column pair: {row['id']}")
    if preference_type == "extra_unrequested_load":
        if rejected[0]["name"] != "load_signal" or rejected[1:] != chosen:
            raise ValueError(f"Invalid extra_unrequested_load pair: {row['id']}")
    if preference_type == "omit_final_analysis" and rejected != chosen[:-1]:
        raise ValueError(f"Invalid omit_final_analysis pair: {row['id']}")
    if preference_type == "duplicate_final_analysis":
        if rejected[:-1] != chosen or rejected[-1] != chosen[-1]:
            raise ValueError(f"Invalid duplicate_final_analysis pair: {row['id']}")


def validate_workflow_dpo_datasets(datasets: dict[str, list[dict[str, Any]]]) -> None:
    if set(datasets) != {"train", "validation"}:
        raise ValueError("Workflow DPO must contain train and validation only.")
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for split, rows in datasets.items():
        if len(rows) != sum(PREFERENCE_COUNTS[split].values()):
            raise ValueError(f"Wrong row count for {split}.")
        if Counter(row["metadata"]["preference_type"] for row in rows) != Counter(PREFERENCE_COUNTS[split]):
            raise ValueError(f"Wrong preference distribution for {split}.")
        for row in rows:
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen_ids.add(row["id"])
            question = row["prompt"][1]["content"].strip().casefold()
            if question in seen_questions:
                raise ValueError(f"Question leakage across DPO splits: {question}")
            seen_questions.add(question)
            if row["metadata"]["split"] != split:
                raise ValueError(f"Wrong split metadata: {row['id']}")
            _validate_pair(row)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_workflow_dpo_datasets(
    output_dir: str | Path,
    seed: int = WORKFLOW_DPO_SEED,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = generate_workflow_dpo_datasets(seed)
    paths: dict[str, Path] = {}
    for split, rows in datasets.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split] = path
    manifest = {
        "version": "workflow_dpo_v1",
        "seed": seed,
        "system_prompt": "workflow_prompt_v2_with_ecg_profile",
        "split_sizes": {split: len(rows) for split, rows in datasets.items()},
        "preference_counts": PREFERENCE_COUNTS,
        "preference_length_direction": {
            "chosen_longer": ["omit_load", "omit_final_analysis"],
            "equal_structure": ["default_invent_column", "explicit_drop_column"],
            "chosen_shorter": ["extra_unrequested_load", "duplicate_final_analysis"],
        },
        "file_sha256": {split: _sha256(path) for split, path in paths.items()},
        "frozen_tests_read_by_generator": False,
        "diagnostic_inspiration": "final v3 error taxonomy only; no final question or output is copied",
        "forbidden_training_sources": [
            "evaluation/workflow_final_cases_v1.jsonl",
            "evaluation/workflow_final_cases_v2.jsonl",
            "evaluation/workflow_final_cases_v3.jsonl",
        ],
        "test_split_created": False,
    }
    with (output / "manifest.json").open("w", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return paths
