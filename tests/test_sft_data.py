import json

from physioagent.agent import parse_tool_call
from physioagent.sft_data import SPLIT_SIZES, TOOL_NAMES, generate_datasets, write_datasets


def test_sft_dataset_sizes_balance_and_no_leakage():
    datasets = generate_datasets(seed=123)
    questions = []
    for split, rows in datasets.items():
        assert len(rows) == SPLIT_SIZES[split] * len(TOOL_NAMES)
        questions.extend(row["prompt"][-1]["content"].strip().lower() for row in rows)
    assert len(questions) == len(set(questions))


def test_sft_completions_are_strict_valid_tool_calls():
    datasets = generate_datasets(seed=456)
    for rows in datasets.values():
        for row in rows:
            call = parse_tool_call(row["completion"][0]["content"])
            assert call.name == row["metadata"]["tool_name"]
            assert call.arguments == row["metadata"]["arguments"]


def test_sft_generation_is_reproducible():
    first = generate_datasets(seed=789)
    second = generate_datasets(seed=789)
    assert first == second


def test_write_sft_manifest(tmp_path):
    write_datasets(tmp_path, seed=321)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["split_sizes"] == {"train": 500, "validation": 100, "test": 100}
