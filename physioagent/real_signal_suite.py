"""真实信号数据划分读取、冻结测试保护和多记录指标汇总。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_SPLITS = frozenset({"development", "validation", "test"})


def load_split_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Real-signal manifest must contain a non-empty records list.")
    for row in records:
        if row.get("split") not in ALLOWED_SPLITS:
            raise ValueError(f"Unknown real-signal split: {row.get('split')!r}")
        required = {"record", "split", "duration_seconds", "channel", "data_dir"}
        if not required.issubset(row):
            raise ValueError(f"Incomplete real-signal record entry: {row!r}")
    return manifest, hashlib.sha256(raw).hexdigest()


def select_split_records(
    manifest: dict[str, Any],
    splits: list[str],
    *,
    allow_frozen_test: bool = False,
) -> list[dict[str, Any]]:
    requested = set(splits)
    unknown = requested - ALLOWED_SPLITS
    if unknown:
        raise ValueError(f"Unknown splits: {sorted(unknown)}")
    if "test" in requested and not allow_frozen_test:
        raise ValueError(
            "The real-signal test split is frozen. Select and record one detector configuration "
            "before rerunning with allow_frozen_test=True."
        )
    return [row for row in manifest["records"] if row["split"] in requested]


def summarize_record_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one record result is required.")
    true_positives = sum(int(row["peak_matching"]["true_positives"]) for row in results)
    false_positives = sum(int(row["peak_matching"]["false_positives"]) for row in results)
    false_negatives = sum(int(row["peak_matching"]["false_negatives"]) for row in results)
    denominator_f1 = 2 * true_positives + false_positives + false_negatives
    hr_errors = [
        float(row["detected"]["heart_rate_error_bpm"])
        for row in results
        if row["detected"]["heart_rate_error_bpm"] is not None
    ]
    return {
        "num_records": len(results),
        "record_ids": [row["record"] for row in results],
        "micro_true_positives": true_positives,
        "micro_false_positives": false_positives,
        "micro_false_negatives": false_negatives,
        "micro_sensitivity": true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0,
        "micro_positive_predictive_value": true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0,
        "micro_f1": 2 * true_positives / denominator_f1 if denominator_f1 else 0.0,
        "mean_absolute_heart_rate_error_bpm": sum(hr_errors) / len(hr_errors) if hr_errors else None,
    }
