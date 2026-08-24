from pathlib import Path

import pytest

from physioagent.mitdb import prepare_mitdb_record
from physioagent.real_signal_suite import (
    load_split_manifest,
    select_split_records,
    summarize_record_results,
)


MANIFEST = Path(__file__).parents[1] / "evaluation" / "real_signal_split_v1.json"


def test_real_signal_manifest_has_fixed_splits():
    manifest, digest = load_split_manifest(MANIFEST)
    by_split = {
        split: [row["record"] for row in manifest["records"] if row["split"] == split]
        for split in ("development", "validation", "test")
    }
    assert by_split == {
        "development": ["100"],
        "validation": ["101", "200"],
        "test": ["207"],
    }
    assert len(digest) == 64


def test_frozen_real_signal_test_requires_explicit_permission():
    manifest, _ = load_split_manifest(MANIFEST)
    with pytest.raises(ValueError, match="test split is frozen"):
        select_split_records(manifest, ["test"])
    selected = select_split_records(manifest, ["test"], allow_frozen_test=True)
    assert [row["record"] for row in selected] == ["207"]


def test_record_summary_uses_micro_counts_and_mean_hr_error():
    results = [
        {
            "record": "a",
            "peak_matching": {"true_positives": 8, "false_positives": 2, "false_negatives": 0},
            "detected": {"heart_rate_error_bpm": 2.0},
        },
        {
            "record": "b",
            "peak_matching": {"true_positives": 2, "false_positives": 0, "false_negatives": 2},
            "detected": {"heart_rate_error_bpm": 4.0},
        },
    ]
    summary = summarize_record_results(results)
    assert summary["micro_sensitivity"] == pytest.approx(10 / 12)
    assert summary["micro_positive_predictive_value"] == pytest.approx(10 / 12)
    assert summary["micro_f1"] == pytest.approx(20 / 24)
    assert summary["mean_absolute_heart_rate_error_bpm"] == pytest.approx(3.0)


def test_prepare_record_validates_inputs_before_importing_wfdb(tmp_path):
    with pytest.raises(ValueError, match="duration_seconds"):
        prepare_mitdb_record("100", 0, 0, tmp_path)
    with pytest.raises(ValueError, match="channel"):
        prepare_mitdb_record("100", 30, -1, tmp_path)
