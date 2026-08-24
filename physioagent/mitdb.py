"""MIT-BIH 小片段下载、转换和参考标注保存。

Download and convert short MIT-BIH excerpts and save their reference annotations.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .real_signal import mean_heart_rate_from_annotations


DATABASE = "mitdb"
BEAT_SYMBOLS = frozenset({"N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q"})
SOURCE_URL = "https://physionet.org/content/mitdb/1.0.0/"
LICENSE_URL = "https://physionet.org/content/mitdb/view-license/1.0.0/"


def prepare_mitdb_record(
    record_name: str,
    duration_seconds: float,
    channel: int,
    output_dir: str | Path,
    *,
    wfdb_module: Any | None = None,
) -> dict[str, Any]:
    """下载一个片段，写出 signal.csv 和 reference.json，并返回参考信息。

    Download one excerpt, write signal.csv and reference.json, and return reference metadata.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    if channel < 0:
        raise ValueError("channel must be non-negative.")
    if wfdb_module is None:
        try:
            import wfdb as wfdb_module
        except ImportError as error:
            raise RuntimeError(
                "Missing wfdb. Install it with: pip install -r requirements-real-data.txt"
            ) from error

    header = wfdb_module.rdheader(record_name, pn_dir=DATABASE)
    sampling_rate = float(header.fs)
    num_samples = round(duration_seconds * sampling_rate)
    if channel >= int(header.n_sig):
        raise ValueError(f"Record {record_name} has only {header.n_sig} channels.")

    record = wfdb_module.rdrecord(
        record_name,
        sampfrom=0,
        sampto=num_samples,
        channels=[channel],
        pn_dir=DATABASE,
    )
    annotation = wfdb_module.rdann(
        record_name,
        "atr",
        sampfrom=0,
        sampto=num_samples,
        shift_samps=True,
        pn_dir=DATABASE,
    )
    signal = record.p_signal[:, 0]
    if len(signal) != num_samples or any(not math.isfinite(float(value)) for value in signal):
        raise ValueError("Downloaded signal is incomplete or contains non-finite values.")

    beat_indices: list[int] = []
    beat_symbols: list[str] = []
    for index, symbol in zip(annotation.sample, annotation.symbol):
        if symbol in BEAT_SYMBOLS:
            beat_indices.append(int(index))
            beat_symbols.append(symbol)
    if len(beat_indices) < 2:
        raise ValueError("The selected segment contains fewer than two annotated beats.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    signal_path = destination / "signal.csv"
    with signal_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["signal"])
        writer.writerows((format(float(value), ".10g"),) for value in signal)

    reference = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "database": DATABASE,
        "record": record_name,
        "channel_index": channel,
        "channel_name": record.sig_name[0],
        "units": record.units[0],
        "sampling_rate_hz": sampling_rate,
        "num_samples": len(signal),
        "duration_seconds": len(signal) / sampling_rate,
        "annotation_extension": "atr",
        "beat_indices": beat_indices,
        "beat_symbols": beat_symbols,
        "num_annotated_beats": len(beat_indices),
        "reference_mean_heart_rate_bpm": mean_heart_rate_from_annotations(beat_indices, sampling_rate),
        "source_url": SOURCE_URL,
        "license": "Open Data Commons Attribution License v1.0",
        "license_url": LICENSE_URL,
        "attribution": "Moody GB, Mark RG. MIT-BIH Arrhythmia Database. PhysioNet.",
    }
    (destination / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return reference
