"""Causal resampling, independent annotations, and chronological split metadata."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural.config import DEFAULT

CATEGORIES = {"Anomaly": 1, "Rare Event": 2, "Communication Gap": 4}


def annotation_masks(index, annotations, channels):
    """Bits retain overlaps. Intervals touch the grid's preceding sample bin.

    A label is evaluated on a bin [t-cadence, t], so a sub-cadence event is not
    silently lost. Labels are never used as telemetry or network features.
    """
    mask = np.zeros((len(index), len(channels)), dtype=np.uint8)
    for row in annotations.itertuples():
        if row.Channel not in channels:
            continue
        if row.Category not in CATEGORIES:
            raise ValueError(f"Unknown annotation category: {row.Category}")
        lo = index.searchsorted(row.StartTime, side="left")
        hi = index.searchsorted(row.EndTime, side="left")
        if lo < len(index) and hi >= 0:
            mask[max(0, lo) : min(len(index), hi + 1), channels.index(row.Channel)] |= (
                CATEGORIES[row.Category]
            )
    return mask


def split_codes(index, config=DEFAULT):
    dates = [
        pd.Timestamp(config.train_end),
        pd.Timestamp(config.validation_end),
        pd.Timestamp(config.calibration_end),
    ]
    if not dates[0] < dates[1] < dates[2]:
        raise ValueError("Chronological split boundaries must increase")
    return np.searchsorted(
        np.array([d.value for d in dates]), index.as_unit("ns").asi8, side="right"
    ).astype(np.uint8)


def valid_window_ends(valid, window, stride=1):
    """Windows never cross an invalid row, chronological split, or removed event."""
    valid = np.asarray(valid, dtype=bool)
    cumulative = np.r_[0, np.cumsum(~valid)]
    ends = np.arange(window - 1, len(valid), stride)
    return ends[(cumulative[ends + 1] - cumulative[ends + 1 - window]) == 0]


def load_annotation_table(folder, channels):
    annotations = pd.read_csv(folder / "labels.csv").merge(
        pd.read_csv(folder / "anomaly_types.csv"), on="ID", validate="many_to_one"
    )
    annotations = annotations[annotations.Channel.isin(channels)].copy()
    for col in ("StartTime", "EndTime"):
        annotations[col] = pd.to_datetime(annotations[col], utc=True, format="ISO8601")
    if (annotations.StartTime > annotations.EndTime).any():
        raise ValueError("Reversed annotation interval")
    return annotations


def prepare(
    folder=Path("data/raw/ESA-Mission1"),
    output=Path("data/processed/esa_neural"),
    config=DEFAULT,
):
    folder, output = Path(folder), Path(output)
    manifest_path = folder / "download_manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            "Verified official download manifest required; run src.ingestion.esa_download first"
        )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("source") != "REAL MISSION DATA"
        or manifest.get("doi") != "10.5281/zenodo.12528696"
    ):
        raise ValueError("Unexpected dataset provenance")
    hashes = {Path(item["path"]).name: item["sha256"] for item in manifest["members"]}
    annotations = load_annotation_table(folder, list(config.channels))
    output.mkdir(parents=True, exist_ok=True)
    # The six selected channel files share the mission time range. The first file
    # defines a grid; other channels outside it are not extrapolated/backfilled.
    index = None
    channel_stats = []
    for j, channel in enumerate(config.channels):
        path = folder / "channels" / f"{channel}.zip"
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if hashes.get(path.name) != digest:
            raise ValueError(f"File differs from verified official download: {channel}")
        # Trust is explicit in this path: downloader restricts to official archive
        # members and validates CRCs; never expose pickle reading to file uploads.
        frame = pd.read_pickle(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        series = frame[channel].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        if index is None:
            index = pd.date_range(
                series.index[0].ceil(f"{config.cadence_seconds}s"),
                series.index[-1].floor(f"{config.cadence_seconds}s"),
                freq=f"{config.cadence_seconds}s",
            )
            values = np.lib.format.open_memmap(
                output / "values.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(index), len(config.channels)),
            )
            values[:] = np.nan
        resampled = series.reindex(
            index,
            method="ffill",
            tolerance=pd.Timedelta(seconds=config.max_hold_seconds),
        )
        values[:, j] = pd.to_numeric(resampled, errors="raise").to_numpy(
            dtype=np.float32
        )
        channel_stats.append(
            {
                "channel": channel,
                "raw_rows": len(series),
                "first": series.index[0].isoformat(),
                "last": series.index[-1].isoformat(),
                "missing_fraction": float(np.isnan(values[:, j]).mean()),
                "sha256": digest,
            }
        )
        print(
            f"Prepared {channel}: {len(series):,} raw rows → {len(index):,} grid rows",
            flush=True,
        )
        del frame, series, resampled
        gc.collect()
    values.flush()
    labels = annotation_masks(index, annotations, list(config.channels))
    splits = split_codes(index, config)
    np.save(output / "labels.npy", labels)
    np.save(output / "timestamps_ns.npy", index.as_unit("ns").asi8)
    np.save(output / "splits.npy", splits)
    annotations.to_csv(output / "annotations.csv", index=False)
    channel_metadata = pd.read_csv(folder / "channels.csv")
    channel_metadata[channel_metadata.Channel.isin(config.channels)].to_csv(
        output / "channels.csv", index=False
    )
    report = {
        "source": "REAL MISSION DATA",
        "doi": manifest["doi"],
        "config": config.to_dict(),
        "rows": len(index),
        "start": index[0].isoformat(),
        "end": index[-1].isoformat(),
        "channel_stats": channel_stats,
        "split_rows": {
            name: int((splits == i).sum())
            for i, name in enumerate(["train", "validation", "calibration", "test"])
        },
        "annotated_event_ids": int(annotations.ID.nunique()),
        "label_policy": "Anomaly=1, rare nominal event=2, communication gap=4; interval overlaps grid bin; no label in model inputs",
        "resampling": "Causal zero-order hold, bounded to 300 seconds; no backfill or interpolation from the future",
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="data/raw/ESA-Mission1")
    parser.add_argument("--output", default="data/processed/esa_neural")
    args = parser.parse_args()
    prepare(Path(args.folder), Path(args.output))
