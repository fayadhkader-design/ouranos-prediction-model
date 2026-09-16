"""Optional ESA-AD adapter. Real channel identifiers are retained by default.
Never infer spacecraft subsystems from anonymized channel numbers.
"""

from pathlib import Path
import json
import argparse
import pandas as pd


def load_preprocessed_csv(path):
    df = pd.read_csv(path)
    if "timestamp" not in df:
        raise ValueError("ESA preprocessed CSV requires timestamp")
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    if (
        df.timestamp.isna().any()
        or df.timestamp.duplicated().any()
        or not df.timestamp.is_monotonic_increasing
    ):
        raise ValueError("ESA timestamps must be unique and increasing")
    labels = [c for c in df if c.startswith("is_anomaly_")]
    channels = [c for c in df if c not in labels + ["timestamp", "source"]]
    if not channels:
        raise ValueError("No telemetry channels found")
    telemetry = df[["timestamp"] + channels].copy()
    telemetry["source"] = "REAL MISSION DATA"
    return telemetry, df[["timestamp"] + labels].copy()


def load_raw_mission(
    folder, channels, start, end, cadence="30s", max_hold="5min", trusted_pickle=False
):
    """Read official zipped pandas pickles only with explicit trust acknowledgement.
    Bounded past-only zero-order hold; gaps remain NaN. Do not backfill.
    Returned annotation table is kept separate from telemetry.
    """
    if not trusted_pickle:
        raise ValueError(
            "Official channel .zip files contain pickle: pass trusted_pickle=True only after verifying provenance"
        )
    folder = Path(folder)
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    start = (
        start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    )
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end <= start:
        raise ValueError("end must be after start")
    if not channels:
        raise ValueError("Select channels explicitly to bound memory")
    idx = pd.date_range(start, end, freq=cadence)
    if len(idx) > 2_000_000:
        raise ValueError("Request a shorter interval or coarser cadence")
    out = pd.DataFrame(index=idx)
    for channel in channels:
        if Path(channel).name != channel:
            raise ValueError("Invalid channel name")
        frame = pd.read_pickle(folder / "channels" / f"{channel}.zip")
        frame.index = pd.to_datetime(frame.index, utc=True)
        if channel not in frame:
            raise ValueError(f"Expected column {channel}")
        series = frame[channel].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        out[channel] = series.reindex(
            idx, method="ffill", tolerance=pd.Timedelta(max_hold)
        )
    out.index.name = "timestamp"
    out = out.reset_index()
    out["source"] = "REAL MISSION DATA"
    annotations = pd.read_csv(folder / "labels.csv")
    for c in ["StartTime", "EndTime"]:
        annotations[c] = pd.to_datetime(annotations[c], utc=True)
    annotations = annotations[
        annotations.Channel.isin(channels)
        & (annotations.EndTime >= start)
        & (annotations.StartTime <= end)
    ]
    types = folder / "anomaly_types.csv"
    if types.exists():
        annotations = annotations.merge(
            pd.read_csv(types), on="ID", how="left", suffixes=("", "_type")
        )
    return out, annotations


def apply_mapping(telemetry, mapping):
    """mapping: source identifier -> target name, scale, offset, subsystem and unit.
    A human must supply semantic mapping, physical units and context engineering.
    """
    targets = [v["target"] for v in mapping.values()]
    if (
        len(set(targets)) != len(targets)
        or "timestamp" in targets
        or "source" in targets
    ):
        raise ValueError("Duplicate or reserved mapping target")
    out = telemetry[["timestamp", "source"]].copy()
    for source, definition in mapping.items():
        if source not in telemetry:
            raise ValueError(f"Missing source channel {source}")
        for key in ["target", "unit", "subsystem"]:
            if key not in definition:
                raise ValueError(f"Mapping must declare {key}")
        out[definition["target"]] = pd.to_numeric(
            telemetry[source], errors="raise"
        ) * definition.get("scale", 1) + definition.get("offset", 0)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Import ESA-ADB preprocessed CSV while preserving anonymous channels"
    )
    p.add_argument("csv")
    p.add_argument("--mapping")
    p.add_argument("--out", default="data/processed/esa")
    a = p.parse_args()
    telemetry, labels = load_preprocessed_csv(a.csv)
    if a.mapping:
        telemetry = apply_mapping(telemetry, json.loads(Path(a.mapping).read_text()))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(out / "telemetry.csv", index=False)
    labels.to_csv(out / "annotations.csv", index=False)
    print(
        f"Imported {len(telemetry)} rows of REAL MISSION DATA; no synthetic model scoring performed"
    )
