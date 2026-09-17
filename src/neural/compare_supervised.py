"""Post-hoc diagnostic on common scored minute bins, not model selection."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.neural.supervised import DATA, REPORT, MODEL, CHANNELS
from src.neural.prepare import load_annotation_table
from src.neural.evaluate import event_metrics
from src.evaluation import confusion


def compare():
    index = pd.to_datetime(np.load(DATA / "timestamps.npy", mmap_mode="r"), utc=True)
    labels = np.load(DATA / "labels.npy", mmap_mode="r")
    y = ((labels & 1) != 0).any(axis=1)
    score = np.load(DATA / "scores.npy", mmap_mode="r")
    v2threshold = json.loads((MODEL / "threshold.json").read_text())["chosen"][
        "threshold"
    ]
    old = Path("data/processed/esa_neural")
    old_index = pd.to_datetime(
        np.load(old / "timestamps_ns.npy", mmap_mode="r"), utc=True
    )
    old_score = np.load(old / "neural_scores.npy", mmap_mode="r")
    old_alarm = np.load(old / "neural_alarms.npy", mmap_mode="r")
    # Preserve either original V1 alert within a minute; require both half-minute
    # positions to be scoreable. No threshold is changed in this diagnostic.
    v1 = (
        pd.DataFrame(
            {"alarm": old_alarm, "valid": np.isfinite(old_score)}, index=old_index
        )
        .resample("60s", label="right", closed="right")
        .agg({"alarm": "max", "valid": "min"})
        .reindex(index)
        .fillna(False)
    )
    common = (
        (index >= pd.Timestamp("2007-01-01", tz="UTC"))
        & np.isfinite(score)
        & v1.valid.to_numpy(dtype=bool)
    )
    annotations = load_annotation_table(Path("data/raw/ESA-Mission1"), CHANNELS)
    report = {
        "scope": "Post-hoc common-coverage retrospective comparison; V1 alerts OR-aggregated within right-closed minute bins; V2 same frozen threshold",
        "common_minutes": int(common.sum()),
        "positive_common_minutes": int((common & y).sum()),
        "models": {},
    }
    for name, alarm in [
        ("autoencoder", v1.alarm.to_numpy(dtype=bool)),
        ("supervised", score > v2threshold),
    ]:
        events, episodes = event_metrics(
            index,
            labels,
            common,
            alarm & common,
            annotations,
            CHANNELS,
            np.full(len(index), 255, dtype=np.uint8),
            cadence_seconds=60,
        )
        report["models"][name] = {
            "detected_events": sum(e["detected"] for e in events),
            "total_events": len(events),
            "point": confusion(y[common], alarm[common]),
            **episodes,
            "false_episodes_per_1000_nominal_hours": episodes[
                "false_nonoverlapping_episodes"
            ]
            / ((common & ~y).sum() / 60)
            * 1000,
        }
    # Yearly diagnostics expose alert prevalence changes without retuning the model.
    yearly = []
    for year in range(2005, 2014):
        mask = (index.year == year) & np.isfinite(score)
        negative = mask & ~y
        yearly.append(
            {
                "year": year,
                "scored_minutes": int(mask.sum()),
                "negative_alert_fraction": float((score[negative] > v2threshold).mean())
                if negative.any()
                else None,
            }
        )
    report["supervised_by_year"] = yearly
    (REPORT / "common_coverage_comparison.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    compare()
