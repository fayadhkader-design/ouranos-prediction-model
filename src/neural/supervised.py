"""Supervised anomaly classifier with causal, transient-preserving minute bins.
The post-2007 comparison is retrospective: it was already examined for V1.
"""

import hashlib
import json
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from src.neural.prepare import annotation_masks, load_annotation_table
from src.neural.evaluate import event_metrics
from src.evaluation import confusion, event_starts

CHANNELS = [f"channel_{i}" for i in range(41, 47)]
DATA = Path("data/processed/esa_supervised")
REPORT = Path("reports/esa_supervised")
MODEL = Path("models/esa_supervised")


def bin_statistics(series, index):
    # Bin timestamp is its right edge; no reading after t enters the row at t.
    grouped = series.resample("60s", closed="right", label="right").agg(
        ["last", "min", "max", "std"]
    )
    grouped["std"] = grouped["std"].fillna(0)
    return grouped.reindex(index).to_numpy(dtype=np.float32)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    folder = Path("data/raw/ESA-Mission1")
    manifest = json.loads((folder / "download_manifest.json").read_text())
    hashes = {Path(m["path"]).name: m["sha256"] for m in manifest["members"]}
    index = pd.date_range("2000-01-01 00:01:00Z", "2014-01-01 00:00:00Z", freq="60s")
    x = np.lib.format.open_memmap(
        DATA / "features.npy", mode="w+", dtype=np.float32, shape=(len(index), 24)
    )
    for j, c in enumerate(CHANNELS):
        path = folder / "channels" / f"{c}.zip"
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != hashes[path.name]:
                raise ValueError("Source integrity mismatch")
        frame = pd.read_pickle(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        series = frame[c].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        x[:, j * 4 : j * 4 + 4] = bin_statistics(series, index)
        x.flush()
        print("Prepared transient-preserving bins:", c, flush=True)
        del frame, series
    annotations = load_annotation_table(folder, CHANNELS)
    labels = annotation_masks(index, annotations, CHANNELS)
    np.save(DATA / "labels.npy", labels)
    np.save(DATA / "timestamps.npy", index.as_unit("ns").asi8)
    (DATA / "provenance.json").write_text(
        json.dumps(
            {
                "source": "REAL MISSION DATA",
                "doi": manifest["doi"],
                "channels": CHANNELS,
                "aggregation": "Right-closed one-minute last/min/max/std; no gap filling; every observed spike contributes to extrema",
                "rows": len(index),
            },
            indent=2,
        )
    )


def features(x, idx, center, scale):
    current = np.asarray(x[idx])
    previous = np.asarray(x[np.maximum(idx - 1, 0)])
    # Four descriptors/channel plus past-only one-minute last-value differences.
    raw = np.column_stack([current, current[:, ::4] - previous[:, ::4]])
    return np.clip((raw - center) / scale, -30, 30).astype(np.float32)


def classifier():
    return nn.Sequential(
        nn.Linear(30, 48), nn.GELU(), nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1)
    )


def select_examples(
    mask, y, index, annotations, seed, negative_count=80000, per_event=200
):
    rng = np.random.default_rng(seed)
    negatives = np.flatnonzero(mask & ~y)
    negatives = rng.choice(
        negatives, min(len(negatives), negative_count), replace=False
    )
    positive = []
    for _, group in annotations[annotations.Category.eq("Anomaly")].groupby("ID"):
        candidates = []
        for row in group.itertuples():
            lo = index.searchsorted(row.StartTime)
            hi = min(len(index), index.searchsorted(row.EndTime) + 1)
            ids = np.arange(lo, hi)
            ids = ids[mask[ids] & y[ids]]
            candidates.extend(ids)
        candidates = np.unique(candidates)
        if len(candidates):
            positive.extend(
                rng.choice(candidates, min(len(candidates), per_event), replace=False)
            )
    positive = np.unique(positive).astype(int)
    if len(positive) < 2:
        raise ValueError("Too few labeled positive examples")
    return np.r_[negatives, positive], len(positive)


def train_evaluate():
    REPORT.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2027)
    rng = np.random.default_rng(2027)
    x = np.load(DATA / "features.npy", mmap_mode="r")
    labels = np.load(DATA / "labels.npy", mmap_mode="r")
    index = pd.to_datetime(np.load(DATA / "timestamps.npy", mmap_mode="r"), utc=True)
    annotations = load_annotation_table(Path("data/raw/ESA-Mission1"), CHANNELS)
    finite = np.isfinite(x).all(axis=1)
    valid = finite & np.r_[False, finite[:-1]] & ~((labels & 4) != 0).any(axis=1)
    # No difference feature may cross an annotated communication gap.
    valid &= np.r_[False, ~((labels[:-1] & 4) != 0).any(axis=1)]
    y = ((labels & 1) != 0).any(axis=1)
    train = valid & (index < pd.Timestamp("2004-01-01", tz="UTC"))
    val = (
        valid
        & (index >= pd.Timestamp("2004-01-01", tz="UTC"))
        & (index < pd.Timestamp("2005-01-01", tz="UTC"))
    )
    cal = (
        valid
        & (index >= pd.Timestamp("2005-01-01", tz="UTC"))
        & (index < pd.Timestamp("2007-01-01", tz="UTC"))
    )
    test = valid & (index >= pd.Timestamp("2007-01-01", tz="UTC"))
    # Exclude boundary rows so previous-value differences never cross splits.
    for mask in (train, val, cal, test):
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        mask[starts] = False
    ti, np_train = select_examples(train, y, index, annotations, 2027)
    vi, np_val = select_examples(val, y, index, annotations, 2028, negative_count=30000)
    raw = np.column_stack([x[ti], x[ti, ::4] - x[np.maximum(ti - 1, 0), ::4]])
    center = np.median(raw, axis=0)
    scale = np.maximum(np.std(raw, axis=0), 1e-6)
    xt = torch.from_numpy(features(x, ti, center, scale))
    yt = torch.from_numpy(y[ti].astype(np.float32))
    xv = torch.from_numpy(features(x, vi, center, scale))
    yv = y[vi]
    model = classifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    lossfn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(ti) - np_train) / np_train)
    )
    best = -1
    stale = 0
    history = []
    best_state = None
    for epoch in range(1, 61):
        model.train()
        order = rng.permutation(len(ti))
        losses = []
        for start in range(0, len(order), 512):
            ids = order[start : start + 512]
            optimizer.zero_grad()
            loss = lossfn(model(xt[ids]).squeeze(1), yt[ids])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            prob = torch.sigmoid(model(xv).squeeze(1)).numpy()
        ap = float(average_precision_score(yv, prob))
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_sample_average_precision": ap,
            }
        )
        if ap > best + 1e-4:
            best = ap
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        print(history[-1], flush=True)
        if stale >= 8:
            break
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = {
        "state_dict": best_state,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "channels": CHANNELS,
        "best_epoch": best_epoch,
        "source": "REAL MISSION DATA",
        "architecture": "30-48-24-1 GELU binary classifier",
        "score_not_calibrated_probability": True,
    }
    torch.save(checkpoint, MODEL / "classifier.pt")
    (REPORT / "training.json").write_text(
        json.dumps(
            {
                "training_positive_examples": np_train,
                "training_examples": len(ti),
                "validation_positive_examples": np_val,
                "validation_examples": len(vi),
                "best_epoch": best_epoch,
                "history": history,
                "selection": "Earlier labeled events capped at 200 bins/event; random negatives; validation AP on sampled set is not full-stream precision",
                "split_dates": ["2004-01-01", "2005-01-01", "2007-01-01"],
            },
            indent=2,
        )
    )
    scores = np.lib.format.open_memmap(
        DATA / "scores.npy", mode="w+", dtype=np.float32, shape=(len(index),)
    )
    scores[:] = np.nan
    ids = np.flatnonzero(cal | test)
    with torch.inference_mode():
        for start in range(0, len(ids), 16384):
            batch = ids[start : start + 16384]
            scores[batch] = torch.sigmoid(
                model(torch.from_numpy(features(x, batch, center, scale))).squeeze(1)
            ).numpy()
    scores.flush()
    # Choose one-bin transient threshold on calibration only. Event recall is the
    # objective, constrained to <=10 false episode starts/1000 negative hours.
    cal_annotations = annotations[
        (annotations.EndTime >= pd.Timestamp("2005-01-01", tz="UTC"))
        & (annotations.StartTime < pd.Timestamp("2007-01-01", tz="UTC"))
    ]
    cal_hours = float((cal & ~y).sum() / 60)
    candidates = np.unique(
        np.r_[
            np.linspace(0.05, 0.99, 30),
            np.quantile(scores[cal & ~y], [0.99, 0.995, 0.999, 0.9995, 0.9999]),
            1.0,
        ]
    )
    trials = []
    for threshold in candidates:
        alarm = cal & (scores > threshold)
        false_starts = int((event_starts(alarm) & ~y).sum())
        detected = 0
        total = 0
        for _, group in cal_annotations[cal_annotations.Category.eq("Anomaly")].groupby(
            "ID"
        ):
            caught = False
            covered = False
            for r in group.itertuples():
                lo = index.searchsorted(r.StartTime)
                hi = min(len(index), index.searchsorted(r.EndTime) + 1)
                covered |= bool(cal[lo:hi].any())
                caught |= bool(alarm[lo:hi].any())
            if covered:
                total += 1
                detected += int(caught)
        trials.append(
            {
                "threshold": float(threshold),
                "detected_events": detected,
                "scorable_events": total,
                "false_starts_per_1000h": false_starts / cal_hours * 1000,
            }
        )
    feasible = [r for r in trials if r["false_starts_per_1000h"] <= 10]
    chosen = max(
        feasible,
        key=lambda r: (
            r["detected_events"],
            -r["false_starts_per_1000h"],
            r["threshold"],
        ),
    )
    (MODEL / "threshold.json").write_text(
        json.dumps(
            {
                "chosen": chosen,
                "policy": "Max calibration event detection with <=10 false episode starts/1000 nominal hours; ties favor fewer false starts then higher threshold",
                "trials": trials,
            },
            indent=2,
        )
    )
    alarm = test & (scores > chosen["threshold"])
    events, episodes = event_metrics(
        index,
        labels,
        test,
        alarm,
        annotations,
        CHANNELS,
        np.full(len(index), 255, dtype=np.uint8),
        cadence_seconds=60,
    )
    pd.DataFrame(events).to_csv(REPORT / "events.csv", index=False)
    result = {
        "source": "REAL MISSION DATA",
        "test_status": "Retrospective comparison: these 2007–2013 years were already inspected for V1. No claim of fresh untouched final validation.",
        "model": "Supervised binary neural classifier",
        "features": "Past-only minute last/min/max/std and preceding-minute differences; no annotated label as input",
        "threshold": chosen,
        "point": confusion(y[test], alarm[test]),
        "average_precision": float(average_precision_score(y[test], scores[test])),
        "detected_events": sum(e["detected"] for e in events),
        "total_events": len(events),
        "scored_rows": int(test.sum()),
        "test_rows": int((index >= pd.Timestamp("2007-01-01", tz="UTC")).sum()),
        **episodes,
        "nominal_hours": float((test & ~y).sum() / 60),
        "false_alerts_per_1000_nominal_hours": episodes["false_nonoverlapping_episodes"]
        / float((test & ~y).sum() / 60)
        * 1000,
        "limitations": [
            "Only six channels from one mission",
            "Not a calibrated probability or failure predictor",
            "Minute binning delays alerts until bin closes",
            "Brief-event extrema preserved, but independent extrema are not simultaneous sensor readings",
            "Event-capped positive sampling and weighted loss alter class prior",
            "No physical subsystem mapping",
            "Test years already inspected; additional independent validation required",
        ],
    }
    (REPORT / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare()
    train_evaluate()
