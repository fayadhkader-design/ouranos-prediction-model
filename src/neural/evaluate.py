"""Freeze threshold on nominal calibration, then evaluate untouched test years.
Reports ordinary point and event-overlap metrics, NOT official ESA-ADB metrics.
No point adjustment or predicted failure-time claims.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from src.evaluation import confusion, event_starts
from src.neural.config import NeuralConfig
from src.neural.network import window_batch
from src.neural.prepare import valid_window_ends
from src.neural.train import checkpoint_model, nominal_rows


def read_annotations(path):
    annotations = pd.read_csv(path)
    for column in ("StartTime", "EndTime"):
        annotations[column] = pd.to_datetime(
            annotations[column], utc=True, format="ISO8601"
        )
    return annotations


def sustained_array(exceeded, count):
    cumulative = np.r_[0, np.cumsum(np.asarray(exceeded, dtype=np.int64))]
    output = np.zeros(len(exceeded), dtype=bool)
    if len(output) >= count:
        output[count - 1 :] = (cumulative[count:] - cumulative[:-count]) == count
    return output


def score_dataset(data, output, checkpoint, pca):
    config = NeuralConfig(**checkpoint["config"])
    values = np.load(data / "values.npy", mmap_mode="r")
    labels = np.load(data / "labels.npy", mmap_mode="r")
    splits = np.load(data / "splits.npy", mmap_mode="r")
    valid = np.isfinite(values).all(axis=1) & ((labels & 4) == 0).all(axis=1)
    scores = {
        name: np.lib.format.open_memmap(
            output / f"{name}_scores.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(values),),
        )
        for name in ("neural", "pca")
    }
    for array in scores.values():
        array[:] = np.nan
    top = np.lib.format.open_memmap(
        output / "neural_top_channel.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(values),),
    )
    top[:] = 255
    model = checkpoint_model(checkpoint)
    center, scale = (
        np.array(checkpoint["center"], dtype=np.float32),
        np.array(checkpoint["scale"], dtype=np.float32),
    )
    torch.set_num_threads(4)
    for split in (1, 2, 3):
        ends = valid_window_ends(valid & (splits == split), config.window)
        print(f"Scoring split {split}: {len(ends):,} causal windows", flush=True)
        with torch.inference_mode():
            for i in range(0, len(ends), 4096):
                idx = ends[i : i + 4096]
                batch = window_batch(values, idx, config.window, center, scale)
                predicted = model(batch)[:, -1, :].numpy()
                actual = batch[:, -1, :].numpy()
                residual = np.square(actual - predicted)
                scores["neural"][idx] = residual.max(axis=1)
                top[idx] = residual.argmax(axis=1)
                scores["pca"][idx] = np.square(
                    actual - pca.inverse_transform(pca.transform(actual))
                ).max(axis=1)
                if (i // 4096 + 1) % 100 == 0:
                    print(
                        f"Split {split}: {min(i + 4096, len(ends)):,}/{len(ends):,}",
                        flush=True,
                    )
    for array in scores.values():
        array.flush()
    top.flush()
    return scores


def event_metrics(
    index,
    labels,
    test_mask,
    alarm,
    annotations,
    channels,
    top_channels,
    test_start="2007-01-01T00:00:00Z",
    cadence_seconds=30,
):
    """Each true event is a union of its selected-channel intervals, not a broad
    min-start/max-end envelope. An event can only be detected inside that union.
    Fragmented predicted episodes are explicitly counted; no point adjustment.
    """
    anomalies = annotations[annotations.Category.eq("Anomaly")]
    per_event = []
    for event_id, group in anomalies.groupby("ID", sort=True):
        any_test = False
        eligible_count = 0
        total_count = 0
        first_hit = None
        observed_channel = None
        correct_channel = None
        onset = group.StartTime.min()
        for row in group.itertuples():
            lo = max(0, index.searchsorted(row.StartTime, side="left"))
            hi = min(len(index), index.searchsorted(row.EndTime, side="left") + 1)
            if hi <= lo:
                continue
            # test_mask includes missingness/warmup; test interval separately tracks
            # events with zero coverage so they do not silently disappear.
            in_test_time = index[lo:hi] >= pd.Timestamp(test_start)
            if not in_test_time.any():
                continue
            any_test = True
            total_count += int(in_test_time.sum())
            eligible_count += int(test_mask[lo:hi].sum())
            hits = np.flatnonzero(alarm[lo:hi] & test_mask[lo:hi])
            if len(hits):
                position = lo + hits[0]
                if first_hit is None or position < first_hit:
                    first_hit = position
                    observed_channel = (
                        channels[int(top_channels[position])]
                        if top_channels[position] < len(channels)
                        else None
                    )
                    # Compare the dominant residual with channels annotated at that
                    # timestamp, not a guessed physical subsystem.
                    active = group[
                        (group.StartTime <= index[position])
                        & (
                            group.EndTime + pd.Timedelta(seconds=cadence_seconds)
                            >= index[position]
                        )
                    ].Channel
                    correct_channel = observed_channel in set(active)
        if not any_test:
            continue
        per_event.append(
            {
                "event_id": event_id,
                "onset": onset.isoformat(),
                "end": group.EndTime.max().isoformat(),
                "detected": first_hit is not None,
                "first_detection": index[first_hit].isoformat()
                if first_hit is not None
                else None,
                "detection_delay_minutes": (index[first_hit] - onset).total_seconds()
                / 60
                if first_hit is not None
                else None,
                "scorable": eligible_count > 0,
                "channel_interval_coverage": eligible_count / total_count
                if total_count
                else None,
                "dominant_channel_at_detection": observed_channel,
                "dominant_channel_annotated": correct_channel,
            }
        )
    starts = event_starts(alarm & test_mask)
    # False episode if its entire predicted span overlaps no scored anomaly point.
    positions = np.flatnonzero(starts)
    ends = np.flatnonzero((alarm & test_mask) & ~np.r_[(alarm & test_mask)[1:], False])
    y = ((labels & 1) != 0).any(axis=1)
    true_episodes = sum(bool(y[a : b + 1].any()) for a, b in zip(positions, ends))
    false_episodes = len(positions) - true_episodes
    false_starts = int((starts & ~y & test_mask).sum())
    return per_event, {
        "predicted_episodes": len(positions),
        "true_overlapping_episodes": true_episodes,
        "false_nonoverlapping_episodes": false_episodes,
        "episodes_starting_outside_anomaly": false_starts,
    }


def evaluate(
    data=Path("data/processed/esa_neural"),
    models=Path("models/esa_neural"),
    reports=Path("reports/esa_neural"),
    rescore=True,
):
    data, models, reports = Path(data), Path(models), Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        models / "autoencoder.pt", map_location="cpu", weights_only=True
    )
    config = NeuralConfig(**checkpoint["config"])
    pca = joblib.load(models / "pca_baseline.joblib")
    if rescore:
        score_dataset(data, data, checkpoint, pca)
    values = np.load(data / "values.npy", mmap_mode="r")
    labels = np.load(data / "labels.npy", mmap_mode="r")
    splits = np.load(data / "splits.npy", mmap_mode="r")
    timestamp_ns = np.load(data / "timestamps_ns.npy", mmap_mode="r")
    index = pd.to_datetime(timestamp_ns, utc=True)
    annotations = read_annotations(data / "annotations.csv")
    top_channels = np.load(data / "neural_top_channel.npy", mmap_mode="r")
    nominal = nominal_rows(values, labels)
    y = ((labels & 1) != 0).any(axis=1)
    rare = ((labels & 2) != 0).any(axis=1)
    report = {
        "source": "REAL MISSION DATA",
        "dataset": "ESA Anomaly Dataset",
        "mission": "ESA-Mission1",
        "doi": "10.5281/zenodo.12528696",
        "channels": list(config.channels),
        "subsystem": "subsystem_5 (anonymized)",
        "test_start": config.calibration_end,
        "test_end": index[-1].isoformat(),
        "threshold_policy": f"{config.threshold_quantile:.1%} quantile of unannotated calibration-window scores; {config.warning_dwell} consecutive 30-second bins above threshold",
        "metrics_policy": "Ordinary point scores and event overlap; no point adjustment; not the official ESA-ADB composite metrics. Rare nominal events count as negatives. Communication gaps and invalid windows are excluded.",
        "failure_prediction": False,
        "models": {},
    }
    thresholds = {}
    for name in ("neural", "pca"):
        score = np.load(data / f"{name}_scores.npy", mmap_mode="r")
        cal = (splits == 2) & nominal & np.isfinite(score)
        # All observations in a calibration window must be unannotated, just like
        # training. A just-ended event cannot contaminate threshold fitting.
        cal_ends = valid_window_ends(cal, config.window)
        if len(cal_ends) < 1000:
            raise ValueError("Insufficient healthy calibration scores")
        threshold = float(np.quantile(score[cal_ends], config.threshold_quantile))
        thresholds[name] = {
            "threshold": threshold,
            "calibration_rows": len(cal_ends),
            "quantile": config.threshold_quantile,
        }
    # Freeze threshold provenance before any test performance is calculated.
    (models / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    downsampled = pd.DataFrame(
        {
            "timestamp": index[::120],
            "split": np.asarray(splits[::120]),
            "anomaly": y[::120],
            "rare_event": rare[::120],
        }
    )
    for name in ("neural", "pca"):
        score = np.load(data / f"{name}_scores.npy", mmap_mode="r")
        threshold = thresholds[name]["threshold"]
        eligible = (splits == 3) & np.isfinite(score)
        alarm = sustained_array((score > threshold) & eligible, config.warning_dwell)
        cal_eligible = (splits == 2) & np.isfinite(score)
        cal_alarm = sustained_array(
            (score > threshold) & cal_eligible, config.warning_dwell
        )
        np.save(data / f"{name}_alarms.npy", alarm)
        metrics = confusion(y[eligible], alarm[eligible])
        if y[eligible].any() and (~y[eligible]).any():
            auprc = float(average_precision_score(y[eligible], score[eligible]))
            auroc = float(roc_auc_score(y[eligible], score[eligible]))
        else:
            auprc = auroc = None
        per_event, episodes = event_metrics(
            index,
            labels,
            eligible,
            alarm,
            annotations,
            list(config.channels),
            top_channels
            if name == "neural"
            else np.full(len(values), 255, dtype=np.uint8),
            test_start=config.calibration_end,
            cadence_seconds=config.cadence_seconds,
        )
        pd.DataFrame(per_event).to_csv(reports / f"{name}_events.csv", index=False)
        scorable = [e for e in per_event if e["scorable"]]
        detected = [e for e in per_event if e["detected"]]
        nominal_hours = float((eligible & ~y).sum() * config.cadence_seconds / 3600)
        rare_mask = eligible & rare & ~y
        counts = {
            "test_rows": int((splits == 3).sum()),
            "scored_rows": int(eligible.sum()),
            "coverage_fraction": float(eligible.sum() / (splits == 3).sum()),
            "positive_rows": int((eligible & y).sum()),
            "anomaly_prevalence": float(y[eligible].mean()),
            "nominal_hours": nominal_hours,
        }
        report["models"][name] = {
            "threshold": threshold,
            "point": metrics,
            "auprc": auprc,
            "auroc": auroc,
            "event_recall_all": len(detected) / len(per_event) if per_event else None,
            "event_recall_scorable": len(detected) / len(scorable)
            if scorable
            else None,
            "detected_events": len(detected),
            "total_events": len(per_event),
            "scorable_events": len(scorable),
            "median_detection_delay_minutes_detected_only": float(
                np.median([e["detection_delay_minutes"] for e in detected])
            )
            if detected
            else None,
            **episodes,
            "false_alerts_per_1000_nominal_hours": episodes[
                "false_nonoverlapping_episodes"
            ]
            / nominal_hours
            * 1000,
            "rare_event_false_positive_rate": float(alarm[rare_mask].mean())
            if rare_mask.any()
            else None,
            "calibration_nominal_false_positive_rate": float(
                cal_alarm[cal_eligible & nominal].mean()
            ),
            "counts": counts,
        }
        downsampled[name + "_score"] = score[::120]
        downsampled[name + "_alarm"] = alarm[::120]
        print(name, json.dumps(report["models"][name], indent=2), flush=True)
    downsampled.to_csv(reports / "hourly_overview.csv", index=False)
    report["model_sha256"] = hashlib.sha256(
        (models / "autoencoder.pt").read_bytes()
    ).hexdigest()
    (reports / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-scores", action="store_true")
    args = parser.parse_args()
    evaluate(rescore=not args.reuse_scores)
