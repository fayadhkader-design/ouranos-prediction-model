"""Frozen-policy, held-out synthetic evaluation. No point adjustment or label leakage."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from src.pipeline import train_default, analyze
from src.simulation.generator import generate
from src.risk.engine import sustained


def first_time(mask, timestamps):
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    return pd.Timestamp(timestamps.iloc[idx[0]]) if len(idx) else None


def event_starts(mask):
    a = np.asarray(mask, dtype=bool)
    return a & ~np.r_[False, a[:-1]]


def confusion(y_true, y_pred):
    a = np.asarray(y_true, dtype=bool)
    b = np.asarray(y_pred, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Label/prediction shapes differ")
    tp = int((a & b).sum())
    fp = int((~a & b).sum())
    tn = int((~a & ~b).sum())
    fn = int((a & ~b).sum())
    divide = lambda x, y: x / y if y else None
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "true_positive_rate": divide(tp, tp + fn),
        "false_positive_rate": divide(fp, fp + tn),
    }


def run_metrics(sim, result):
    t = sim.telemetry.timestamp
    truth = sim.truth
    s = result.scores
    failure = first_time(truth.simulated_failure, t)
    onset = first_time(truth.degradation, t)
    # An alert before degradation is a false alert, not a true positive.
    eligible = truth.degradation & (~truth.simulated_failure)
    starts = event_starts(s.warning)
    warning = first_time(starts & eligible, t)
    baseline = first_time(result.baseline.any(axis=1), t)
    anomaly = first_time(sustained(s.anomaly >= 0.5, 6) & eligible, t)
    healthy = (~truth.degradation) & s.ready
    false_events = int((starts & healthy).sum())
    duration = float(healthy.sum() * result.model.config.cadence_minutes / 60)
    lead = (
        (baseline - warning).total_seconds() / 3600
        if baseline is not None and warning is not None
        else None
    )
    expected_subsystem = {
        "reaction_wheel": "ADCS",
        "battery": "BATTERY",
        "solar": "POWER",
    }.get(sim.metadata["scenario"])
    warning_subsystem = None
    if warning is not None:
        warning_subsystem = str(s.loc[s.timestamp == warning, "subsystem"].iloc[0])
    valid = s.ready & ~truth.simulated_failure
    cm = confusion(truth.degradation[valid], s.warning[valid])
    origin = t.iloc[0]
    hours = lambda v: (v - origin).total_seconds() / 3600 if v is not None else None
    return {
        **sim.metadata,
        "detected_before_failure": warning is not None,
        "ouranos_warning_hours": hours(warning),
        "anomaly_detection_hours": hours(anomaly),
        "conventional_alert_hours": hours(baseline),
        "failure_hours": hours(failure),
        "degradation_start_hours": hours(onset),
        "reference_detectable_hours": hours(first_time(truth.reference_detectable, t)),
        "lead_time_hours": lead,
        "false_alert_events": false_events,
        "healthy_exposure_hours": duration,
        "warning_subsystem": warning_subsystem,
        "subsystem_correct": warning_subsystem == expected_subsystem
        if warning is not None
        else None,
        **cm,
    }


def summarize(rows):
    faults = [r for r in rows if r["scenario"] != "normal"]
    detected = sum(r["detected_before_failure"] for r in faults)
    paired = [r["lead_time_hours"] for r in faults if r["lead_time_hours"] is not None]
    healthy_hours = sum(r["healthy_exposure_hours"] for r in rows)
    false_events = sum(r["false_alert_events"] for r in rows)
    n = len(faults)
    interval = (
        [
            float(beta.ppf(0.025, detected, n - detected + 1)) if detected else 0.0,
            float(beta.ppf(0.975, detected + 1, n - detected)) if detected < n else 1.0,
        ]
        if n
        else [None, None]
    )
    counts = {k: sum(r[k] for r in rows) for k in ["tp", "fp", "tn", "fn"]}
    tp, fp, tn, fn = [counts[k] for k in ["tp", "fp", "tn", "fn"]]
    safe = lambda a, b: a / b if b else None
    return {
        "degradation_runs": n,
        "healthy_control_runs": sum(r["scenario"] == "normal" for r in rows),
        "detected_before_failure": detected,
        "event_recall": safe(detected, n),
        "event_recall_95pct_interval": interval,
        "paired_lead_time_runs": len(paired),
        "early_warning_runs": sum(v > 0 for v in paired),
        "median_lead_time_hours": float(np.median(paired)) if paired else None,
        "lead_time_p10_hours": float(np.quantile(paired, 0.1)) if paired else None,
        "lead_time_p90_hours": float(np.quantile(paired, 0.9)) if paired else None,
        "false_alert_events": false_events,
        "healthy_exposure_hours": healthy_hours,
        "false_alerts_per_1000_hours": safe(false_events * 1000, healthy_hours),
        "zero_false_alert_poisson_95pct_upper_per_1000h": float(
            -np.log(0.05) / healthy_hours * 1000
        )
        if false_events == 0 and healthy_hours
        else None,
        "subsystem_accuracy_at_warning": safe(
            sum(r["subsystem_correct"] is True for r in faults), detected
        ),
        "point_metrics": {
            "counts": counts,
            "precision": safe(tp, tp + fp),
            "recall": safe(tp, tp + fn),
            "true_positive_rate": safe(tp, tp + fn),
            "false_positive_rate": safe(fp, fp + tn),
        },
        "policy": "Point positives start at injected degradation onset (includes subtle undetectable phase). Warm-up and post-failure excluded. False alerts count warning episode starts during healthy exposure. Lead time uses paired observed alerts; misses are separately reported.",
        "source": "SIMULATED DATA",
        "not_a_real_mission_validation": True,
    }


def evaluate(model, runs=100, healthy_runs=100, seed_start=1000, out=Path("reports")):
    if runs < 1 or healthy_runs < 1:
        raise ValueError("Both fault and healthy runs must be positive")
    if seed_start <= 29:
        raise ValueError(
            "Evaluation seed range must not overlap training/calibration seeds"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(runs + healthy_runs):
        seed = seed_start + i
        rng = np.random.default_rng(seed)
        scenario = (
            ("reaction_wheel", "battery", "solar")[i % 3] if i < runs else "normal"
        )
        sim = generate(
            scenario,
            seed=seed,
            onset_hours=float(rng.uniform(36, 60)),
            duration_hours=float(rng.uniform(100, 165)),
            noise_scale=float(rng.uniform(0.8, 1.25)),
        )
        result = analyze(sim.telemetry, model)
        rows.append(run_metrics(sim, result))
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{runs + healthy_runs}", flush=True)
    summary = summarize(rows)
    summary["by_scenario"] = {
        s: summarize([r for r in rows if r["scenario"] == s])
        for s in ("reaction_wheel", "battery", "solar")
    }
    summary["model"] = model.training_metadata
    summary["evaluation_seed_range"] = [
        seed_start,
        seed_start + runs + healthy_runs - 1,
    ]
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "by_scenario"}, indent=2),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--healthy-runs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    path = Path("models/healthy.joblib")
    model = joblib.load(path) if path.exists() else train_default(path)
    evaluate(model, args.runs, args.healthy_runs, args.seed_start, Path(args.out))
