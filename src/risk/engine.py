"""Deterministic score contributions reconcile exactly with the displayed score."""

import numpy as np
import pandas as pd
from src.config import DEFAULT


def category(score, config=DEFAULT):
    return next(label for bound, label in config.categories if score < bound)


def sustained(mask, count):
    return mask.astype(int).rolling(count, min_periods=count).sum().eq(count)


def top_mean(frame):
    k = min(2, frame.shape[1])
    return pd.Series(
        np.partition(frame.to_numpy(), -k, axis=1)[:, -k:].mean(axis=1),
        index=frame.index,
    )


def calculate_risk(z, anomaly, trends, groups, quality, support, config=DEFAULT):
    scores = {}
    components = {}
    details = {}
    weights = config.weights
    confidence = (quality * support).clip(0, 1)
    # Confidence means data/context support only, never probability of correctness/failure.
    readiness = np.minimum(np.arange(len(z)) / max(config.warmup, 1), 1)
    confidence = confidence * readiness
    for group, channels in groups.items():
        level = trends["smoothed"][channels].abs()
        sev = top_mean(((level - 2.5) / 7.5).clip(0, 1))
        deg = top_mean(trends["channel_degradation"][channels])
        pers = (
            (level.max(axis=1) > 3)
            .rolling(config.windows[1], min_periods=config.windows[1])
            .mean()
            .fillna(0)
        )
        urgency = (trends["slopes"][channels].abs().max(axis=1) / 2).clip(0, 1) * deg
        raw = pd.DataFrame(
            {
                "anomaly": anomaly[group],
                "degradation": deg,
                "severity": sev,
                "persistence": pers,
                "urgency": urgency,
            }
        )
        points = raw.mul(pd.Series(weights)) * 100
        points = points.mul(confidence, axis=0).ewm(span=12, adjust=False).mean()
        points.iloc[: config.warmup] = 0
        components[group] = points
        scores[group] = points.sum(axis=1).clip(0, 100)
        details[group] = raw
    subsystems = pd.DataFrame(scores)
    leader = subsystems.idxmax(axis=1)
    global_score = 0.85 * subsystems.max(axis=1) + 0.15 * subsystems.mean(axis=1)
    # Decompose global score across exactly the same leader + fleet-mean weighting.
    global_components = sum(components.values()) * (0.15 / len(components))
    for group in components:
        mask = leader == group
        global_components.loc[mask] += components[group].loc[mask] * 0.85
    output = pd.DataFrame(
        {
            "risk": global_score,
            "subsystem": leader,
            "confidence": confidence,
            "anomaly": anomaly.max(axis=1),
            "degradation": pd.DataFrame(
                {g: r.degradation for g, r in details.items()}
            ).max(axis=1),
        }
    )
    output["status"] = [category(v, config) for v in output.risk]
    output["warning"] = sustained(
        output.risk >= config.warning_threshold, config.warning_dwell
    )
    output["ready"] = np.arange(len(z)) >= config.warmup
    return output, subsystems, components, global_components, details
