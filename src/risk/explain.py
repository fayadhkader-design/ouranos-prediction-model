import numpy as np
from src.config import UNITS

ISSUES = {
    "ADCS": "Reaction-wheel / attitude-control departure",
    "BATTERY": "Battery behavior departure",
    "POWER": "Power-generation / distribution departure",
    "THERMAL": "Thermal behavior departure",
    "COMMUNICATIONS": "Radio thermal departure",
    "PROPULSION": "Propulsion telemetry departure",
}


def explain(result, index=-1):
    i = index if index >= 0 else len(result.telemetry) + index
    row = result.scores.iloc[i]
    group = row.subsystem
    channels = result.model.groups[group]
    importance = (
        result.trends["smoothed"].iloc[i][channels].abs().sort_values(ascending=False)
    )
    evidence = []
    for c in importance.head(5).index:
        actual = float(result.expanded.iloc[i][c])
        expected = float(result.expected.iloc[i][c])
        delta = actual - expected
        evidence.append(
            {
                "signal": c,
                "observed": actual,
                "expected": expected,
                "deviation": delta,
                "deviation_percent": 100 * delta / abs(expected)
                if abs(expected) > 1e-6
                else None,
                "residual_sigma": float(result.residuals.iloc[i][c]),
                "trailing_variance_change_percent": float(
                    (result.residuals[c].iloc[max(0, i - 23) : i + 1].var(ddof=0) - 1)
                    * 100
                ),
                "slope_sigma_per_hour": float(result.trends["slopes"].iloc[i][c]),
                "slope_units_per_hour": float(
                    result.trends["slopes"].iloc[i][c] * result.model.scale[c]
                ),
                "unit": UNITS.get(c, ""),
            }
        )
    # Estimated beginning is first sustained evidence visible by now, never injected truth.
    g = result.risk_details[group].iloc[: i + 1]
    marker = (g.severity > 0.15) & (g.persistence > 0.75)
    idx = np.flatnonzero(marker.to_numpy())
    observed_onset = (
        result.telemetry.timestamp.iloc[idx[0]].isoformat() if len(idx) else None
    )
    contributions = result.global_components.iloc[i].to_dict()
    previous = result.global_components.iloc[max(0, i - 1)]
    deltas = (result.global_components.iloc[i] - previous).to_dict()
    issue = ISSUES[group]
    if group == "ADCS":
        wheels = {
            w: sum(
                float(importance.get(f"reaction_wheel_{w}_{s}", 0))
                for s in ("current", "temperature", "rpm")
            )
            for w in (1, 2, 3)
        }
        issue = f"Possible Reaction Wheel {max(wheels, key=wheels.get)} degradation"
    return {
        "subsystem": group,
        "possible_issue": issue,
        "risk": float(row.risk),
        "confidence": float(row.confidence),
        "confidence_definition": "Fraction of available inputs and in-range operating context, discounted during warm-up; not calibrated diagnostic confidence.",
        "observed_evidence_onset": observed_onset,
        "trend": "Increasing departure"
        if row.degradation > 0.2
        else "No sustained directional evidence",
        "evidence": evidence,
        "contributions": contributions,
        "score_change": float(row.risk - result.scores.risk.iloc[max(0, i - 1)]),
        "contribution_changes": deltas,
        "reason": "Context regression residuals depart from healthy multi-sensor patterns. Isolation Forest measures joint novelty; trailing residual slopes and persistence distinguish sustained changes. Signal evidence is associative, not a causal diagnosis.",
    }
