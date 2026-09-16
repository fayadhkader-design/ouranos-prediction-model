"""Nuisance stress checks, intentionally outside the clean evaluation distribution."""

import json
from pathlib import Path
import joblib
import numpy as np
from src.pipeline import analyze
from src.simulation.generator import generate
from src.evaluation import event_starts


def run():
    model = joblib.load("models/healthy.joblib")
    rows = []
    for case in [
        "nominal",
        "single_spike",
        "fourfold_sensor_noise",
        "unmodeled_sensor_drift",
        "unmodeled_load_step",
    ]:
        sim = generate(
            seed=3001,
            hours=240,
            noise_scale=4 if case == "fourfold_sensor_noise" else 1,
        )
        df = sim.telemetry
        if case == "single_spike":
            df.loc[900, "reaction_wheel_2_current"] += 0.8
            df.loc[900, "reaction_wheel_2_temperature"] += 20
        elif case == "unmodeled_sensor_drift":
            drift = np.maximum(np.arange(len(df)) / 12 - 48, 0) / 150
            df["reaction_wheel_2_current"] += 0.12 * drift
            df["reaction_wheel_2_temperature"] += 2 * drift
        elif case == "unmodeled_load_step":
            df.loc[900:, "reaction_wheel_2_current"] += 0.10
            df.loc[900:, "reaction_wheel_2_temperature"] += 3
        r = analyze(df, model)
        rows.append(
            {
                "case": case,
                "max_risk": float(r.scores.risk.max()),
                "warning_episodes": int(event_starts(r.scores.warning).sum()),
                "first_warning_hours": float(np.flatnonzero(r.scores.warning)[0] / 12)
                if r.scores.warning.any()
                else None,
            }
        )
    report = {
        "source": "SIMULATED DATA",
        "interpretation": "Sensor drift and unmodeled loads can resemble degradation. Stress cases have no simulated equipment failure and warnings may be nuisance alarms; these are separate from the frozen nominal evaluation.",
        "cases": rows,
    }
    Path("reports/stress.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
