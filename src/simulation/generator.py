"""Illustrative coupled telemetry, not a spacecraft dynamics/physics model.
Truth is returned separately and is never passed to fitted models.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.config import DEFAULT

SCENARIOS = ("normal", "reaction_wheel", "battery", "solar")


@dataclass
class Simulation:
    telemetry: pd.DataFrame
    truth: pd.DataFrame
    metadata: dict


def generate(
    scenario="normal",
    seed=42,
    hours=240,
    onset_hours=48,
    duration_hours=150,
    cadence_minutes=5,
    noise_scale=1.0,
):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if (
        hours <= 0
        or cadence_minutes <= 0
        or onset_hours < 0
        or duration_hours <= 0
        or noise_scale <= 0
    ):
        raise ValueError("Invalid simulation parameters")
    rng = np.random.default_rng(seed)
    n = int(hours * 60 / cadence_minutes)
    if n < 2:
        raise ValueError("At least two telemetry samples required")
    h = np.arange(n) * cadence_minutes / 60
    phase = 2 * np.pi * h / 1.6 + rng.uniform(0, 2 * np.pi)
    sun = np.clip((np.sin(phase) + 0.20) / 1.20, 0, 1)
    thermal = np.sin(phase - 0.7)
    # Mode schedule is deliberately independent of degradation.
    blocks = rng.choice(
        [0, 1, 2, 3], size=int(np.ceil(hours / 3)) + 1, p=[0.38, 0.40, 0.19, 0.03]
    )
    modes = blocks[np.floor(h / 3).astype(int)]
    load = np.choose(modes, [0.30, 0.72, 0.92, 0.12]) + 0.035 * np.sin(phase * 0.5)
    noise = lambda scale: rng.normal(0, scale * noise_scale, n)
    q = (
        np.maximum((h - onset_hours) / duration_hours, 0)
        if scenario != "normal"
        else np.zeros(n)
    )
    d = q**1.35
    current = 2.1 + 3.3 * load - 3.0 * sun + noise(0.045)
    soc = 78 + 10 * np.sin(phase - 0.25) - 4 * load + noise(0.16)
    data = {
        "timestamp": pd.date_range(
            "2026-01-01", periods=n, freq=f"{cadence_minutes}min", tz="UTC"
        ),
        "orbital_sin": np.sin(phase),
        "orbital_cos": np.cos(phase),
        "illumination": sun,
        "commanded_load": load,
        "spacecraft_mode": np.array(["CRUISE", "SCIENCE", "DOWNLINK", "SAFE"])[modes],
        "mode_science": (modes == 1).astype(float),
        "mode_downlink": (modes == 2).astype(float),
        "mode_safe": (modes == 3).astype(float),
        "battery_current": current,
        "battery_state_of_charge": soc,
        "battery_capacity": 100 + noise(0.15),
        "battery_resistance_proxy": 0.08 + noise(0.0015),
        "battery_voltage": 28 + 0.019 * (soc - 78) - 0.08 * current + noise(0.025),
        "battery_temperature": 23 + 2.5 * thermal + 1.7 * load + noise(0.12),
        "solar_array_output": 1200 * sun + noise(3),
        "solar_array_temperature": 17 + 14 * thermal + noise(0.18),
        "bus_voltage": 28 - 0.13 * load + noise(0.035),
        "bus_current": 3 + 3.3 * load + noise(0.04),
        "cpu_temperature": 31 + 7 * load + 1.8 * thermal + noise(0.16),
        "radio_temperature": 25 + 9 * (modes == 2) + 1.6 * thermal + noise(0.15),
        "propulsion_pressure": 22 + 0.12 * thermal + noise(0.035),
        "thruster_temperature": 18 + 2 * thermal + noise(0.16),
        "attitude_error": 0.021 + 0.008 * load + noise(0.002),
    }
    for i in (1, 2, 3):
        data[f"reaction_wheel_{i}_rpm"] = (
            2400 + 150 * i + 320 * load + 70 * np.sin(phase) + noise(9)
        )
        data[f"reaction_wheel_{i}_current"] = (
            0.29 + 0.16 * load + 0.013 * i + 0.012 * np.sin(phase) + noise(0.005)
        )
        data[f"reaction_wheel_{i}_temperature"] = (
            27 + 3 * thermal + 3 * load + 0.7 * i + noise(0.12)
        )
    if scenario == "reaction_wheel":
        data["reaction_wheel_2_current"] += 0.55 * d
        data["reaction_wheel_2_temperature"] += 20 * d
        data["reaction_wheel_2_rpm"] += noise(90) * d
        data["attitude_error"] += 0.16 * d
    elif scenario == "battery":
        data["battery_capacity"] -= 30 * d
        data["battery_resistance_proxy"] += 0.13 * d
        data["battery_voltage"] -= 0.13 * d * current + np.abs(noise(0.7)) * d
        data["battery_temperature"] += 14 * d * (0.6 + 0.4 * sun)
        data["battery_state_of_charge"] -= 12 * d
    elif scenario == "solar":
        data["solar_array_output"] -= 1200 * sun * 0.55 * d
        data["bus_voltage"] -= 0.5 * d * sun
        data["battery_current"] += 0.5 * d * sun
    frame = pd.DataFrame(data)
    frame["source"] = "SIMULATED DATA"
    truth = pd.DataFrame(
        {
            "timestamp": frame.timestamp,
            "degradation": q > 0,
            "progress": q,
            "reference_detectable": False,
            "simulated_failure": q >= 1,
        }
    )
    # An oracle reference: at least two known injected mean shifts exceed 3 sensor
    # noise sigmas for six samples. This is never supplied to the detector.
    if scenario == "reaction_wheel":
        strength = np.column_stack([0.55 * d / 0.005, 20 * d / 0.12, 0.16 * d / 0.002])
    elif scenario == "battery":
        strength = np.column_stack(
            [30 * d / 0.15, 0.13 * d / 0.0015, 14 * d * (0.6 + 0.4 * sun) / 0.12]
        )
    elif scenario == "solar":
        strength = np.column_stack(
            [660 * d * sun / 3, 0.5 * d * sun / 0.035, 0.5 * d * sun / 0.045]
        )
    else:
        strength = np.zeros((n, 3))
    reference = pd.Series(((strength / noise_scale) > 3).sum(axis=1) >= 2)
    truth["reference_detectable"] = reference.rolling(6, min_periods=6).sum().eq(6)
    # Actual model warning and anomaly times are measured separately.
    from src.detection.baseline import threshold_events

    truth["conventional_threshold"] = threshold_events(frame, DEFAULT).any(axis=1)
    return Simulation(
        frame,
        truth,
        {
            "scenario": scenario,
            "seed": seed,
            "hours": hours,
            "onset_hours": onset_hours if scenario != "normal" else None,
            "duration_hours": duration_hours,
            "noise_scale": noise_scale,
            "cadence_minutes": cadence_minutes,
            "source": "SIMULATED DATA",
        },
    )
