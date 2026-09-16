"""Versioned, explicit V0 policy. Units are synthetic engineering units."""

from dataclasses import dataclass, field

SUBSYSTEMS = {
    "BATTERY": [
        "battery_voltage",
        "battery_current",
        "battery_temperature",
        "battery_state_of_charge",
        "battery_capacity",
        "battery_resistance_proxy",
    ],
    "POWER": [
        "solar_array_output",
        "solar_array_temperature",
        "bus_voltage",
        "bus_current",
    ],
    "THERMAL": [
        "cpu_temperature",
        "radio_temperature",
        "solar_array_temperature",
        "battery_temperature",
        "thruster_temperature",
    ],
    "ADCS": [
        *[
            f"reaction_wheel_{i}_{s}"
            for i in (1, 2, 3)
            for s in ("rpm", "current", "temperature")
        ],
        "attitude_error",
    ],
    "COMMUNICATIONS": ["radio_temperature"],
    "PROPULSION": ["propulsion_pressure", "thruster_temperature"],
}
CHANNELS = list(dict.fromkeys(c for cs in SUBSYSTEMS.values() for c in cs))
CONTEXT = [
    "orbital_sin",
    "orbital_cos",
    "illumination",
    "commanded_load",
    "mode_science",
    "mode_downlink",
    "mode_safe",
]
UNITS = {
    c: (
        "°C"
        if "temperature" in c
        else "rpm"
        if "rpm" in c
        else "A"
        if "current" in c
        else "V"
        if "voltage" in c
        else ""
    )
    for c in CHANNELS
}
UNITS.update(
    battery_capacity="Ah",
    battery_state_of_charge="%",
    battery_resistance_proxy="Ω",
    solar_array_output="W",
    attitude_error="deg",
    propulsion_pressure="bar",
)


@dataclass(frozen=True)
class Config:
    cadence_minutes: int = 5
    windows: tuple = (6, 24, 72)
    warning_threshold: float = 45.0
    warning_dwell: int = 6
    warmup: int = 72
    categories: tuple = (
        (25, "NOMINAL"),
        (45, "WATCH"),
        (65, "ELEVATED"),
        (85, "HIGH"),
        (101, "CRITICAL"),
    )
    weights: dict = field(
        default_factory=lambda: {
            "anomaly": 0.25,
            "degradation": 0.30,
            "severity": 0.20,
            "persistence": 0.15,
            "urgency": 0.10,
        }
    )
    thresholds: dict = field(
        default_factory=lambda: {
            "battery_temperature": ("high", 39.0),
            "battery_capacity": ("low", 76.0),
            "battery_voltage": ("low", 24.0),
            "reaction_wheel_2_current": ("high", 0.88),
            "reaction_wheel_2_temperature": ("high", 49.0),
            "attitude_error": ("high", 0.15),
            "solar_efficiency": ("low", 0.62),
            "cpu_temperature": ("high", 65.0),
            "radio_temperature": ("high", 60.0),
            "propulsion_pressure": ("low", 15.0),
        }
    )

    def __post_init__(self):
        if (
            self.cadence_minutes <= 0
            or len(self.windows) < 2
            or min(self.windows) < 2
            or tuple(sorted(self.windows)) != self.windows
        ):
            raise ValueError("Use positive cadence and increasing feature windows >= 2")
        if set(self.weights) != {
            "anomaly",
            "degradation",
            "severity",
            "persistence",
            "urgency",
        }:
            raise ValueError("Risk weights must name all five policy components")
        if (
            self.warning_dwell < 1
            or not 0 <= self.warning_threshold <= 100
            or self.warmup < max(self.windows)
        ):
            raise ValueError("Invalid warning policy or insufficient feature warm-up")
        if abs(sum(self.weights.values()) - 1) > 1e-9 or any(
            v < 0 for v in self.weights.values()
        ):
            raise ValueError("Risk weights must be nonnegative and sum to one")


DEFAULT = Config()
