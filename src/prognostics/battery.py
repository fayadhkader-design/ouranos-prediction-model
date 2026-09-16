"""Extension boundary: no untrained SOH/RUL numbers are exposed."""

from dataclasses import dataclass
from typing import Protocol
import pandas as pd


@dataclass(frozen=True)
class BatteryPrognosis:
    state_of_health: float
    rul_cycles_interval: tuple[float, float]
    confidence_level: float
    validation_report: str


class BatteryPrognosticsModel(Protocol):
    validated: bool

    def fit(self, cycles: pd.DataFrame) -> None: ...
    def predict(self, cycles: pd.DataFrame) -> BatteryPrognosis: ...


class UntrainedBatteryPrognostics:
    validated = False

    def predict(self, cycles):
        raise RuntimeError(
            "RUL unavailable: a real battery model must be trained and independently validated"
        )
