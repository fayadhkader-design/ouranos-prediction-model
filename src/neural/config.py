"""Predeclared real-data experiment, independent of test outcomes."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NeuralConfig:
    mission: str = "ESA-Mission1"
    channels: tuple = tuple(f"channel_{i}" for i in range(41, 47))
    cadence_seconds: int = 30
    max_hold_seconds: int = 300
    window: int = 64
    train_stride: int = 8
    train_end: str = "2006-07-01T00:00:00Z"
    validation_end: str = "2006-10-01T00:00:00Z"
    calibration_end: str = "2007-01-01T00:00:00Z"
    hidden: int = 64
    bottleneck: int = 12
    max_train_windows: int = 60000
    max_validation_windows: int = 12000
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 6
    learning_rate: float = 0.001
    threshold_quantile: float = 0.995
    warning_dwell: int = 3
    seed: int = 2026

    def to_dict(self):
        return asdict(self)


DEFAULT = NeuralConfig()
