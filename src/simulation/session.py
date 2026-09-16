"""Deterministic replay with causal injection and no pre-injection history mutation."""

from dataclasses import dataclass
from src.simulation.generator import generate


@dataclass
class SimulationSession:
    seed: int = 42
    hours: int = 240
    cursor: int = 576
    running: bool = False
    scenario: str = "normal"
    onset_hours: float = 48.0
    duration_hours: float = 150.0

    def reset(self):
        self.cursor = 576
        self.running = False
        self.scenario = "normal"
        self.onset_hours = 48.0
        self.duration_hours = 150.0

    def inject(self, scenario):
        if self.scenario != "normal":
            raise ValueError("Reset before injecting another degradation scenario")
        if self.cursor >= self.hours * 12:
            raise ValueError("Reset the completed simulation before injecting")
        self.scenario = scenario
        self.onset_hours = self.cursor / 12
        self.duration_hours = max(
            12.0, min(150.0, (self.hours - self.onset_hours) * 0.8)
        )

    def advance(self, samples=12):
        if self.running:
            self.cursor = min(self.hours * 12, self.cursor + samples)
        if self.cursor >= self.hours * 12:
            self.running = False

    def data(self):
        return generate(
            self.scenario, self.seed, self.hours, self.onset_hours, self.duration_hours
        )
