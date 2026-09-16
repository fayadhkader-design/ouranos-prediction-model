from dataclasses import dataclass
from pathlib import Path
import joblib
import pandas as pd
from src.simulation.generator import generate
from src.ingestion.schema import validate_telemetry
from src.detection.model import NormalBehaviorModel
from src.detection.baseline import threshold_events
from src.detection.trends import trend_evidence
from src.preprocessing.features import relationship_signals
from src.risk.engine import calculate_risk


@dataclass
class Result:
    telemetry: pd.DataFrame
    expanded: pd.DataFrame
    scores: pd.DataFrame
    subsystems: pd.DataFrame
    global_components: pd.DataFrame
    subsystem_components: dict
    risk_details: dict
    residuals: pd.DataFrame
    expected: pd.DataFrame
    trends: dict
    baseline: pd.DataFrame
    model: NormalBehaviorModel


def train_default(path=None):
    # Separate seeds and chronological sequences, never evaluate against training seeds.
    train = generate(seed=11, hours=24 * 30).telemetry
    calibration = generate(seed=29, hours=24 * 14).telemetry
    model = NormalBehaviorModel().fit(train, calibration)
    model.training_metadata.update(
        training_seed=11, calibration_seed=29, training_days=30, calibration_days=14
    )
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    return model


def analyze(telemetry, model):
    df, quality = validate_telemetry(
        telemetry, cadence_minutes=model.config.cadence_minutes
    )
    if "source" not in df or not df.source.eq("SIMULATED DATA").all():
        if model.training_metadata["training_source"] == "SIMULATED DATA":
            raise ValueError(
                "Synthetic model cannot score real mission data. Fit and validate a mission-specific model first."
            )
    z, expected, features, anomaly, support = model.score(df)
    trends = trend_evidence(z, model.config)
    scores, subsystems, components, global_components, details = calculate_risk(
        z, anomaly, trends, model.groups, quality, support, model.config
    )
    scores.insert(0, "timestamp", df.timestamp)
    return Result(
        df,
        relationship_signals(df),
        scores,
        subsystems,
        global_components,
        components,
        details,
        z,
        expected,
        trends,
        threshold_events(df, model.config),
        model,
    )
