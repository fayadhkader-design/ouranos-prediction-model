import json
import numpy as np
import pandas as pd
import pytest
from src.config import CHANNELS, CONTEXT, SUBSYSTEMS, Config
from src.simulation.generator import generate
from src.pipeline import analyze
from src.preprocessing.features import engineer
from src.risk.explain import explain
from src.risk.engine import category, sustained
from src.detection.model import NormalBehaviorModel
from src.simulation.session import SimulationSession
from src.ingestion.schema import validate_telemetry
from src.evaluation import confusion, first_time, event_starts, run_metrics, summarize
from src.prognostics.battery import UntrainedBatteryPrognostics


def test_generation_schema_and_reproducibility():
    a = generate(seed=40, hours=72).telemetry
    b = generate(seed=40, hours=72).telemetry
    pd.testing.assert_frame_equal(a, b)
    assert set(CHANNELS + CONTEXT + ["timestamp", "spacecraft_mode"]) <= set(a)
    assert a.source.eq("SIMULATED DATA").all()
    assert np.isfinite(a[CHANNELS + CONTEXT]).all().all()
    assert a.solar_array_output.corr(a.illumination) > 0.99
    assert len(a.spacecraft_mode.unique()) >= 3


@pytest.mark.parametrize(
    "scenario,channel,direction",
    [
        ("reaction_wheel", "reaction_wheel_2_current", 1),
        ("battery", "battery_capacity", -1),
        ("solar", "solar_array_output", -1),
    ],
)
def test_injection_preserves_healthy_history(scenario, channel, direction):
    healthy = generate(hours=120, seed=50).telemetry
    fault = generate(scenario, hours=120, seed=50, onset_hours=48, duration_hours=60)
    pd.testing.assert_frame_equal(healthy.iloc[:577], fault.telemetry.iloc[:577])
    assert (
        direction
        * (
            fault.telemetry[channel].iloc[-300:].mean()
            - healthy[channel].iloc[-300:].mean()
        )
        > 0
    )
    assert fault.truth.degradation.sum() > 0 and fault.truth.simulated_failure.any()
    assert not set(fault.truth.columns[1:]) & set(fault.telemetry.columns)


def test_features_are_past_only_and_slopes_have_units():
    z = pd.DataFrame({"x": np.arange(100) / 12})
    f = engineer(z, (6, 12, 24))
    fp = engineer(z.iloc[:60], (6, 12, 24))
    pd.testing.assert_frame_equal(f.iloc[:60], fp)
    assert np.allclose(f.x__slope_24.iloc[24:], 1)
    assert f.x__std_6.iloc[20] > 0
    assert np.isfinite(f).all().all()


def test_no_future_leakage(model):
    df = generate(
        "battery", seed=81, hours=120, onset_hours=30, duration_hours=70
    ).telemetry
    full = analyze(df, model)
    prefix = analyze(df.iloc[:900], model)
    np.testing.assert_allclose(
        full.scores.risk.iloc[:900], prefix.scores.risk, rtol=1e-10
    )
    np.testing.assert_allclose(full.residuals.iloc[:900], prefix.residuals)
    assert set(model.channels).isdisjoint(
        {"degradation", "reference_detectable", "simulated_failure", "progress"}
    )


@pytest.mark.parametrize(
    "scenario,group",
    [("reaction_wheel", "ADCS"), ("battery", "BATTERY"), ("solar", "POWER")],
)
def test_early_warning_and_attribution(model, scenario, group):
    sim = generate(scenario, seed=61)
    r = analyze(sim.telemetry, model)
    m = run_metrics(sim, r)
    assert m["detected_before_failure"] and m["lead_time_hours"] > 0
    assert m["warning_subsystem"] == group
    assert r.scores.risk.between(0, 100).all()
    assert r.scores.anomaly.between(0, 1).all()
    assert r.scores.degradation.between(0, 1).all()
    assert np.max(np.abs(r.scores.risk.diff().fillna(0))) < 10
    assert r.subsystems.columns.tolist() == list(SUBSYSTEMS)
    assert np.isfinite(r.subsystems).all().all()


def test_healthy_control(model):
    r = analyze(generate(seed=66).telemetry, model)
    assert not r.scores.warning.any()
    assert not r.baseline.any().any()
    assert r.scores.risk.max() < 25


def test_isolated_multisensor_spike_not_persistent_degradation(model):
    df = generate(seed=67, hours=120).telemetry
    df.loc[900, "reaction_wheel_2_current"] += 0.8
    df.loc[900, "reaction_wheel_2_temperature"] += 20
    r = analyze(df, model)
    assert r.baseline.any(axis=1).iloc[900]
    assert not r.scores.warning.any()
    assert r.scores.degradation.iloc[1100:].max() < 0.1


def test_score_explanations_reconcile_every_sample(model):
    r = analyze(
        generate(
            "reaction_wheel", seed=77, hours=120, onset_hours=24, duration_hours=70
        ).telemetry,
        model,
    )
    np.testing.assert_allclose(
        r.global_components.sum(axis=1), r.scores.risk, atol=1e-10
    )
    for i in (0, 72, 600, 1000):
        e = explain(r, i)
        assert sum(e["contributions"].values()) == pytest.approx(e["risk"])
        assert sum(e["contribution_changes"].values()) == pytest.approx(
            e["score_change"]
        )
        assert e["evidence"] and e["confidence_definition"]
        assert len(json.dumps(e, allow_nan=False)) > 0
    e = explain(r, 1000)
    assert "Wheel 2" in e["possible_issue"]


def test_reset_and_injection_causality():
    session = SimulationSession()
    before = session.data().telemetry
    session.cursor = 720
    session.inject("battery")
    session.running = True
    session.advance(12)
    assert session.onset_hours == 60 and session.cursor == 732
    pd.testing.assert_frame_equal(
        session.data().telemetry.iloc[:720], before.iloc[:720]
    )
    with pytest.raises(ValueError):
        session.inject("solar")
    session.reset()
    assert (
        session.cursor == 576 and not session.running and session.scenario == "normal"
    )
    pd.testing.assert_frame_equal(session.data().telemetry, before)
    session.cursor = 2880
    with pytest.raises(ValueError):
        session.inject("battery")


def test_metric_calculations_and_missing_events():
    c = confusion([0, 0, 1, 1], [0, 1, 0, 1])
    assert (
        c["precision"] == 0.5 and c["recall"] == 0.5 and c["false_positive_rate"] == 0.5
    )
    assert confusion([0], [0])["precision"] is None
    t = pd.Series(pd.date_range("2026-01-01", periods=5, freq="h"))
    assert first_time([0, 1, 1, 0, 1], t) == t[1]
    assert first_time([0] * 5, t) is None
    assert event_starts([0, 1, 1, 0, 1]).sum() == 2
    assert sustained(pd.Series([0, 1, 1, 1, 0]), 3).tolist() == [
        False,
        False,
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError):
        confusion([1], [1, 0])


def test_warning_before_onset_is_not_true_positive(model):
    sim = generate("battery", hours=120, onset_hours=60, duration_hours=50)
    r = analyze(sim.telemetry, model)
    r.scores["warning"] = True
    metrics = run_metrics(sim, r)
    assert not metrics["detected_before_failure"] and metrics["lead_time_hours"] is None


def test_summary_does_not_discard_misses():
    base = {
        "scenario": "battery",
        "detected_before_failure": True,
        "lead_time_hours": 4.0,
        "healthy_exposure_hours": 100.0,
        "false_alert_events": 1,
        "subsystem_correct": True,
        "tp": 10,
        "fp": 2,
        "tn": 98,
        "fn": 0,
    }
    miss = {
        **base,
        "detected_before_failure": False,
        "lead_time_hours": None,
        "false_alert_events": 0,
        "subsystem_correct": None,
        "tp": 0,
        "fp": 0,
        "tn": 100,
        "fn": 10,
    }
    report = summarize([base, miss])
    assert report["event_recall"] == 0.5 and report["paired_lead_time_runs"] == 1
    assert (
        report["false_alerts_per_1000_hours"] == 5
        and report["median_lead_time_hours"] == 4
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "NOMINAL"),
        (24.99, "NOMINAL"),
        (25, "WATCH"),
        (45, "ELEVATED"),
        (65, "HIGH"),
        (85, "CRITICAL"),
        (100, "CRITICAL"),
    ],
)
def test_categories(value, expected):
    assert category(value) == expected


def test_validation_and_causal_fill():
    df = generate(hours=24).telemetry
    bad = df.copy()
    bad.loc[10, "battery_voltage"] = np.nan
    good, q = validate_telemetry(bad, cadence_minutes=5)
    assert (
        good.battery_voltage.iloc[10] == df.battery_voltage.iloc[9] and q.iloc[10] < 1
    )
    bad.loc[0, "battery_voltage"] = np.nan
    with pytest.raises(ValueError):
        validate_telemetry(bad)
    with pytest.raises(ValueError):
        validate_telemetry(df.iloc[::-1])
    with pytest.raises(ValueError):
        validate_telemetry(df.drop(columns="battery_current"))
    with pytest.raises(ValueError):
        validate_telemetry(df.drop(index=5), cadence_minutes=5)
    with pytest.raises(ValueError):
        validate_telemetry(pd.concat([df, df.iloc[[-1]]]))


def test_real_data_rejected_by_synthetic_model(model):
    df = generate(hours=24).telemetry
    df["source"] = "REAL MISSION DATA"
    with pytest.raises(ValueError, match="Synthetic model"):
        analyze(df, model)


def test_untrained_interfaces_and_bad_config():
    with pytest.raises(RuntimeError):
        UntrainedBatteryPrognostics().predict(pd.DataFrame())
    with pytest.raises(ValueError):
        NormalBehaviorModel().score(pd.DataFrame())
    with pytest.raises(ValueError):
        Config(weights={"anomaly": 2})
    with pytest.raises(ValueError):
        generate("unknown")
