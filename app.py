"""Run with: streamlit run app.py. All displays use the real Python pipeline."""

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from src.pipeline import train_default, analyze
from src.simulation.generator import generate
from src.simulation.session import SimulationSession
from src.risk.explain import explain
from src.evaluation import first_time
from src.config import SUBSYSTEMS, UNITS
from src.dashboard.style import CSS
from src.dashboard.charts import risk_chart, telemetry_chart, CYAN, AMBER, RED, MUTED
from src.ingestion.esa import load_preprocessed_csv

ROOT = Path(__file__).parent
st.set_page_config(
    page_title="Ouranos · Mission health",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

source = st.sidebar.radio("Workspace", ["Demo", "ESA telemetry"], key="workspace_source")
if source == "ESA telemetry":
    if not (ROOT / "data/processed/esa_supervised/features.npy").exists():
        st.title("ESA telemetry workspace")
        st.info("ESA replay is not installed on this deployment. It requires the separately prepared mission dataset and neural-model dependencies. The Demo workspace is available.")
        st.link_button("Dataset setup instructions", "https://github.com/fayadhkader-design/ouranos-prediction-model#readme")
        st.stop()
    try:
        from src.dashboard.real_mission import render
        render()
    except ImportError as exc:
        st.error(f"Real-model dependencies are unavailable: {exc}. Install requirements-neural.txt.")
    st.stop()



@st.cache_resource
def get_model():
    path = ROOT / "models/healthy.joblib"
    return joblib.load(path) if path.exists() else train_default(path)


@st.cache_resource(max_entries=6)
def get_run(scenario, seed, onset, duration):
    sim = generate(scenario, seed=seed, onset_hours=onset, duration_hours=duration)
    return sim, analyze(sim.telemetry, get_model())


if "session" not in st.session_state:
    st.session_state.session = SimulationSession()
session = st.session_state.session


def reset():
    session.reset()
    st.session_state.pop("replay_hour", None)


def toggle_playback():
    session.running = not session.running
    if session.running:
        st.session_state.pop("replay_hour", None)


def inject(scenario):
    st.session_state.plot_group = {
        "battery": "BATTERY",
        "reaction_wheel": "ADCS",
        "solar": "POWER",
    }[scenario]
    st.session_state.plot_channel = {
        "battery": "battery_capacity",
        "reaction_wheel": "reaction_wheel_2_current",
        "solar": "solar_array_output",
    }[scenario]
    session.inject(scenario)
    session.running = True
    st.session_state.pop("replay_hour", None)


def start_guided(scenario):
    session.reset()
    session.seed = 42
    session.scenario = scenario
    session.cursor = 36 * 12
    session.running = True
    st.session_state.pop("replay_hour", None)
    st.session_state.plot_group = {"battery": "BATTERY", "reaction_wheel": "ADCS", "solar": "POWER"}[scenario]
    st.session_state.plot_channel = {"battery": "battery_capacity", "reaction_wheel": "reaction_wheel_2_current", "solar": "solar_array_output"}[scenario]


def jump_demo(stage, scenario):
    if session.scenario == "normal" or session.scenario != scenario:
        start_guided(scenario)
    session.running = False
    sim, result = get_run(session.scenario, session.seed, session.onset_hours, session.duration_hours)
    if stage == "healthy":
        session.cursor = max(72, int(session.onset_hours * 12) - 1)
    else:
        mask = result.scores.warning if stage == "warning" else result.baseline.any(axis=1)
        positions = mask.to_numpy().nonzero()[0]
        if len(positions):
            session.cursor = int(positions[0]) + 1
    st.session_state.pop("replay_hour", None)


with st.sidebar:
    st.markdown("### Scenario playback")
    demo_scenario = st.selectbox("Demo scenario", ["reaction_wheel", "battery", "solar"], format_func=lambda value: {"reaction_wheel": "Reaction wheel degradation", "battery": "Battery degradation", "solar": "Solar array degradation"}[value])
    st.button("Run scenario", type="primary", width="stretch", on_click=start_guided, args=(demo_scenario,))
    st.caption("About one minute at the default speed. Starts healthy, then injects gradual degradation at hour 48.")
    with st.expander("Jump to event"):
        st.button("Healthy operations", width="stretch", on_click=jump_demo, args=("healthy", demo_scenario))
        st.button("First warning", width="stretch", on_click=jump_demo, args=("warning", demo_scenario))
        st.button("Conventional alert", width="stretch", on_click=jump_demo, args=("threshold", demo_scenario))
        st.caption("Jumps to calculated events in the selected scenario.")
    st.divider()
    st.markdown("### Playback")
    st.caption("Manual controls")
    st.button(
        "Pause" if session.running else "Start playback",
        type="primary",
        width="stretch",
        on_click=toggle_playback,
    )
    st.button("Return to normal", on_click=reset, width="stretch")
    disabled = session.scenario != "normal" or session.cursor >= session.hours * 12
    st.markdown("#### Inject degradation")
    st.button(
        "Battery degradation",
        on_click=inject,
        args=("battery",),
        disabled=disabled,
        width="stretch",
    )
    st.button(
        "Reaction wheel degradation",
        on_click=inject,
        args=("reaction_wheel",),
        disabled=disabled,
        width="stretch",
    )
    st.button(
        "Solar array degradation",
        on_click=inject,
        args=("solar",),
        disabled=disabled,
        width="stretch",
    )
    st.caption(
        "Injection begins at the current playback time. Reset to compare a different scenario."
    )
    st.divider()
    speed = st.selectbox(
        "Playback speed",
        options=[1, 3, 6, 12],
        index=1,
        format_func=lambda x: f"{x} hours / second",
    )
    seed = st.number_input(
        "Random seed",
        min_value=30,
        max_value=999999,
        value=session.seed,
        step=1,
        disabled=session.running,
    )
    if seed != session.seed:
        session.seed = int(seed)
        reset()
    if not session.running:
        if st.button("Advance 12 hours", width="stretch"):
            session.cursor = min(2880, session.cursor + 144)
            st.session_state.pop("replay_hour", None)
        hour = st.slider(
            "Replay position · hours",
            min_value=6,
            max_value=240,
            value=max(6, int(session.cursor / 12)),
            key="replay_hour",
        )
        if int(hour) != max(6, int(session.cursor / 12)):
            session.cursor = int(hour) * 12
    st.button("Reset", on_click=reset, width="stretch")
    st.divider()
    st.caption("EXPERIMENTAL V0 · NOT FLIGHT QUALIFIED")
    st.caption(
        "Risk is an evidence index, not a probability of failure."
    )

st.markdown(
    '<div class="masthead"><span class="wordmark">Ouranos<span>Telemetry workbench</span></span><span class="provenance">Demo workspace · V0</span></div>',
    unsafe_allow_html=True,
)
st.caption("Review telemetry against expected behavior, then inspect the evidence behind each warning.")


@st.fragment(run_every=1.0 if session.running else None)
def dashboard():
    if session.running:
        session.advance(int(speed * 12))
        if not session.running:
            st.rerun()
    sim, result = get_run(
        session.scenario, session.seed, session.onset_hours, session.duration_hours
    )
    cursor = session.cursor
    i = cursor - 1
    row = result.scores.iloc[i]
    explanation = explain(result, i)
    view = result.scores.iloc[:cursor]
    time = sim.telemetry.timestamp.iloc[:cursor]
    origin = sim.telemetry.timestamp.iloc[0]
    hour_of = lambda ts: (
        (ts - origin).total_seconds() / 3600 if ts is not None else None
    )
    warning = hour_of(first_time(view.warning, time))
    baseline = hour_of(first_time(result.baseline.iloc[:cursor].any(axis=1), time))
    failure = hour_of(first_time(sim.truth.simulated_failure.iloc[:cursor], time))
    onset = (
        session.onset_hours
        if session.scenario != "normal" and i / 12 >= session.onset_hours
        else None
    )
    anomaly = hour_of(first_time((view.anomaly >= 0.5).rolling(6).sum().eq(6), time))
    leader = row.subsystem if row.risk >= 25 else "No sustained concern"
    color = RED if row.risk >= 65 else AMBER if row.risk >= 25 else CYAN
    top, clockcol = st.columns([4, 1])
    with top:
        st.markdown(
            '<div class="asset">SAT-001</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="score-row"><span class="score">{int(row.risk)}<small> / 100</small></span><div><div class="status" style="color:{color}">{row.status.title()}</div><div class="quiet">Ouranos Risk Score</div></div></div>',
            unsafe_allow_html=True,
        )
    with clockcol:
        st.markdown(
            f'<div class="quiet">Mission elapsed</div><div style="font:500 1.5rem ui-monospace;white-space:nowrap">{i / 12:.1f} h</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            ("STREAMING" if session.running else "PAUSED")
            + " · "
            + sim.telemetry.spacecraft_mode.iloc[i]
        )
    if failure is not None:
        st.error(
            "Scenario complete · This is a scripted scenario endpoint, not a predicted failure time."
        )
    elif warning is not None:
        st.warning(
            f"Sustained warning · Leading subsystem: {leader}. {explanation['possible_issue']}."
        )
    else:
        st.caption(
            "No sustained warning. Readings remain consistent with the learned baseline."
            if row.risk < 25
            else "WATCH · A departure is developing; the sustained-warning rule has not yet been met."
        )
    if baseline is not None:
        if warning is not None:
            lead = baseline - warning
            if lead > 0:
                st.success(
                    f"Early warning · Detected {lead:.1f} hours before conventional alert."
                )
            else:
                st.info(
                    f"No early-warning advantage: conventional alert preceded Ouranos by {-lead:.1f} hours."
                )
        else:
            st.error(
                "Conventional threshold alert reached. Ouranos has not issued a sustained warning."
            )
    elif warning is not None:
        st.info(
            f"Ouranos warning at {warning:.1f} h. Conventional limits have not alerted. Final lead time is not yet known."
        )
    overview, evidence, evaluation, data = st.tabs(
        ["Mission overview", "Evidence & score audit", "Evaluation", "Data & model"]
    )
    with overview:
        left, right = st.columns([3.2, 1], gap="large")
        with left:
            st.subheader("Risk trajectory")
            events = [
                ("Injection", onset, MUTED),
                ("Ouranos", warning, CYAN),
                ("Threshold", baseline, AMBER),
                ("Failure", failure, RED),
            ]
            st.plotly_chart(
                risk_chart(result, cursor, events), width="stretch", key="risk_history"
            )
            st.caption(
                "Dashed horizontal line: warning policy at 45, sustained for 30 minutes. All processing uses current and past samples."
            )
        with right:
            st.subheader("Subsystem risk")
            for group, value in result.subsystems.iloc[i].items():
                st.markdown(
                    f'<div class="subsystem-row"><span>{group if group == "ADCS" else group.title()}</span><strong>{value:.0f}</strong><div class="track"><span style="width:{value:.1f}%"></span></div></div>',
                    unsafe_allow_html=True,
                )
            st.caption(
                "Higher = greater risk. Shared sensors can affect more than one subsystem."
            )
            st.metric(
                "Data / context support",
                f"{row.confidence:.0%}",
                help=explanation["confidence_definition"],
            )
        st.subheader("Telemetry inspection")
        c1, c2, c3 = st.columns([2, 3, 1])
        group = c1.selectbox("Subsystem", list(SUBSYSTEMS), index=3, key="plot_group")
        channel = c2.selectbox(
            "Telemetry channel",
            SUBSYSTEMS[group],
            index=4 if group == "ADCS" else 0,
            format_func=lambda c: c.replace("_", " ").capitalize(),
            key="plot_channel",
        )
        window = c3.selectbox(
            "History", [24, 48, 120, 240], index=1, format_func=lambda h: f"{h} hours"
        )
        st.plotly_chart(
            telemetry_chart(result, cursor, channel, window),
            width="stretch",
            key="telemetry",
        )
        st.caption(
            f"{UNITS.get(channel, 'Engineering units')} · Healthy envelope is residual standard deviation, not a prediction confidence interval."
        )
        with st.expander("Why did the score change?"):
            delta = pd.DataFrame(
                {
                    "Current points": explanation["contributions"],
                    "Change since previous sample": explanation["contribution_changes"],
                }
            )
            st.dataframe(delta.round(3), width="stretch")
            st.write(
                f"Net change: {explanation['score_change']:+.3f} points. Contributions sum to {row.risk:.3f}."
            )
            st.caption(
                "These are exact risk-policy contributions. Sensor residuals below explain the evidence; they are not additive Isolation Forest feature attributions."
            )
            st.dataframe(
                pd.DataFrame(explanation["evidence"]), hide_index=True, width="stretch"
            )
    with evidence:
        st.subheader("Evidence behind the current assessment")
        st.write(
            explanation["possible_issue"]
            if row.risk >= 25
            else "No actionable subsystem diagnosis"
        )
        st.write(explanation["reason"])
        st.write(
            f"**Trend:** {explanation['trend']} · **First sustained evidence:** {explanation['observed_evidence_onset'] or 'Not established'}"
        )
        st.caption(explanation["confidence_definition"])
        st.dataframe(
            pd.DataFrame(explanation["evidence"]).round(4),
            hide_index=True,
            width="stretch",
        )
        st.subheader("Exact score composition")
        composition = pd.DataFrame(
            {
                "Points": explanation["contributions"],
                "Delta": explanation["contribution_changes"],
            }
        )
        st.dataframe(composition.round(4), width="stretch")
        st.caption(
            "Global score = 85% highest subsystem + 15% mean subsystem risk. Each subsystem = 25% anomaly + 30% degradation + 20% severity + 15% persistence + 10% urgency, support-weighted and causally smoothed."
        )
        st.subheader("Observed event log")
        events = {
            "Injected degradation": onset,
            "Sustained anomaly signal": anomaly,
            "Ouranos warning": warning,
            "Conventional alert": baseline,
            "Scenario endpoint": failure,
        }
        st.dataframe(
            pd.DataFrame(
                [
                    {"Event": k, "Elapsed hours": v, "Observed": v is not None}
                    for k, v in events.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        audit = result.global_components.iloc[:cursor].copy()
        audit.insert(0, "timestamp", time)
        for c in result.global_components:
            audit[c + "_change"] = (
                result.global_components[c].iloc[:cursor].diff().fillna(0)
            )
        st.download_button(
            "Download score audit · CSV",
            audit.to_csv(index=False),
            "ouranos_score_audit.csv",
            "text/csv",
        )
        st.download_button(
            "Download explanation · JSON",
            json.dumps(explanation, indent=2),
            "ouranos_explanation.json",
            "application/json",
        )
    with evaluation:
        st.subheader("Does the early warning hold across runs?")
        path = ROOT / "reports/metrics.json"
        if path.exists():
            report = json.loads(path.read_text())
            a, b, c = st.columns(3)
            a.metric(
                "Detected before scenario endpoint",
                f"{report['detected_before_failure']} / {report['degradation_runs']}",
            )
            b.metric(
                "Median measured lead time",
                f"{report['median_lead_time_hours']:.1f} h"
                if report["median_lead_time_hours"] is not None
                else "No paired detections",
            )
            c.metric(
                "False alerts / 1,000 healthy hours",
                f"{report['false_alerts_per_1000_hours']:.3f}",
            )
            st.caption(
                f"{report['healthy_control_runs']} healthy controls · {report['healthy_exposure_hours']:,.1f} healthy hours · seeds {report['evaluation_seed_range']} · DEMO"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Scenario": k,
                            "Detected": v["detected_before_failure"],
                            "Runs": v["degradation_runs"],
                            "Median lead (h)": v["median_lead_time_hours"],
                            "Subsystem accuracy": v["subsystem_accuracy_at_warning"],
                        }
                        for k, v in report["by_scenario"].items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.write("**Point-level metrics**", report["point_metrics"])
            if report["zero_false_alert_poisson_95pct_upper_per_1000h"] is not None:
                st.caption(
                    f"Zero observed false alerts is not proof of zero risk. Approximate one-sided 95% Poisson upper bound: {report['zero_false_alert_poisson_95pct_upper_per_1000h']:.3f} alerts / 1,000 hours; independence/stationarity assumptions apply."
                )
            st.info(
                "These results describe the included scenarios. Operational accuracy and false-alarm tolerance remain unvalidated."
            )
            stress_path = ROOT / "reports/stress.json"
            if stress_path.exists():
                stress = json.loads(stress_path.read_text())
                st.subheader("Harder nuisance checks")
                st.dataframe(
                    pd.DataFrame(stress["cases"]), hide_index=True, width="stretch"
                )
                st.warning(
                    "Sensor calibration drift and unmodeled load changes triggered warnings without an equipment-failure endpoint. The model cannot yet distinguish these causes; operational false-alarm acceptability remains unproven."
                )
            st.caption(report["policy"])
            st.download_button(
                "Download evaluation report",
                path.read_text(),
                "ouranos_metrics.json",
                "application/json",
            )
        else:
            st.info(
                "No evaluation report yet. Run: python -m src.evaluation --runs 100 --healthy-runs 100"
            )
    with data:
        st.subheader("Model provenance")
        st.download_button("Download model provenance · JSON", json.dumps(result.model.training_metadata, indent=2), "ouranos_model_provenance.json", "application/json")
        st.write(
            "The current model was fitted on 30 healthy operating days and normalized on 14 independent healthy days. Scenario labels are excluded from all model inputs."
        )
        st.download_button(
            "Download visible telemetry · CSV",
            sim.telemetry.iloc[:cursor].to_csv(index=False),
            "ouranos_demo_telemetry.csv",
            "text/csv",
        )
        st.subheader("ESA telemetry · optional import")
        st.write(
            "ESA-ADB CSV ingestion preserves anonymous channel names and separates annotations. Import previews are never scored using this demo model."
        )
        uploaded = st.file_uploader("ESA preprocessed CSV", type=["csv"])
        if uploaded:
            try:
                telemetry, labels = load_preprocessed_csv(uploaded)
                st.success(
                    f"ESA telemetry import · {len(telemetry):,} rows. Provenance is user-supplied and unverified."
                )
                st.dataframe(telemetry.head(100), hide_index=True, width="stretch")
                st.caption(
                    f"{len(labels.columns) - 1} annotation channels separated. Physical mapping, context inputs and a mission-specific validated model are still required."
                )
            except Exception as exc:
                st.error(f"Import could not be completed: {exc}")
        st.link_button(
            "Official ESA anomaly benchmark", "https://github.com/kplabs-pl/ESA-ADB"
        )
        st.subheader("Battery prognostics")
        st.write(
            "Remaining useful life is unavailable. The extension interface requires a trained and independently validated battery model before any SOH or RUL estimate is shown."
        )
    st.caption(
        "Ouranos V0 · Experimental predictive-maintenance prototype. No flight-qualification or calibrated failure-probability claim."
    )


dashboard()
