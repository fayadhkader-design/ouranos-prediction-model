from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_dashboard_load_inject_explain_reset():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Demo").run()
    assert not at.exception
    assert at.session_state["session"].scenario == "normal"
    button(at, "INJECT REACTION WHEEL DEGRADATION").click().run()
    assert not at.exception
    assert at.session_state["session"].scenario == "reaction_wheel"
    assert at.session_state["session"].running
    # Pause then scrub to a known late interval; charts, explanations and event metrics must render.
    button(at, "PAUSE PLAYBACK").click().run()
    at.slider(key="replay_hour").set_value(180).run()
    assert not at.exception
    assert any("OURANOS EARLY WARNING" in s.value for s in at.success)
    button(at, "RESET").click().run()
    assert not at.exception
    assert (
        at.session_state["session"].scenario == "normal"
        and at.session_state["session"].cursor == 576
    )


def test_guided_demo_and_measured_checkpoints():
    at=AppTest.from_file(str(APP),default_timeout=30).run()
    assert not at.exception
    assert at.sidebar.radio[0].value=='Demo'
    button(at,'RUN GUIDED DEMO').click().run()
    assert not at.exception
    assert at.session_state['session'].scenario=='reaction_wheel'
    assert at.session_state['session'].running
    button(at,'SHOW FIRST OURANOS WARNING').click().run()
    assert not at.exception
    assert not at.session_state['session'].running
    assert any('EMERGING DEGRADATION DETECTED' in x.value for x in at.warning)
    button(at,'SHOW CONVENTIONAL ALERT').click().run()
    assert not at.exception
    assert any('OURANOS EARLY WARNING' in x.value for x in at.success)
    button(at,'SHOW HEALTHY OPERATIONS').click().run()
    assert not at.exception
    assert at.session_state['session'].cursor<48*12
