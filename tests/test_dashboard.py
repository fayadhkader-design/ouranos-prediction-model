from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_dashboard_load_inject_explain_reset():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.session_state["session"].scenario == "normal"
    button(at, "INJECT REACTION WHEEL DEGRADATION").click().run()
    assert not at.exception
    assert at.session_state["session"].scenario == "reaction_wheel"
    assert at.session_state["session"].running
    # Pause then scrub to a known late interval; charts, explanations and event metrics must render.
    button(at, "PAUSE SIMULATION").click().run()
    at.slider(key="replay_hour").set_value(180).run()
    assert not at.exception
    assert any("OURANOS EARLY WARNING" in s.value for s in at.success)
    button(at, "RESET").click().run()
    assert not at.exception
    assert (
        at.session_state["session"].scenario == "normal"
        and at.session_state["session"].cursor == 576
    )
