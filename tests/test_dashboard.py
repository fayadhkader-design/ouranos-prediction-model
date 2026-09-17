from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_dashboard_load_inject_explain_reset():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert not at.exception
    assert at.session_state["session"].scenario == "normal"
    button(at, "Reaction wheel degradation").click().run()
    assert not at.exception
    assert at.session_state["session"].scenario == "reaction_wheel"
    assert at.session_state["session"].running
    # Pause then scrub to a known late interval; charts, explanations and event metrics must render.
    button(at, "Pause").click().run()
    at.slider(key="replay_hour").set_value(180).run()
    assert not at.exception
    assert any("Early warning" in s.value for s in at.success)
    button(at, "Reset").click().run()
    assert not at.exception
    assert (
        at.session_state["session"].scenario == "normal"
        and at.session_state["session"].cursor == 576
    )


def test_guided_demo_and_measured_checkpoints():
    at=AppTest.from_file(str(APP),default_timeout=30).run()
    assert not at.exception
    assert len(at.sidebar.radio) == 0
    button(at,'Run scenario').click().run()
    assert not at.exception
    assert at.session_state['session'].scenario=='reaction_wheel'
    assert at.session_state['session'].running
    button(at,'First warning').click().run()
    assert not at.exception
    assert not at.session_state['session'].running
    assert any('Sustained warning' in x.value for x in at.warning)
    button(at,'Conventional alert').click().run()
    assert not at.exception
    assert any('Early warning' in x.value for x in at.success)
    button(at,'Healthy operations').click().run()
    assert not at.exception
    assert at.session_state['session'].cursor<48*12
