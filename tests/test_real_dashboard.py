"""Integration checks for the actual saved-model dashboard when artifacts exist."""
from pathlib import Path
import pytest
pytest.importorskip('torch')
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

@pytest.mark.skipif(not Path('models/esa_contextual/lifecycle.json').exists(),reason='Real-model artifacts not installed')
def test_real_dashboard_replay_reset():
    app=AppTest.from_string('from src.dashboard.real_mission import render\nrender()',default_timeout=60).run()
    assert not app.exception
    assert not app.exception
    assert app.title[0].value=='ESA Mission 1 · Anomaly workbench'
    assert app.selectbox[0].value=='id_184'
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state.real_cursor==1
    app.selectbox[0].select('id_145').run()
    assert not app.exception
    assert app.selectbox[0].value=='id_145'
