# Build validation — 2026-09-16

- Dependencies installed in workspace `work/venv`; `pip check` returned no broken requirements.
- Healthy regression and six Isolation Forests trained; artifact saved to `models/healthy.joblib`.
- All three fault scenarios and nominal controls executed through the actual pipeline.
- 100 randomized degradation runs + 100 randomized healthy controls evaluated; full per-run data saved.
- 29 pytest tests passed, including Streamlit AppTest injection, pause, replay, measured lead and reset.
- Prefix equivalence verifies future telemetry does not change earlier inference.
- Exact component sums and deltas reconcile with risk scores; no non-finite explanation values.
- Nuisance stress checks executed and disclosed separately in `stress.json` and dashboard.
- Streamlit launched locally; HTTP health returned `ok`.
- Browser interaction verified live injection, rising score, Wheel 2 attribution, pause/replay, conventional alert and calculated 108.5-hour lead for seed 42.
- Desktop 1440×1000 and mobile 375×812 layouts visually checked; temporary viewport reset.
- Rendering error in the completed-alert path corrected; full tests passed after the correction.
- No real ESA archive was downloaded. Adapter tests use synthetic format fixtures.

This is a working synthetic research MVP, not evidence of flight readiness or validated real-mission predictive performance. The stress checks demonstrate sensor-drift and unmodeled-load nuisance warnings.
