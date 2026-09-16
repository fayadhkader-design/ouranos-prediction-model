# Ouranos

**A working, local spacecraft telemetry and emerging-degradation research MVP.**

Ouranos learns healthy operating relationships, detects departures, measures persistent trends, attributes evidence to subsystems, and calculates a transparent 0–100 risk index. The Streamlit dashboard runs the actual Python pipeline. It supports live synthetic replay, controlled injections, telemetry inspection, exact score-change audits, and measured comparisons with conventional thresholds.

**V0 is an experimental predictive-maintenance prototype, not flight-qualified spacecraft operations software.** Its risk score is not a calibrated probability of failure. It does not predict arbitrary satellite failures or provide validated remaining useful life.

## Measured result in this build

The included `reports/metrics.json` and `reports/runs.csv` contain actual results from 100 randomized degradation runs and 100 independent healthy controls, using evaluation seeds 1000–1199:

| Measure | Result |
| --- | ---: |
| Detected before simulated failure | 100 / 100 |
| Warned before conventional threshold | 100 / 100 |
| Median paired warning lead | 85.25 simulated hours |
| 10th–90th percentile lead | 66.07–113.13 hours |
| Correct leading subsystem at warning | 100 / 100 |
| False warning episodes in nominal exposure | 0 |
| Healthy exposure, including pre-injection intervals | 27,632.17 hours |
| False episodes per 1,000 healthy hours | 0 observed |
| Point precision / recall | 1.000 / 0.860 |
| Point false-positive rate | 0.000 |

These are **controlled same-simulator results**, not real-mission validation. The detector benefits from accurately observed operating context and low sensor noise, while injected faults are sustained and coherent. Baseline thresholds are illustrative synthetic engineering limits, not a spacecraft manufacturer's certified limits. All three fault types are covered, but the simulator family is shared between training and testing. This is an easy demonstration distribution.

Zero observed false alarms is not proof of zero false-alarm risk. Under a Poisson stationary-event approximation, the one-sided 95% upper bound is 0.1084 events per 1,000 healthy hours; correlated telemetry and distribution shift limit that interpretation. Event-recall 95% exact binomial interval is approximately 96.38–100%, also conditional on this simulator.

**Harder nuisance checks exposed weaknesses:** an isolated two-sensor spike and fourfold sensor noise produced no warning episodes, but unmodeled sensor drift produced 5 episodes and an unmodeled load step produced 4. These cases contain no equipment failure. They show both cause ambiguity and alert chattering. Results are in `reports/stress.json` and visible in the dashboard. Operationally acceptable false-alarm performance remains unproven.

## Install and run

Python 3.12+ recommended; the delivered build was run with Python 3.14. Dependencies were installed in the workspace's `work/venv`. The runnable model and reports are included; no internet or mission dataset is required for the demo.

From this `ouranos` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.cli train
python -m streamlit run app.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. Alternatively, `./run.sh` uses `.venv`, falls back to the already-installed workspace environment, or creates a new environment and installs requirements. Pass `--server.port 8502` to the launcher if 8501 is occupied.

`requirements-lock.txt` records the exact environment used for this build. Cross-platform or older Python installations may need the bounded requirements instead of the exact lock. Retrain the model when changing scikit-learn versions; joblib artifacts are version-dependent and should be loaded only from trusted sources.

## Demo walkthrough

1. The app opens paused after 48 simulated healthy hours. Inspect the nominal risk trace and learned expected telemetry.
2. Click **INJECT REACTION WHEEL DEGRADATION**. Injection starts at the current cursor, preserves all preceding telemetry, selects Wheel 2 current, and starts playback.
3. Watch current, temperature, RPM variability and attitude error depart from healthy relationships. Risk is computed at every five-minute sample and causally smoothed. At faster playback rates, the UI skips intermediate samples, so displayed steps can be larger.
4. A warning requires risk at least 45 for six consecutive samples. Inspect **WHY DID THE SCORE CHANGE?** or the evidence tab for residuals, physical slopes, contribution deltas and data/context support.
5. Continue until the conventional threshold is crossed. Only then does the app display the measured early-warning advantage. It does not reveal a future alarm time during playback.
6. Continue to the scripted simulated-failure marker, or pause and scrub the replay. Reset before trying the battery or solar scenario.

For the default seed 42 reaction-wheel run: injection is at 48 h, Ouranos warning at 61.9167 h, conventional alert at 170.4167 h, and scripted failure at 198 h: **108.5 h measured lead**. These values are calculated from the run. Reset restores a clean healthy session and identical seeded data.

## Architecture

```text
Telemetry / optional ESA import
  → timestamp, cadence, gap and schema validation
  → operating-context regression trained on healthy data
  → standardized sensor + cross-sensor residuals
  → trailing multi-window time-series features
  → subsystem Isolation Forest novelty + directional trend evidence
  → subsystem risk contributions + causal smoothing
  → global Ouranos risk + sustained-warning policy
  → dashboard, exact score audit and evaluation reports
```

```text
ouranos/
  app.py                     Streamlit application
  data/raw/                  Optional externally downloaded mission files
  data/simulated/            Generated telemetry, labels and provenance
  data/processed/            Scores, subsystem results and score audit
  models/healthy.joblib      Trained regression and six Isolation Forests
  notebooks/                Small reproducible inspection notebook
  src/
    ingestion/              Strict schema, ESA adapters and mapping
    preprocessing/          Causal features and cross-sensor relationships
    detection/              Healthy models, trends and baseline rules
    risk/                   Risk policy and explanations
    simulation/             Generator and replay/reset state
    prognostics/            Future battery SOH/RUL interface
    dashboard/              Charts and visual styles
    pipeline.py             Fit/score orchestration
    cli.py                  Reproducible training and simulation commands
    evaluation.py           Randomized evaluation and metric definitions
    stress.py               Nuisance-distribution checks
  tests/                    Pipeline, metrics, ingestion and UI tests
  reports/                  Measured metrics, run-level CSV and stress results
  docs/ESA.md                Official-source research and loader instructions
  requirements.txt          Bounded application dependencies
  requirements-lock.txt     Exact tested environment
  run.sh                    Local launch helper
```

The backend is a reusable Python processing library and CLI, not a separate HTTP service. Streamlit calls this library directly. It is a single-user local prototype, without authentication, durable telemetry storage, message queues, spacecraft command uplink or an operational deployment configuration.

## Telemetry and simulation

`src/config.py` defines the extensible channel registry and six subsystems: BATTERY, POWER, THERMAL, ADCS, COMMUNICATIONS and PROPULSION. All requested telemetry channels are present. Additional context includes orbital sine/cosine, illumination, commanded load and encoded spacecraft mode. A directly simulated battery resistance proxy illustrates a future estimated engineering parameter; it is not inferred SOH.

Five-minute samples follow a 96-minute orbit, phase-shifted thermal response, random three-hour operating modes, variable loads and correlated electrical/thermal relationships. Training covers 30 healthy simulated days (seed 11); score normalization uses 14 independent healthy days (seed 29). Noise, orbit phase and mode scheduling are seeded. The model does not receive elapsed mission time, injected progress, scenario name or truth labels.

A smooth accelerating injection `progress ** 1.35` changes:

- **Reaction wheel:** Wheel 2 current and temperature rise, RPM noise grows, and attitude error increases.
- **Battery:** capacity declines, resistance proxy and charging temperature rise, voltage sags and becomes less stable, and charge state declines.
- **Solar:** output falls in proportion to illumination, with secondary bus-voltage/current effects. Eclipse is not mistaken for an output failure.

Truth labels are kept in a separate frame: degradation starts when the injected effect first becomes positive; an oracle detectability reference marks six samples with at least two imposed mean shifts exceeding three sensor-noise standard deviations; conventional threshold labels come from actual telemetry crossings; simulated failure occurs when injected progress reaches one. The oracle marker is an illustrative signal-to-noise reference, not an objective limit on learnability or a real fault annotation. A learned warning may precede it, particularly because of temporal averaging. Actual anomaly and warning times are independently measured.

The simulator is not an orbital mechanics, thermal-network, battery electrochemistry or reaction-wheel bearing model. It does not preserve a fully coupled energy budget. Scripted failure is a scenario endpoint; telemetry after it continues mathematically and is excluded from evaluation.

## Healthy behavior, features and trends

A multi-output Ridge regression learns each sensor and diagnostic relationship from healthy operating context. Residuals are divided by training residual standard deviations. Diagnostic relationships include wheel current per krpm, battery voltage/current coupling and solar output relative to illumination.

Trailing windows default to 6, 24 and 72 samples (0.5, 2 and 6 hours). Features include residual deviation, mean, standard deviation, min, max, per-hour rate, least-squares slopes and short/long variance ratio. The full transform produces 522 features. Each subsystem's Isolation Forest uses a compact subset: current residual, two-hour mean/standard deviation and six-hour slope, jointly across its channels. There is no deep-learning dependency.

Each forest has 64 trees, maximum 512 samples per tree and fixed seed 101. If `s` is negative forest score, its normalized anomaly score is:

```text
clip((s − healthy calibration median) / (2 × max(q99.9 − median, 0.01)), 0, 1)
```

Thus healthy calibration q99.9 maps to 0.5, not to a failure probability. Scores may saturate; no calibrated statistical tail probability is claimed.

The trend detector uses a six-hour residual slope, agreement of smoothed directional changes, and departure magnitude. It gates weak/incoherent slopes and averages the two strongest channel trend scores in each subsystem. Isolated spikes do not suffice in the supplied tests. Its generic absolute directional logic can also flag recovery, sensor drift or changed operating regimes; mission-specific semantics remain future work.

Validation rejects duplicate, unordered or irregular timestamps. Up to two missing samples can be forward-filled from the past; missingness lowers support. Leading missing values and longer gaps raise errors. No centered rolling windows, backward filling or future-fitted normalization is used. The dashboard caches an entire deterministic run for fast replay, but all transforms are causal; a prefix-equivalence test verifies that future samples do not change earlier results. UI event markers and downloads are restricted to the visible prefix.

## Transparent risk score

Each subsystem combines bounded terms:

| Component | Weight | Interpretation |
| --- | ---: | --- |
| Anomaly | 25% | Multivariate Isolation Forest novelty |
| Degradation | 30% | Sustained residual slope, coherence and magnitude |
| Severity | 20% | Average of two strongest smoothed residual departures above 2.5σ, saturating at 10σ |
| Persistence | 15% | Fraction of the last two hours with a smoothed residual above 3σ |
| Urgency | 10% | Strongest six-hour slope / 2σ per hour, gated by degradation |

These weights are an interpretable V0 policy, not learned failure-risk calibration. Multiply component points by data/context support and apply a trailing exponential mean (span 12 samples). Risk is suppressed during the first 72 warm-up samples. Support is the fraction of valid input values times the fraction of context features within training ranges, ramped during warm-up. **Support is not diagnostic confidence or probability that the attribution is correct.** A value of 100% only means complete inputs and in-range context.

```text
Global risk = 0.85 × highest subsystem risk + 0.15 × mean subsystem risk
```

Categories: 0–24 NOMINAL, 25–44 WATCH, 45–64 ELEVATED, 65–84 HIGH, 85–100 CRITICAL. Boundaries, feature windows, weights and conventional limits live in `src/config.py`; retrain when changing model configuration. Warning policy: score ≥45 for six consecutive samples (six five-minute sample bins, approximately 30 minutes). Alert clear is currently immediate on dropping below threshold; there is no separate clear hysteresis, so borderline cases can chatter.

Exact global component points sum to the displayed unrounded risk. Their sample-to-sample deltas sum to the risk change, including changes in the leading subsystem. Both JSON explanations and CSV score audits are exportable. Signal evidence gives expected/observed values, physical deviations, percent change where defined, residual sigma, variance change and per-hour slopes. These are **associative diagnostic evidence**, not causal proofs or additive forest-feature attribution. First sustained evidence is estimated from observed residual severity/persistence, never copied from injected truth.

## Reproduce and test

```bash
python -m src.cli train
python -m src.cli simulate --scenario reaction_wheel --seed 42
python -m src.evaluation --runs 100 --healthy-runs 100 --seed-start 1000
python -m src.stress
python -m pytest -q
```

All 29 tests passed in the delivered build. Tests cover deterministic generation, degradation injection, schema/gaps, features, model bounds, held-out nominal operation, early warnings, subsystem attribution, exact explanation reconciliation, isolated spikes, causality, reset, metric edge cases, ESA fixture formats, refusal to apply the synthetic model to real data, and Streamlit injection/replay/reset.

Evaluation randomizes onset (36–60 h), degradation duration (100–165 h), sensor noise (0.8–1.25×), mode schedule and orbital phase. Fault types rotate evenly across runs. There is no evaluation-time retraining, threshold tuning or point adjustment.

- **Event recall:** fraction of fault runs with a new warning episode after onset and before scripted failure. Pre-onset warnings are never true positives.
- **Point metrics:** each eligible post-onset sample is positive, including the initially subtle phase. Warm-up and post-failure are excluded. Recall therefore differs from event recall.
- **False alerts:** new sustained-warning episodes during healthy exposure, not every sample while an alert is active.
- **Lead:** first conventional crossing minus first qualifying Ouranos warning. Positive means earlier. Missing comparisons remain null; paired sample count and misses are reported separately. Conventional alarms receive no dwell delay, favoring the baseline in that respect.
- **Anomaly detection time:** first six consecutive normalized-anomaly samples ≥0.5 during degradation, measured separately from the combined risk warning.

Default reports are reproducible. `--out reports/another-run --seed-start 4000` preserves the supplied reference report while running another experiment. `reports/runs.csv` includes per-run event times and confusion counts.

## Connecting real telemetry

See [docs/ESA.md](docs/ESA.md) for inspected official sources, download location, layouts, trusted-pickle handling and optional CSV/raw loaders. No ESA mission archive was downloaded and no mission telemetry is bundled. Tests for the adapter use clearly synthetic format fixtures.

The UI clearly labels SIMULATED DATA and supports a separate REAL MISSION DATA CSV preview. User uploads are not independently authenticated. ESA anonymous identifiers must not be guessed into physical subsystems. A verified mapping must specify physical name, unit, scale and subsystem; mission context and healthy intervals must also be supplied. Raw gaps remain missing under bounded zero-order hold. The current synthetic model explicitly refuses real-mission inference.

For an additional channel, extend the registry/subsystem membership and ingestion mapping, provide a simulator signal or real measurement, validate units and context dependencies, then retrain and re-evaluate. New real spacecraft types require a mission-specific model and alarm policy rather than reuse of synthetic limits.

## Battery SOH and remaining useful life

`src/prognostics/battery.py` defines a typed future interface for SOH, RUL intervals, confidence level and validation-report provenance. The untrained implementation raises an explicit error. There are no fake RUL values or trained-model claims. A future NASA battery integration requires cycle parsing, unit harmonization, cell-level held-out validation, uncertainty calibration and a domain-transfer study before orbital battery use.

## Known weaknesses and next milestones

1. Obtain real telemetry with verified context and physical mappings; audit gaps, label quality and rare nominal operations. Train and validate on chronological mission splits.
2. Compare broader conventional baselines and tune warning budgets on a distinct development set; reserve a final untouched mission test set.
3. Disambiguate sensor faults, regime changes, maneuvers and equipment degradation. Add hysteresis/event grouping to reduce repeated nuisance episodes.
4. Replace illustrative support with calibrated diagnostic uncertainty; evaluate channel-level localization and recovery behavior. Shared sensors and duplicated relationships can overweight evidence.
5. Add varied fault rates, temporary faults, multiple faults, outages, unseen modes and independently developed simulation dynamics. Compare alternative anomaly models and ablate the forest/trend contributions.
6. Add incremental streaming state, persistent event storage, authenticated APIs and performance monitoring. Current replay is cached batch computation with a causal prefix, not a deployed telemetry service.
7. Develop and validate the separate battery prognostics module before exposing SOH/RUL estimates.

The MVP answers the central question **yes for these three controlled synthetic scenarios**. It does **not** establish that false alarms are acceptable for spacecraft operations or that the same lead times transfer to real mission data.
