# Ouranos: recent-context neural model, V3

A working real-telemetry anomaly detector is now available as a saved model and CSV inference command. It is a research candidate, not a validated spacecraft failure predictor. The Streamlit dashboard now has a separate real-mission workspace using this model and a calibrated recovery policy. See [the current release](REAL_MISSION_RELEASE.md) for integration and separate-mission results. The synthetic risk score remains separate.

## Result

The winner was selected on 2005–2006 calibration data before computing its 2007–2013 results. Those later years were already examined in prior experiments and influenced the decision to investigate context features: this is retrospective development evidence, not untouched validation.

| Candidate | Calibration events / 8 | Later events / 29 | False alert episodes / 1,000 nominal hours |
|---|---:|---:|---:|
| **Recent-context neural network (selected)** | **6** | **19** | **2.65** |
| Same architecture with feature augmentation | 5 | 6 | 0.033 |
| Histogram gradient-boosted trees | 0 | 2 | 0 |

Selection maximized calibration event detections subject to no more than 10 false episode starts per 1,000 nominal hours AND at most 0.1% of nominal minutes in alarm. Ties favor fewer false starts, then less time in alarm. A model that never alerts therefore cannot win just by having zero false alarms. Candidate thresholds and one- versus three-minute persistence were evaluated only on calibration data. Calibration is model-selection data, not an independent performance estimate.

The selected model's later-year results:

- 19/29 annotated events detected (65.5%); 28/29 have at least some scorable data.
- 160 false episodes over 60,406.62 scored nominal hours: 2.6487 per 1,000 hours.
- Minute precision 33.01%; minute recall **2.68%**.
- Nominal minute false-positive rate 0.01142%.
- Median detection delay among detected events: 0.604 minutes, about 36 seconds **after** annotated onset. This is not advance warning.
- 3,631,996 of 3,682,081 later-year minutes scored (98.64%). Missing data and insufficient history stay unscored.
- Detection uses right-closed minute bins. An event shorter than a minute can match the bin containing it, whose timestamp occurs after the event ended.

### Comparison on identical coverage

On 3,631,973 common scored minutes, each using its frozen calibration policy:

| Metric | Previous supervised network | Selected context network |
|---|---:|---:|
| Events with alert overlap | 20/29 | 19/29 |
| False episodes / 1,000 nominal hours | 1,560.20 | 2.65 |
| Minute precision | 0.268% | 33.01% |
| Minute recall | 53.56% | 2.69% |
| Nominal minutes flagged | 41.80% | 0.01142% |

This is a large reduction in nuisance alarms with similar event coverage, but a large loss in the fraction of anomalous time flagged. It is more useful for catching brief deviations or event onsets than continuously identifying every minute of an ongoing abnormal state. Different training features, sampling and calibration constraints mean this is a system comparison, not proof that one isolated change caused the improvement.

## What changed

The old network relied heavily on absolute levels, which changed across mission years. V3 compares each signal against its preceding hour and day. At timestamp t, those baselines use only observations strictly before t; extrema and variation from the current minute are available once that minute closes.

Ten features per channel (60 total) describe current last/min/max deviations from the preceding hour, within-minute standard deviation and range, minute-to-minute change, hour-versus-day baseline separation, and deviation from the preceding day. Long-window local standard deviation normalizes some features; a floor based on training-only signal scale prevents division by tiny values. Other features retain a fixed training scale, so the detector does not remove every sustained change immediately. There are no absolute-level inputs.

The neural network is 60 → 64 → 32 → 1 with GELU activations (6,017 parameters), trained with weighted binary cross entropy and AdamW on real annotation labels. Epoch 37 was selected using 2004 sampled validation average precision. The output is a raw logit ranking score, **not a probability**, failure estimate or Ouranos 0–100 risk score. Selected alert threshold: 7.8189931262; one closed minute above threshold raises an alert.

Training: 2000–2003, 123,210 eligible sampled rows including 3,487 positive rows. Negatives are randomly sampled; positives are capped at 500 rows per event. Validation: 2004, 40,494 sampled rows. Calibration: 2005–2006 full scorable stream. Later comparison: 2007–2013 full scorable stream. Global scales and neural scaling are fitted only using training data. Past context can cross a split boundary because it is already observed history; labels never enter inference features.

## Did synthetic data help?

Not in this experiment. We tested small, channel-coherent gain perturbations of training features (0.85–1.15 multipliers), representing a limited feature-space augmentation. It is not a physics-based spacecraft simulator or a generative model of real telemetry. Augmentation was applied only to training examples, never validation/calibration/evaluation data. Its candidate detected fewer calibration events and was not selected. The winning model trains on real telemetry without that augmentation.

## Run the actual model

From the project root, with `requirements-neural.txt` installed:

```bash
python -m src.neural.predict_contextual --input telemetry.csv --output predictions.csv
```

Input columns: `timestamp,channel_41,channel_42,channel_43,channel_44,channel_45,channel_46`. These are the six anonymous ESA Mission 1 channels in the training data, in their original numeric representation. Do not substitute unrelated battery/wheel measurements just because there are six columns. Timestamps must be unique and increasing; CSV timestamps are parsed as UTC. Supply the original subminute observations where available: pre-averaging or retaining only the last value loses short-event extrema.

The command bins observations into one-minute last/min/max/std statistics and preserves missing bins. It needs approximately one day of prior history (at least 90% coverage in the preceding 60- and 1,440-minute windows). `UNSCORED` means insufficient/missing data, not healthy. Output contains a model score, threshold, `ANOMALY` / `NO ALERT` / `UNSCORED`, and the channel with the largest standardized recent deviation. That channel is descriptive evidence, not a causal model attribution or physical subsystem diagnosis. User CSV provenance is explicitly unverified.

This is offline batch inference. Use only closed minute bins for operational replay. Repeated calls must include preceding history; this command is not a stateful live mission connector. Model files are locally generated trusted joblib artifacts; do not load untrusted joblib files.

Reproduce training and comparisons:

```bash
python -m src.neural.contextual
python -m src.neural.compare_contextual
python -m pytest -q
```

Requires existing verified ESA raw metadata and `data/processed/esa_supervised/` arrays. Training creates separate artifacts and leaves V1/V2 intact. Protocol, candidate thresholds, histories, per-event records, yearly metrics and model hashes are saved in `reports/esa_contextual/`; the selected policy and models are in `models/esa_contextual/`. No new large dataset download is required.

## Verification and limits

- 52 tests pass, including causal history, level invariance, chunk equivalence, missing data, calibration constraints, augmentation determinism, serialization and CSV timestamp handling.
- Saved-model inference reproduced 2,880 full-context predictions from a real telemetry excerpt within 1e-5.
- The CSV command was exercised end to end. Its output and a prepared-bin example are saved alongside metrics.
- Real ESA communication-gap annotations exclude affected minutes during offline evaluation; the inference command itself requires no labels and uses missingness/coverage. Live gap recognition would need its own validation.
- Several long events have poor observation coverage; every event remains in the 29-event denominator. Event detection does not mean complete monitoring of its duration.
- Recent baselines can adapt to persistent faults or very slow degradation. Low minute recall makes that limitation measurable; this model does not solve battery life or slow failure prediction.
- Six channels, one mission, one training seed and a small number of calibration events. No independent mission or genuinely untouched evaluation has been completed.
- Anonymous channel identities prevent justified mapping to batteries, wheels or other physical subsystems.

The next acceptance test should reserve new real evaluation data and set an operator-approved alert budget. These results establish a better retrospective anomaly-alert candidate, not a finished spacecraft operations product.
