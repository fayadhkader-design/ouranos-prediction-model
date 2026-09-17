# Ouranos: first neural network trained on real ESA telemetry

This is a separate anomaly-detection experiment using **real ESA Mission 1 telemetry**, not the synthetic spacecraft simulator. A trained PyTorch checkpoint is saved at `models/esa_neural/autoencoder.pt`. The existing Streamlit application still runs the synthetic demo; the real model has not replaced its spacecraft risk policy.

## Data and provenance

Official source: [ESA Anomalies Dataset, DOI 10.5281/zenodo.12528696](https://doi.org/10.5281/zenodo.12528696), linked by the [ESA-ADB repository](https://github.com/kplabs-pl/ESA-ADB). We selected channels 41–46, the six-channel subset declared in the repository's [Mission 1 experiments](https://github.com/kplabs-pl/ESA-ADB/blob/main/mission1_experiments.py). All belong to anonymized `subsystem_5`, with anonymized physical units. No battery/wheel semantics are assumed.

The downloader retrieved approximately 483 MiB of selected official ZIP members using HTTP byte ranges rather than the full 3.78 GB mission archive. Every member passed ZIP CRC32 verification and has a local SHA256 recorded in `reports/esa_neural/download_manifest.json`. Partial downloads do not verify the full-archive MD5. Raw data is in `data/raw/ESA-Mission1/`, excluded from Git. Do not redistribute mission data without checking publisher terms. Pickle loading is restricted to the verified official download path, never user uploads.

The six channels cover 2000-01-01 through 2013-12-31. Preprocessing produces 14,728,319 30-second grid rows, with past-only zero-order hold bounded to five minutes. There is no backward fill. Missing windows and annotated communication gaps are excluded. Labels are separate from network inputs; they are used to exclude annotated intervals from nominal fitting and to evaluate predictions. Rare events are excluded from nominal fitting but count as negatives in testing. This is not wholly unsupervised training, because annotations curate the fitting data.

## Model and chronological splits

Architecture: a compact **51,212-parameter temporal autoencoder**, with 64 trailing samples × six channels (approximately 32 minutes), flattened through `384 → 64 → 12 → 64 → 384`. Hidden layers use GELU. The undercomplete bottleneck learns a reconstruction of normal multivariate windows. This is a neural network, but not an LSTM, Transformer, physical spacecraft model, or future-failure predictor.

- Training: 2000-01-01 to before 2006-07-01. There are 6,740,748 eligible nominal rows and 841,811 eligible stride-eight windows; a seeded subset of **60,000 windows** is actually used for gradient training.
- Validation: July–September 2006; 12,000 nominal windows used for early stopping.
- Calibration: October–December 2006; threshold fixed at the 99.5th percentile of nominal window scores.
- Test: January 2007–December 2013, evaluated after model and threshold policy selection. No test-outcome tuning was performed.

Scaling uses training-only median and IQR (standard-deviation fallback for constant IQR). AdamW optimizes Smooth L1 reconstruction loss. Training ran 22 epochs; epoch 16 was selected under the declared minimum-improvement rule. Training is deliberately small and CPU-based; more compute alone does not establish better accuracy.

At inference, the score is the largest squared standardized reconstruction residual across the six channels at the terminal sample. The network sees that sample and its history, so this is **contemporaneous anomaly detection**, not a forecast. An alert requires three consecutive above-threshold bins. Invalid windows reset the rule. No full-window score is assigned to earlier timestamps.

A two-component PCA reconstruction model, fitted on the same earlier nominal period and using its own calibration threshold under the same policy, provides a simple baseline. It uses current measurements rather than temporal windows, so it is a practical baseline rather than a matched-capacity ablation.

## Held-out results

| Metric | Neural autoencoder | PCA baseline |
|---|---:|---:|
| Distinct events detected | **4 / 29** | 7 / 29 |
| Event recall | 13.79% | 24.14% |
| Point precision | 60.63% | 8.95% |
| Point recall | 85.71% | 99.52% |
| Point false-positive rate | 0.496% | 9.019% |
| Average precision (PR area summary) | 0.8301 | 0.8976 |
| ROC AUC | 0.9601 | 0.9962 |
| False nonoverlapping alert episodes | 567 | 154,017 |
| False episodes / 1,000 nominal hours | 9.33 | 2,533.00 |

7,361,476 of 7,364,160 test rows were scored (99.964% coverage), spanning 60,804.12 scored nominal hours. Anomaly prevalence is approximately 0.883% of scored test samples. False episodes count contiguous alarm stretches with no overlap with any annotated anomaly; they are not debounced operator incidents. An immediate clear rule can fragment one underlying condition into many episodes, especially for PCA.

**The neural network is not an overall winner.** It produces fewer false alarms at the declared thresholds, but PCA ranks anomalous samples better by average precision and catches more distinct events. Neither result is adequate to claim operational reliability.

**Why high point recall but low event recall?** A few long anomalies dominate anomalous sample counts. Most missed events are brief; some last only seconds. Thirty-second resampling can discard transient measurements, and a three-bin sustained-alert policy inherently disadvantages brief events. A few longer events are also missed. Annotation intervals are mapped to overlapping sample bins, but that does not recover discarded telemetry samples. No point adjustment expands a successful detection over an entire event.

The four detected events had a median detection delay of 1.22 minutes **after annotated onset**. This is conditional on detection and excludes 25 misses. It is not an early-warning lead time before failure. The data does not supply the physical failure endpoints and certified static thresholds needed to reproduce the synthetic demo's lead-time claim.

## Reproduce

From the project directory, using Python 3.12+ (tested with 3.14):

```bash
python -m pip install -r requirements-neural.txt
python -m src.ingestion.esa_download
python -m src.neural.prepare
python -m src.neural.train
python -m src.neural.evaluate
python -m pytest -q
```

`python -m src.neural.evaluate --reuse-scores` only rebuilds metrics from existing score arrays. Use it only with the exact checkpoint and prepared dataset that produced those arrays; after changing a model or data, run full evaluation. The evaluation command overwrites output reports, so preserve previous runs before comparing changes.

Files:

- `models/esa_neural/autoencoder.pt`: trained weights, scaling, architecture and configuration; load using PyTorch `weights_only=True`.
- `models/esa_neural/pca_baseline.joblib`: trusted local PCA artifact.
- `models/esa_neural/thresholds.json`: calibration thresholds.
- `reports/esa_neural/training_history.json`: actual epoch losses.
- `reports/esa_neural/training.json`: training scope and preparation metadata.
- `reports/esa_neural/metrics.json`: complete held-out metrics.
- `reports/esa_neural/neural_events.csv` and `pca_events.csv`: every test anomaly event, including misses.
- `reports/esa_neural/hourly_overview.csv`: sampled score overview for inspection; hourly sampling can hide brief events. Metrics use full-resolution arrays, not this overview.
- `data/processed/esa_neural/`: full prepared telemetry, labels, timestamps, scores and alarms, excluded from Git.

38 tests cover the original MVP and neural mechanics, including causal windows, chronological boundaries, training-only scaling, gradients, checkpoint restoration, event unions, short-event labels, sustained alerts and mixed ISO timestamp precision. Small test fixtures are fabricated for mechanical checks; the performance report comes from actual ESA telemetry.

## Scientific limitations and next experiment

This is one small model, one seed and six anonymized channels from one mission. Labels are assumed to be complete when counting false positives. Communication-gap masks use retrospective annotations, so the reported availability policy is an offline evaluation protocol, not a demonstrated online gap classifier. Scores are neither failure probabilities nor a validated Ouranos spacecraft risk score. No physical subsystem diagnosis, remaining useful life or arbitrary satellite failure prediction is supported.

The next experiment should preserve short-lived telemetry anomalies, distinguish transient alerts from persistent degradation alerts, evaluate event-level false-alarm budgets, and test additional mission channels. Select those changes on a development period and reserve a fresh final test period/mission: the 2007–2013 results are now observed and must not be repeatedly tuned against while still called untouched.
