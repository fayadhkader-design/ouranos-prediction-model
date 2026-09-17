# Real-mission dashboard, alert recovery, and separate-mission evaluation

The application now opens a **Real ESA telemetry** workspace connected to the saved Mission 1 neural model. The synthetic simulation remains a separate sidebar choice. The real workspace supports recorded event replay, start/pause/reset, channel inspection, live execution of the model on uploaded CSVs, downloadable predictions, and transparent measured performance. It never turns neural logits into a fabricated 0–100 failure probability or assigns anonymous channels to physical equipment.

## Alert recovery: a modest improvement with a trade-off

The neural network weights and alert-entry threshold are unchanged. A causal alert lifecycle keeps a warning open while weaker evidence persists, instead of requiring every subsequent minute to cross the high entry threshold.

Entry: score > 7.8189931262. Recovery: score <= 1.8869786263 for one minute. An alert also ends at missing data or 15 minutes after the most recent high-score trigger. New high-score samples refresh that timeout. This policy was selected using 2005–2006 calibration data only, maximizing event recall then minute recall under <=10 false episode starts per 1,000 nominal hours and <=0.1% nominal time in alarm.

| Mission 1 retrospective result | Instantaneous V3 alerts | Alerts with recovery |
|---|---:|---:|
| Events detected | 19/29 | 19/29 |
| Anomalous minutes flagged | 2.68% | 5.43% |
| Minute precision | 33.01% | 24.92% |
| False episodes / 1,000 nominal hours | 2.65 | 1.57 |
| Nominal minutes flagged | 0.01142% | 0.03432% |
| False flagged minutes | 414 | 1,244 |

**Fewer episodes does not mean less false-alarm time.** Warnings are less fragmented but remain active longer; anomalous-minute coverage doubles while false-alarm minutes triple. Most anomalous minutes remain missed. This is an alert-management improvement, not a solution to sustained-fault detection or failure prediction. Both instantaneous and active-alert columns remain available in CSV output.

## Previously unused real data: ESA Mission 2

The second mission was downloaded from the same official ESA release, DOI [10.5281/zenodo.12528696](https://doi.org/10.5281/zenodo.12528696). Six new channels (70, 71, 72, 78, 79, 80) were selected using channel metadata only: a group of six target channels with matching anonymous subsystem/physical-unit labels. That is **not proof of physical equivalence**. Ascending channel number mapped to the six existing network slots without label-driven permutation.

Before reading their telemetry or calculating their evaluation metrics, the protocol froze the model checksum, policy checksum, channel mapping and two variants:

1. Completely frozen Mission 1 model, scales and thresholds.
2. The same weights and thresholds, with only six global scales estimated from the first 365 days of Mission 2 observations without labels.

Evaluation used the later period, 2000-12-31 00:01 UTC to 2003-06-30 23:59 UTC. It scored 1,313,188 of 1,313,279 minutes, including 21,859.95 nominal hours. Only **two annotated events** occur in the selected evaluated channels. That is too few for a strong event-recall estimate. Annotations were evaluated only after both inference passes; no outcome-driven tuning followed.

| Mission 2 variant | Events detected | False episodes / 1,000 nominal h | Nominal time in alarm |
|---|---:|---:|---:|
| Frozen, instantaneous | 1/2 | 2,209.66 | 5.39% |
| Frozen, recovery | 1/2 | 2,035.87 | 11.83% |
| Scale adaptation, instantaneous | 0/2 | 932.16 | 20.47% |
| Scale adaptation, recovery | 0/2 | 736.64 | 34.50% |

**The transfer test failed to establish useful cross-mission performance.** Recovery reduces episode fragmentation but worsens time in alarm on this mission too. The simple scale adaptation did not fix transfer. The primary frozen-recovery minute precision is approximately 0.000645%, and minute recall 0.0629%. The network cannot be called mission-independent or production-ready.

This is a separate-source transfer stress test of a specific positional mapping, not proof that the Mission 1 model would fail on every new mission, nor proof of physical sensor equivalence. The data and results are now examined and must not be reused as an untouched final holdout in future work. The original benchmark's official composite metrics were not implemented here; these are our documented event-overlap and ordinary minute metrics. [Official benchmark](https://github.com/kplabs-pl/ESA-ADB).

## Run and inspect

```bash
python -m pip install -r requirements-neural.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Use **Real ESA telemetry** in the sidebar. The model and prepared Mission 1 data must be present; missing artifacts produce an explicit message. Uploaded CSVs are limited to 50 MB and scored locally. They must contain the actual Mission 1 channel schema; uploaded provenance is unverified. About a day of prior observations is required. Warmup and missingness produce `UNSCORED`, not a nominal assessment.

Command-line inference with the same recovery policy:

```bash
python -m src.neural.predict_contextual --input telemetry.csv --output predictions.csv --recovery
```

Omit `--recovery` for the original instantaneous V3 decisions. Replay recomputes predictions from real prepared telemetry using the saved network. It does not feed event labels into inference. The event dropdown includes missed events; the dotted annotation-onset marker is a retrospective review aid.

Reproduce the additional experiments (the transfer protocol and frozen model checksums must remain consistent):

```bash
python -m src.neural.lifecycle
python -m src.neural.transfer
python -m pytest -q
```

`src.neural.transfer` requires the six verified Mission 2 archive members. The generalized official downloader supports `download(Path('data/raw'), channels=[...], mission='ESA-Mission2')`. Raw telemetry remains excluded from Git. Selected raw members total about 222 MB; the full multi-GB archive is not downloaded. ZIP-member CRC32 and local SHA256 checks are used; no claim is made to verify the complete archive checksum from partial downloads.

## Verification

- 56 passing tests, including causal recovery, timeout refresh, gap resets, no invented triggers, real replay reset, event switching, and returning to the synthetic simulator.
- Real-model CSV inference exercised through the browser: 3,024 scored bins of 4,320, with 1,296 warmup bins retained as unscored.
- The application launches locally at port 8501. Existing synthetic injection, explanation and reset checks continue to pass.
- CSV results retain both instantaneous and active-alert decisions; upload result identity is tied to content, not just filename.

Machine-readable evidence: `reports/esa_contextual/lifecycle_metrics.json`, `models/esa_contextual/lifecycle.json`, `reports/esa_transfer/protocol.json`, and `reports/esa_transfer/metrics.json`. Historical experiment reports remain separate.

The integration and evaluations are implemented. Reliable sustained-fault monitoring and generalization remain research problems. A mission-specific training/adaptation study with a genuinely reserved subsequent evaluation is needed before an operator pilot; no further fitting to Mission 2's reported results was performed here.
