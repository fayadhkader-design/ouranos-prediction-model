# Real ESA classifier: false-alarm investigation

This is a retrospective investigation of the frozen supervised classifier on six anonymous ESA Mission 1 channels, not a newly trained model or independent validation. The model and its 0.99 threshold were unchanged. The dashboard still uses the synthetic demo.

## Main finding

The excessive alerts are strongly associated with changes in absolute signal levels across mission years. The network is sensitive to those levels, even when no anomaly is annotated. This supports investigating temporal distribution shift; it does not establish the physical cause of that shift or prove every unannotated interval healthy.

| Year | Minutes without an Anomaly label flagged |
|---|---:|
| 2005 | 0.0017% |
| 2006 | 0.0025% |
| 2007 | 0.0550% |
| 2008 | 0.3528% |
| 2009 | 50.4102% |
| 2010 | 79.8882% |
| 2011 | 91.5750% |
| 2012 | 69.1142% |
| 2013 | 1.3646% |

These percentages use every saved, scorable nominal minute in each year. Nominal here means no Anomaly label across the six channels. Excluding rare-event annotations as well gives 91.86% in the 2011 sample, so rare events do not explain the large alert fraction.

## Which inputs matter?

For a fixed random sample of 20,000 nominal minutes in 2011, the original alert fraction was 91.775%. We replaced selected standardized inputs with their median from a separate 30,000-row sample of unannotated 2000–2003 data and rescored the frozen model:

| Replaced inputs | Alert fraction after replacement |
|---|---:|
| Channel 41 descriptors and difference | 40.62% |
| Channel 45 descriptors and difference | 43.38% |
| Channel 46 descriptors and difference | 55.94% |
| Channel 43 descriptors and difference | 59.31% |
| All six channels' last/min/max levels | 0.00% |
| All six channels' standard deviations and minute differences | 93.20% |

This is a diagnostic intervention on model inputs, not a proposed fix. It can produce unrealistic combinations and remove actual anomaly evidence. It neither proves causal responsibility nor measures detection performance of a corrected model. Nevertheless, the large difference between replacing levels and replacing short-term variability is strong evidence that the current model's unwanted alerts depend on absolute levels.

Channel 43 also showed marked distribution change: 35.6% of sampled 2011 last values fell outside the training reference's 1st–99th percentile interval. Channels 41 and 45 showed lower median levels and substantial tail exceedances. These are anonymous channels; we cannot identify a wheel, battery, operating mode, or engineering unit from these results.

## What to do next

Develop and compare a model using past-only recent-baseline deviations alongside absolute levels and variability. Preserve a slower reference to avoid normalizing away sustained degradation. Assess chronological development folds spanning changing conditions, reporting both event detection and time spent in alarm under a fixed false-alert budget. A persistence rule alone is unlikely to solve a model flagging most of an entire year. A larger network alone has no demonstrated justification.

The 2007–2013 period has already been inspected and must remain development evidence. Any future claim of generalization requires a separately reserved evaluation. No improvement in detection, false-alert rate, or failure prediction is claimed by this audit.

## Reproduce and inspect

From the project root, using the installed neural dependencies and existing prepared data/checkpoint:

```bash
python -m src.neural.diagnose_false_alarms
python -m pytest -q
```

Outputs in `reports/esa_diagnostics/`:

- `audit.json`: population counts, annual rates, seeded sample checks and caveats.
- `feature_drift.csv`: all 30 model inputs by year; standardized, clipped model-input distributions.
- `channel_sensitivity.csv`: channel and feature-group replacement experiments.

Random seed 921; up to 20,000 sampled nominal minutes per year. Pre-2005 alert fractions are sampled retrospective scores, not complete annual estimates. All 2005–2013 sampled rescored outputs matched the saved predictions within absolute tolerance 1e-6. The existing suite passes: 43 tests. No retraining, threshold selection, raw-data download, or deployment was performed.
