# Supervised ESA anomaly experiment — V2

This experiment trains a neural network **directly on normal/anomaly labels**, rather than reconstruction loss. It uses the same verified real ESA Mission 1 channels 41–46. It is saved separately from the autoencoder and is **not promoted to the dashboard**.

## Result: better event coverage, unacceptable false alarms

On the previously examined 2007–2013 period, V2 detected **22/29 events (75.9%)**, versus V1's 4/29. However, V2 generated **1,559 false alarm episodes per 1,000 nominal hours** and alerted on **41.8% of scored nominal minutes**. Point precision was **0.30%**, point recall **51.7%**, and average precision **0.4468**. Higher event detection does not make this an operational improvement.

The threshold was selected on 2005–2006 calibration data, where it detected 7/8 scorable events with 1.20 false episode starts per 1,000 nominal hours. The large degradation on later years demonstrates that this calibration policy did not transfer. It is consistent with distribution shift or poor generalization; this experiment does not establish the physical cause.

**Event overlap must not be mistaken for precise event recognition.** The yearly audit shows alarms on 50.4%, 79.9%, 91.6% and 69.1% of nominal minutes in 2009–2012 respectively. Some events are counted as detected during near-continuous false alarming. On common scoreable minute bins, the autoencoder still catches 4/29 events with 1.92 false episodes per 1,000 nominal hours, versus V2's 22/29 with 1,559.35. The gain does not represent acceptable selective detection.

The comparison is **retrospective**, not a new untouched test: 2007–2013 was already examined for V1. No thresholds were adjusted in response to V2's later-year result. A new mission/channel holdout is still required before claiming generalization.

## What changed

1. **Preserve brief excursions.** Each right-closed one-minute bin contains last, minimum, maximum and standard deviation for each channel. Every observed spike contributes to the corresponding extremum, even when it is not the last measurement. Empty bins stay missing; there is no interpolation or forward filling. Alerts are only available after the bin closes. Channel extrema may have occurred at different times and must not be interpreted as simultaneous spacecraft states.
2. **Use labeled examples.** Binary targets are “anomaly” and “not annotated as anomaly.” Rare nominal events remain negatives. Communication gaps and incomplete current/previous bins are excluded.
3. **Prevent long-event domination in training.** Positive sampling is capped at 200 bins per annotated event, while negatives are sampled independently. This retains short-event examples but is not exact equal weighting of events.
4. **Optimize classification.** A 30-input, 48-hidden, 24-hidden, one-output neural classifier with GELU uses weighted binary cross-entropy and AdamW. Inputs are 24 bin statistics and six previous-minute last-value differences. There are 2,689 trainable parameters. Normalization is fitted only on selected training rows. Normalized features are clipped to ±30. The sigmoid output is a classification score, **not a calibrated anomaly or failure probability**, particularly because sampling and class weighting change the effective class prior.
5. **Allow transient alerts.** One above-threshold bin can alert. The old three-sample persistence requirement is not applied to this transient classifier. The original persistent model remains available separately; the two have not been combined into a validated ensemble.

## Actual training and selection

- Training: 2000–2003; **81,555 examples**, including **1,555 positive examples**.
- Validation: 2004; 30,201 sampled examples including 201 positives. The sampled validation AP is for checkpoint selection only, not full-stream performance. Only a small number of events contribute, so selection can overfit their characteristics.
- Training ran 15 epochs and retained epoch 7 by validation average precision. Seed: 2027. CPU training; batch size 512; AdamW learning rate 0.001.
- Calibration: 2005–2006. Predefined threshold candidates were compared by event detections under a maximum of 10 false episode starts per 1,000 nominal hours. Ties favor fewer false starts and then a higher threshold. Chosen threshold: **0.99**.
- Retrospective evaluation: 2007–2013. Reports include every event and miss, not point-adjusted classifications.

V2 scored 3,636,260 of 3,682,081 minute bins (approximately 98.76%). Its missing-data policy differs from V1's bounded forward fill, so raw point metrics have different denominators. `common_coverage_comparison.json` additionally compares both alarms on common scoreable minute bins, preserving either V1 half-minute alert in each minute. This is a diagnostic, not a retrained matched-architecture ablation. Different model, input, split, loss, and alert-policy changes are combined; the event-recall increase cannot be attributed solely to supervised learning.

False episodes are contiguous above-threshold stretches, with no cooldown/hysteresis, that never overlap an annotated anomaly. Calibration's budget counts episodes whose starts fall outside anomalies, a slightly more conservative definition. Neither quantity establishes how many independent physical problems occurred.

## Files and reproducibility

```bash
python -m pip install -r requirements-neural.txt
python -m src.ingestion.esa_download
python -m src.neural.supervised --prepare
python -m src.neural.compare_supervised
python -m pytest -q
```

The first preparation requires approximately 0.8 GB beyond the already-downloaded raw data. Omit `--prepare` to retrain from existing prepared arrays. Commands overwrite this experiment's outputs; archive prior results before rerunning.

- `src/neural/supervised.py`: causal aggregation, supervised training, calibration and evaluation.
- `models/esa_supervised/classifier.pt`: weights, training-only scaling and model configuration. Load with `torch.load(..., weights_only=True)`.
- `models/esa_supervised/threshold.json`: chosen threshold and all calibration trials.
- `reports/esa_supervised/training.json`: sample counts and actual epoch history.
- `reports/esa_supervised/metrics.json`: retrospective metrics and limitations.
- `reports/esa_supervised/events.csv`: event-by-event detections and misses.
- `reports/esa_supervised/common_coverage_comparison.json`: common-coverage and yearly diagnostics.
- `data/processed/esa_supervised/`: full input features, labels and scores; excluded from Git.

The original model and original reports are retained. No raw mission data is included in Git by these changes. Neither network is deployed as a real spacecraft risk service.

## Next decision

Do not deploy V2 merely because it catches 22 events. It alerts much too often. The next controlled study should evaluate mode-aware or relationship-based features, broader historical calibration, calibrated uncertainty, and a combined transient/persistent policy against a realistic false-alert budget. Choose changes on development data and reserve independent mission/channel data for final evaluation. Increasing persistence would likely sacrifice the recovered brief events; it is not a free fix.

## Verification

All 43 project tests pass. Added checks confirm extrema preserve brief spikes, future readings cannot alter completed bins, empty bins remain missing, previous-minute features are causal, event sampling obeys split/cap rules, and supervised classification produces finite gradients. Reloading the saved checkpoint reproduced 469 sampled full-stream scores to an absolute tolerance of 1e-6. Test fixtures validate mechanics; the reported performance uses real ESA measurements and annotations.
