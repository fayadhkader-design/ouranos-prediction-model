"""Retrospective frozen-model sensitivity audit; does not retrain or retune."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.neural.supervised import DATA, MODEL, CHANNELS, classifier, features


def predict(model, x):
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(x)).squeeze(1)).numpy()


def audit():
    out = Path('reports/esa_diagnostics')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    ck = torch.load(MODEL / 'classifier.pt', map_location='cpu', weights_only=True)
    model = classifier()
    model.load_state_dict(ck['state_dict'])
    model.eval()
    center, scale = np.array(ck['center']), np.array(ck['scale'])
    x = np.load(DATA / 'features.npy', mmap_mode='r')
    labels = np.load(DATA / 'labels.npy', mmap_mode='r')
    scores = np.load(DATA / 'scores.npy', mmap_mode='r')
    years = pd.to_datetime(np.load(DATA / 'timestamps.npy'), utc=True).year.to_numpy()
    threshold = json.loads((MODEL / 'threshold.json').read_text())['chosen']['threshold']
    finite = np.isfinite(x).all(axis=1)
    gaps = ((labels & 4) != 0).any(axis=1)
    valid = finite & np.r_[False, finite[:-1]] & ~gaps & np.r_[False, ~gaps[:-1]]
    nominal = ~((labels & 1) != 0).any(axis=1)
    clean = (labels == 0).all(axis=1)
    rng = np.random.default_rng(921)
    refids = np.flatnonzero(valid & clean & (years < 2004))
    ref = features(x, rng.choice(refids, min(30000, len(refids)), replace=False), center, scale)
    lo, hi = np.quantile(ref, [.01, .99], axis=0)
    baseline = np.median(ref, axis=0)
    names = [f'{c}_{s}' for c in CHANNELS for s in ['last', 'min', 'max', 'std']] + [f'{c}_difference' for c in CHANNELS]
    rows, drift, sensitivity = [], [], []
    for year in range(2000, 2014):
        mask = valid & nominal & (years == year)
        if year >= 2005:
            mask &= np.isfinite(scores)
        ids = np.flatnonzero(mask)
        sample = rng.choice(ids, min(20000, len(ids)), replace=False)
        f = features(x, sample, center, scale)
        pred = predict(model, f)
        if year >= 2005:
            assert np.allclose(pred, scores[sample], atol=1e-6)
        rows.append({'year': year, 'nominal_minutes': len(ids), 'sample_minutes': len(sample),
                     'alert_fraction': float((scores[ids] > threshold).mean()) if year >= 2005 else float((pred > threshold).mean()),
                     'fraction_is_sampled': year < 2005,
                     'unannotated_sample_alert_fraction': float((pred[clean[sample]] > threshold).mean()),
                     'score_median_sample': float(np.median(pred)),
                     'any_feature_outside_training_1_99_sample': float(((f < lo) | (f > hi)).any(axis=1).mean())})
        for j, name in enumerate(names):
            drift.append({'year': year, 'feature': name, 'training_median_scaled': float(baseline[j]),
                          'year_median_scaled': float(np.median(f[:, j])),
                          'outside_training_1_99_fraction': float(((f[:, j] < lo[j]) | (f[:, j] > hi[j])).mean())})
        groups = [(name, list(range(c * 4, c * 4 + 4)) + [24 + c]) for c, name in enumerate(CHANNELS)]
        groups += [('all_levels', [c * 4 + j for c in range(6) for j in range(3)]),
                   ('all_std_and_differences', [c * 4 + 3 for c in range(6)] + list(range(24, 30)))]
        for name, columns in groups:
            replaced = f.copy()
            replaced[:, columns] = baseline[columns]
            new = predict(model, replaced)
            sensitivity.append({'year': year, 'channel': name,
                                'original_sample_alert_fraction': float((pred > threshold).mean()),
                                'replacement_sample_alert_fraction': float((new > threshold).mean()),
                                'mean_score_reduction': float((pred - new).mean())})
        print('Audited', year, flush=True)
    result = {'source': 'REAL MISSION DATA', 'threshold_frozen': threshold, 'seed': 921,
              'reference_sample': len(ref), 'yearly': rows,
              'limitations': ['Retrospective diagnostic; no fresh validation',
                              'Negative means no Anomaly label; separate completely unannotated check included',
                              'Channel replacement can create unrealistic combinations; not physical causal attribution',
                              'Feature summaries use sampled clipped model inputs',
                              'No model or threshold changed']}
    (out / 'audit.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame(drift).to_csv(out / 'feature_drift.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out / 'channel_sensitivity.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
