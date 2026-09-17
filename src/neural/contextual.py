"""Causal recent-context experiment on real ESA telemetry.
No labels enter features. Later years are retrospective evaluation, not a fresh holdout.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import hashlib
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from src.neural.supervised import DATA, CHANNELS, select_examples
from src.neural.prepare import load_annotation_table
from src.neural.evaluate import event_metrics, sustained_array
from src.evaluation import confusion, event_starts

ROOT = Path('models/esa_contextual')
REPORT = Path('reports/esa_contextual')
CACHE = Path('data/processed/esa_contextual')

@dataclass(frozen=True)
class ContextConfig:
    fast: int = 60
    slow: int = 1440
    coverage: float = .9
    scale_floor: float = .05
    seed: int = 2029
    false_starts_budget: float = 10.0
    nominal_alarm_budget: float = .001
    train_end: str = '2004-01-01'
    validation_end: str = '2005-01-01'
    calibration_end: str = '2007-01-01'


def prior_stats(values, window, coverage=.9):
    """At t, use [t-window,t); never includes t or future observations."""
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0).astype(np.float64)
    count = np.r_[0, np.cumsum(valid)]
    total = np.r_[0., np.cumsum(safe)]
    squares = np.r_[0., np.cumsum(safe * safe)]
    right = np.arange(len(values))
    left = np.maximum(0, right - window)
    n = count[right] - count[left]
    mean = (total[right] - total[left]) / np.maximum(n, 1)
    variance = np.maximum(0, (squares[right] - squares[left]) / np.maximum(n, 1) - mean * mean)
    enough = n >= int(np.ceil(window * coverage))
    mean[~enough] = np.nan
    variance[~enough] = np.nan
    return mean, np.sqrt(variance)


def context_block(raw, global_scale, config=ContextConfig()):
    """Return 10 features/channel. Input must contain preceding slow-window history."""
    out = []
    for c in range(6):
        last, minimum, maximum, std = np.asarray(raw[:, c*4:c*4+4], dtype=np.float64).T
        fast, _ = prior_stats(last, config.fast, config.coverage)
        slow, slowstd = prior_stats(last, config.slow, config.coverage)
        denom = np.maximum(slowstd, global_scale[c] * config.scale_floor)
        delta = last - np.r_[np.nan, last[:-1]]
        out.extend([(last-fast)/denom, (minimum-fast)/denom, (maximum-fast)/denom,
                    std/denom, (maximum-minimum)/denom, delta/denom,
                    (fast-slow)/denom, (last-slow)/global_scale[c],
                    std/global_scale[c], delta/global_scale[c]])
    return np.clip(np.column_stack(out), -100, 100).astype(np.float32)


def extract(raw, ids, global_scale, config=ContextConfig(), chunk=65536):
    ids = np.asarray(ids)
    output = np.empty((len(ids), 60), dtype=np.float32)
    for block in np.unique(ids // chunk):
        positions = np.flatnonzero(ids // chunk == block)
        start = int(block * chunk)
        end = min(len(raw), start + chunk)
        left = max(0, start - config.slow)
        f = context_block(raw[left:end], global_scale, config)
        output[positions] = f[ids[positions] - left]
    return output


def network():
    return nn.Sequential(nn.Linear(60, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))


def augment_features(x, rng):
    """Small channel-coherent feature gain jitter, NOT physical synthetic telemetry."""
    gains = rng.uniform(.85, 1.15, (len(x), 6, 1)).astype(np.float32)
    return (x.reshape(-1, 6, 10) * gains).reshape(-1, 60)


def fit_neural(xt, yt, xv, yv, augmented, config):
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    center = np.median(xt, axis=0)
    scale = np.maximum(np.quantile(xt, .95, axis=0)-np.quantile(xt, .05, axis=0), .1).astype(np.float32)
    model = network()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((~yt).sum()/yt.sum())))
    val = torch.from_numpy(np.clip((xv-center)/scale, -30, 30))
    history, best, stale = [], -1., 0
    for epoch in range(1, 41):
        model.train()
        losses = []
        order = rng.permutation(len(xt))
        for start in range(0, len(order), 1024):
            ids = order[start:start+1024]
            batch = augment_features(xt[ids], rng) if augmented else xt[ids]
            batch = torch.from_numpy(np.clip((batch-center)/scale, -30, 30))
            optimizer.zero_grad()
            loss = lossfn(model(batch).squeeze(1), torch.from_numpy(yt[ids].astype(np.float32)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():
            pred = model(val).squeeze(1).numpy()
        ap = float(average_precision_score(yv, pred))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'sample_validation_ap': ap})
        if ap > best + 1e-4:
            best, stale, state, best_epoch = ap, 0, copy.deepcopy(model.state_dict()), epoch
        else:
            stale += 1
        print('augmented' if augmented else 'context_nn', history[-1], flush=True)
        if stale >= 6:
            break
    bundle = {'kind': 'neural', 'state_dict': state, 'center': center, 'scale': scale,
              'best_epoch': best_epoch, 'augmented': augmented}
    return bundle, history


def predictor(bundle):
    if bundle['kind'] == 'tree':
        return lambda f: bundle['model'].predict_proba(f)[:, 1].astype(np.float32)
    model = network()
    model.load_state_dict(bundle['state_dict'])
    model.eval()
    def call(f):
        z = np.clip((f-bundle['center'])/bundle['scale'], -30, 30).astype(np.float32)
        with torch.inference_mode():
            return model(torch.from_numpy(z)).squeeze(1).numpy()
    return call


def event_intervals(index, annotations, start, end):
    groups = []
    for event_id, group in annotations[annotations.Category.eq('Anomaly')].groupby('ID'):
        spans = []
        for r in group.itertuples():
            if r.EndTime < start or r.StartTime >= end:
                continue
            lo = max(index.searchsorted(start), index.searchsorted(r.StartTime))
            hi = min(index.searchsorted(end), index.searchsorted(r.EndTime) + 1)
            if hi > lo:
                spans.append((lo, hi))
        if spans:
            groups.append((event_id, spans))
    return groups


def calibration_trial(scores, eligible, y, groups, threshold, dwell):
    alarm = sustained_array(eligible & (scores > threshold), dwell)
    nominal = eligible & ~y
    false_starts = int((event_starts(alarm) & nominal).sum())
    hours = nominal.sum()/60
    detected = sum(any(alarm[a:b].any() for a,b in spans) for _,spans in groups)
    return {'threshold': float(threshold), 'dwell': dwell, 'detected_events': int(detected),
            'total_events': len(groups), 'false_starts_per_1000h': false_starts/hours*1000,
            'nominal_alarm_fraction': float(alarm[nominal].mean())}


def choose_threshold(scores, eligible, y, groups, config):
    neg = scores[eligible & ~y]
    thresholds = np.unique(np.r_[np.quantile(neg, [.99,.995,.999,.9995,.9998,.9999,.99995,.99998,.99999,1]),
                                  np.quantile(scores[eligible], [.999,.9995,.9999]), np.finfo(np.float32).max])
    trials = [calibration_trial(scores,eligible,y,groups,t,d) for t in thresholds for d in [1,3]]
    feasible = [r for r in trials if r['false_starts_per_1000h'] <= config.false_starts_budget and r['nominal_alarm_fraction'] <= config.nominal_alarm_budget]
    chosen = max(feasible, key=lambda r:(r['detected_events'], -r['false_starts_per_1000h'], -r['nominal_alarm_fraction'], -r['dwell']))
    return chosen, trials


def run():
    started = time.time()
    config = ContextConfig()
    for p in (ROOT, REPORT, CACHE): p.mkdir(parents=True, exist_ok=True)
    (REPORT/'protocol.json').write_text(json.dumps({'config':asdict(config), 'candidates':['context_nn','context_nn_augmented','context_tree'],
        'selection':'Maximum calibration event detection subject to <=10 false starts/1000 nominal h and <=0.1% nominal time alarm; ties fewer false starts then less nominal alarm time',
        'evaluation':'2007–2013 retrospective; already viewed in previous experiments; no fresh validation claim',
        'augmentation':'Feature-space channel-coherent gains .85–1.15, training only; not physics-validated synthetic telemetry'},indent=2))
    torch.set_num_threads(4)
    raw = np.load(DATA/'features.npy',mmap_mode='r')
    labels = np.load(DATA/'labels.npy',mmap_mode='r')
    index = pd.to_datetime(np.load(DATA/'timestamps.npy'),utc=True)
    annotations = load_annotation_table(Path('data/raw/ESA-Mission1'), CHANNELS)
    y = ((labels&1)!=0).any(axis=1)
    finite = np.isfinite(raw).all(axis=1)
    valid = finite & np.r_[False,finite[:-1]] & ~((labels&4)!=0).any(axis=1)
    valid &= np.r_[False,~((labels[:-1]&4)!=0).any(axis=1)]
    train_end, val_end, cal_end = [pd.Timestamp(d,tz='UTC') for d in [config.train_end,config.validation_end,config.calibration_end]]
    train = valid & (index < train_end)
    val = valid & (index >= train_end) & (index < val_end)
    cal = valid & (index >= val_end) & (index < cal_end)
    test = valid & (index >= cal_end)
    rng = np.random.default_rng(config.seed)
    norm_ids = np.flatnonzero(train & (labels==0).all(axis=1))
    norm_ids = rng.choice(norm_ids, min(100000,len(norm_ids)), replace=False)
    global_scale = np.maximum(np.std(raw[norm_ids,::4],axis=0),1e-6)
    ti, _ = select_examples(train,y,index,annotations,config.seed,negative_count=120000,per_event=500)
    vi, _ = select_examples(val,y,index,annotations,config.seed+1,negative_count=40000,per_event=500)
    xt, xv = extract(raw,ti,global_scale,config), extract(raw,vi,global_scale,config)
    goodt, goodv = np.isfinite(xt).all(axis=1), np.isfinite(xv).all(axis=1)
    xt,yt,xv,yv = xt[goodt],y[ti[goodt]],xv[goodv],y[vi[goodv]]
    print('Training',xt.shape,'positives',yt.sum(),'validation',xv.shape,'positives',yv.sum(),flush=True)
    training = {'train_rows':len(xt),'train_positive_rows':int(yt.sum()),'validation_rows':len(xv),'validation_positive_rows':int(yv.sum()),'histories':{}}
    bundles = {}
    for name,aug in [('context_nn',False),('context_nn_augmented',True)]:
        bundle,history = fit_neural(xt,yt,xv,yv,aug,config)
        bundles[name] = bundle
        training['histories'][name] = history
    tree = HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=10,learning_rate=.07,class_weight='balanced',early_stopping=False,random_state=config.seed)
    tree.fit(xt,yt)
    bundles['context_tree'] = {'kind':'tree','model':tree}
    (REPORT/'training.json').write_text(json.dumps(training,indent=2))
    for name,bundle in bundles.items():
        bundle.update(global_scale=global_scale,config=asdict(config),channels=CHANNELS)
        joblib.dump(bundle,ROOT/f'{name}.joblib',compress=3)
    predictions = {name:predictor(bundle) for name,bundle in bundles.items()}
    scores = {name:np.full(len(index),np.nan,dtype=np.float32) for name in bundles}
    # Calibration scores only, so candidate choice is frozen before later-year scoring.
    for start in range(index.searchsorted(val_end),index.searchsorted(cal_end),65536):
        end = min(index.searchsorted(cal_end),start+65536)
        left = max(0,start-config.slow)
        f = context_block(raw[left:end],global_scale,config)[start-left:]
        good = np.isfinite(f).all(axis=1) & cal[start:end]
        for name,pred in predictions.items():
            scores[name][np.arange(start,end)[good]] = pred(f[good])
    groups = event_intervals(index,annotations,val_end,cal_end)
    choices = {}
    for name in bundles:
        eligible = cal & np.isfinite(scores[name])
        chosen,trials = choose_threshold(scores[name],eligible,y,groups,config)
        choices[name] = chosen
        (REPORT/f'{name}_calibration.json').write_text(json.dumps({'chosen':chosen,'trials':trials},indent=2))
        print('CALIBRATION',name,chosen,flush=True)
    winner = max(choices,key=lambda n:(choices[n]['detected_events'],-choices[n]['false_starts_per_1000h'],-choices[n]['nominal_alarm_fraction']))
    selection = {'winner':winner,'candidates':choices,'config':asdict(config),'score_is_failure_probability':False,'source':'REAL MISSION DATA'}
    (ROOT/'selection.json').write_text(json.dumps(selection,indent=2))
    print('FROZEN WINNER',winner,flush=True)
    # Score every candidate for a transparent ablation, without revising selection.
    for start in range(index.searchsorted(cal_end),len(index),65536):
        end = min(len(index),start+65536)
        left = max(0,start-config.slow)
        f = context_block(raw[left:end],global_scale,config)[start-left:]
        good = np.isfinite(f).all(axis=1) & test[start:end]
        for name,pred in predictions.items():
            scores[name][np.arange(start,end)[good]] = pred(f[good])
        if (start-index.searchsorted(cal_end))//65536 % 10 == 0:
            print('Scored through',index[end-1],flush=True)
    report = {'source':'REAL MISSION DATA','winner_selected_before_later_year_scoring':winner,
              'evaluation_status':'Retrospective; these years informed prior feature-design decisions',
              'models':{},'runtime_seconds':None}
    for name,score in scores.items():
        np.save(CACHE/f'{name}_scores.npy',score)
        eligible = test & np.isfinite(score)
        alarm = sustained_array(eligible & (score>choices[name]['threshold']),choices[name]['dwell'])
        events,episodes = event_metrics(index,labels,eligible,alarm,annotations,CHANNELS,np.full(len(index),255,dtype=np.uint8),cadence_seconds=60)
        hours = (eligible & ~y).sum()/60
        yearly = []
        for year in range(2007,2014):
            mask = eligible & (index.year==year)
            yearly.append({'year':year,'scored_minutes':int(mask.sum()),'nominal_alarm_fraction':float(alarm[mask & ~y].mean())})
        metrics = {'point':confusion(y[eligible],alarm[eligible]),'average_precision':float(average_precision_score(y[eligible],score[eligible])),
                   'detected_events':sum(e['detected'] for e in events),'total_events':len(events),'scorable_events':sum(e['scorable'] for e in events),
                   'scored_minutes':int(eligible.sum()),'total_minutes':int((index>=cal_end).sum()),'nominal_hours':float(hours),
                   'false_alerts_per_1000_nominal_hours':episodes['false_nonoverlapping_episodes']/hours*1000,
                   'median_delay_minutes_detected_only':float(np.median([e['detection_delay_minutes'] for e in events if e['detected']])) if any(e['detected'] for e in events) else None,
                   **episodes,'yearly':yearly}
        report['models'][name]=metrics
        pd.DataFrame(events).to_csv(REPORT/f'{name}_events.csv',index=False)
        print('RESULT',name,json.dumps(metrics),flush=True)
    report['runtime_seconds']=time.time()-started
    (REPORT/'metrics.json').write_text(json.dumps(report,indent=2))
    manifest = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*ROOT.glob('*'),REPORT/'protocol.json',REPORT/'metrics.json'] if p.is_file()}
    (REPORT/'sha256.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run()
