"""Frozen-policy common-coverage comparison; never modifies model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.neural.contextual import ROOT, REPORT, CACHE
from src.neural.supervised import DATA, CHANNELS
from src.neural.prepare import load_annotation_table
from src.neural.evaluate import event_metrics, sustained_array
from src.evaluation import confusion


def compare():
    index = pd.to_datetime(np.load(DATA/'timestamps.npy'),utc=True)
    labels = np.load(DATA/'labels.npy',mmap_mode='r')
    y = ((labels&1)!=0).any(axis=1)
    selection = json.loads((ROOT/'selection.json').read_text())
    name = selection['winner']
    score = np.load(CACHE/f'{name}_scores.npy',mmap_mode='r')
    old = np.load(DATA/'scores.npy',mmap_mode='r')
    old_threshold = json.loads(Path('models/esa_supervised/threshold.json').read_text())['chosen']['threshold']
    policy = selection['candidates'][name]
    common = np.isfinite(old)&np.isfinite(score)&(index>=pd.Timestamp('2007-01-01',tz='UTC'))
    annotations = load_annotation_table(Path('data/raw/ESA-Mission1'),CHANNELS)
    report = {'scope':'Identical scored minutes; each frozen calibration policy; later years already examined',
              'common_minutes':int(common.sum()),'positive_minutes':int((common&y).sum()),'models':{}}
    for model,alarm in [('previous_supervised',old>old_threshold),('context_nn',sustained_array(np.isfinite(score)&(score>policy['threshold']),policy['dwell']))]:
        events,episodes = event_metrics(index,labels,common,alarm&common,annotations,CHANNELS,np.full(len(index),255,dtype=np.uint8),cadence_seconds=60)
        report['models'][model] = {'detected_events':sum(e['detected'] for e in events),'total_events':len(events),
            'point':confusion(y[common],alarm[common]),**episodes,
            'false_alerts_per_1000_nominal_hours':episodes['false_nonoverlapping_episodes']/((common&~y).sum()/60)*1000}
    (REPORT/'common_coverage.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': compare()
