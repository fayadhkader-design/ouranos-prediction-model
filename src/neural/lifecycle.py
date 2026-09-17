"""Causal alert recovery policy; keeps a warning open while evidence persists.
Calibrates on 2005–2006 only; frozen V3 network weights/entry threshold unchanged.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.neural.contextual import ROOT, REPORT, CACHE, event_intervals
from src.neural.supervised import DATA, CHANNELS
from src.neural.prepare import load_annotation_table
from src.neural.evaluate import event_metrics, sustained_array
from src.evaluation import confusion, event_starts


def lifecycle_alarm(score, entry, exit_threshold, quiet_minutes, max_minutes):
    """Enter on high score; close after quiet recovery, missingness or timeout.
    Each new high sample refreshes timeout. Causal union of evidence-triggered spans.
    """
    score = np.asarray(score)
    valid = np.isfinite(score)
    trigger = np.flatnonzero(valid & (score > entry))
    quiet_ends = np.flatnonzero(sustained_array(valid & (score <= exit_threshold),quiet_minutes))
    invalid = np.flatnonzero(~valid)
    quiet_ends = np.r_[quiet_ends,len(score)]
    invalid = np.r_[invalid,len(score)]
    difference = np.zeros(len(score)+1,dtype=np.int32)
    ends = np.minimum(trigger+max_minutes,quiet_ends[np.searchsorted(quiet_ends,trigger)])
    ends = np.minimum(ends,invalid[np.searchsorted(invalid,trigger)])
    np.add.at(difference,trigger,1)
    np.add.at(difference,ends,-1)
    return (np.cumsum(difference[:-1])>0)&valid


def apply_recovery(frame, policy):
    """Keep raw instantaneous decisions alongside the causal active-alert state."""
    result=frame.copy()
    result['instant_alert']=result.status.eq('ANOMALY')
    active=lifecycle_alarm(result.model_score.to_numpy(),policy['entry_threshold'],policy['exit_threshold'],policy['quiet_minutes'],policy['max_minutes'])
    result['active_alert']=active
    result['status']=np.where(result.model_score.isna(),'UNSCORED',np.where(active,'ALERT ACTIVE','NO ALERT'))
    return result


def run():
    selection=json.loads((ROOT/'selection.json').read_text())
    name=selection['winner'];entry=selection['candidates'][name]['threshold']
    score=np.load(CACHE/f'{name}_scores.npy',mmap_mode='r')
    index=pd.to_datetime(np.load(DATA/'timestamps.npy'),utc=True)
    labels=np.load(DATA/'labels.npy',mmap_mode='r');y=((labels&1)!=0).any(axis=1)
    annotations=load_annotation_table(Path('data/raw/ESA-Mission1'),CHANNELS)
    cal=(index>=pd.Timestamp('2005-01-01',tz='UTC'))&(index<pd.Timestamp('2007-01-01',tz='UTC'))&np.isfinite(score)
    groups=event_intervals(index,annotations,pd.Timestamp('2005-01-01',tz='UTC'),pd.Timestamp('2007-01-01',tz='UTC'))
    cal_score=np.where(cal,score,np.nan)
    nominal=cal&~y;hours=nominal.sum()/60
    trials=[]
    for quantile in [.90,.99]:
        exit_threshold=float(np.quantile(score[nominal],quantile))
        for quiet in [1,5,15]:
            for maximum in [15,60,360]:
                alarm=lifecycle_alarm(cal_score,entry,exit_threshold,quiet,maximum)
                trials.append({'exit_threshold':exit_threshold,'exit_nominal_quantile':quantile,'quiet_minutes':quiet,'max_minutes':maximum,
                    'false_starts_per_1000h':float((event_starts(alarm)&nominal).sum()/hours*1000),
                    'nominal_alarm_fraction':float(alarm[nominal].mean()),'point_recall':float(alarm[cal&y].mean()),
                    'detected_events':int(sum(any(alarm[a:b].any() for a,b in spans) for _,spans in groups))})
    feasible=[r for r in trials if r['false_starts_per_1000h']<=10 and r['nominal_alarm_fraction']<=.001]
    chosen=max(feasible,key=lambda r:(r['detected_events'],r['point_recall'],-r['nominal_alarm_fraction']))
    policy={'entry_threshold':entry,**chosen,'selection':'Calibration only: max event recall then minute recall under <=10 false starts/1000 nominal hours and <=0.1% nominal alarm time',
            'model':name,'source':'REAL MISSION DATA','trials':trials}
    (ROOT/'lifecycle.json').write_text(json.dumps(policy,indent=2))
    test=(index>=pd.Timestamp('2007-01-01',tz='UTC'))&np.isfinite(score)
    alarm=lifecycle_alarm(np.where(test,score,np.nan),entry,chosen['exit_threshold'],chosen['quiet_minutes'],chosen['max_minutes'])
    events,episodes=event_metrics(index,labels,test,alarm,annotations,CHANNELS,np.full(len(index),255,dtype=np.uint8),cadence_seconds=60)
    hours=(test&~y).sum()/60
    result={'policy':{k:v for k,v in policy.items() if k!='trials'},'point':confusion(y[test],alarm[test]),
        'detected_events':sum(e['detected'] for e in events),'total_events':len(events),
        'false_alerts_per_1000_nominal_hours':episodes['false_nonoverlapping_episodes']/hours*1000,**episodes,
        'status':'Retrospective; not independent validation. Alert-state persistence does not change neural scores.'}
    np.save(CACHE/'lifecycle_alarms.npy',alarm)
    pd.DataFrame(events).to_csv(REPORT/'lifecycle_events.csv',index=False)
    (REPORT/'lifecycle_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
