"""Predeclared, frozen-policy evaluation on previously unused ESA Mission 2.
This evaluates cross-mission transfer, not a claim of physical channel equivalence.
"""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import torch
from src.neural.contextual import ROOT, ContextConfig, context_block, predictor
from src.neural.lifecycle import lifecycle_alarm
from src.neural.supervised import bin_statistics
from src.neural.prepare import annotation_masks, load_annotation_table
from src.neural.evaluate import event_metrics
from src.evaluation import confusion

REPORT=Path('reports/esa_transfer')
DATA=Path('data/processed/esa_transfer')


def run():
    protocol=json.loads((REPORT/'protocol.json').read_text());channels=protocol['channels']
    path=ROOT/'context_nn.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['model_sha256']:
        raise ValueError('Frozen model changed after protocol declaration')
    if hashlib.sha256((ROOT/'lifecycle.json').read_bytes()).hexdigest()!=protocol['lifecycle_sha256']:
        raise ValueError('Frozen policy changed after protocol declaration')
    folder=Path('data/raw/ESA-Mission2')
    manifest=json.loads((folder/'download_manifest.json').read_text())
    hashes={Path(m['path']).name:m['sha256'] for m in manifest['members']}
    DATA.mkdir(parents=True,exist_ok=True)
    raw=None;index=None;stats=[]
    for j,c in enumerate(channels):
        file=folder/'channels'/f'{c}.zip'
        with file.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=hashes[file.name]:raise ValueError('Input integrity mismatch')
        frame=pd.read_pickle(file);frame.index=pd.to_datetime(frame.index,utc=True)
        series=frame[c].sort_index();series=series[~series.index.duplicated(keep='last')]
        if index is None:
            index=pd.date_range(series.index[0].ceil('min'),series.index[-1].floor('min'),freq='min')
            raw=np.lib.format.open_memmap(DATA/'features.npy',mode='w+',dtype=np.float32,shape=(len(index),24))
        raw[:,j*4:j*4+4]=bin_statistics(series,index)
        raw.flush()
        stats.append({'channel':c,'first':series.index[0].isoformat(),'last':series.index[-1].isoformat(),'raw_rows':len(series)})
        print('Prepared new-mission channel',c,len(index),'minutes',flush=True)
        del frame,series
    # Data range is defined by first selected channel. Other missing observations
    # remain missing and coverage is reported, never filled from future samples.
    np.save(DATA/'timestamps.npy',index.as_unit('ns').asi8)
    start=index[0]+pd.Timedelta(days=365)
    adapt=index<start
    global_adapt=np.nanstd(np.asarray(raw[adapt,::4]),axis=0).clip(1e-6)
    bundle=joblib.load(path);config=ContextConfig(**bundle['config']);predict=predictor(bundle)
    torch.set_num_threads(4)
    policy=json.loads((ROOT/'lifecycle.json').read_text())
    scores={name:np.full(len(index),np.nan,dtype=np.float32) for name in ['frozen','unlabeled_scale_adaptation']}
    for name,scale in [('frozen',bundle['global_scale']),('unlabeled_scale_adaptation',global_adapt)]:
        for left in range(0,len(index),65536):
            right=min(len(index),left+65536);context=max(0,left-config.slow)
            f=context_block(raw[context:right],scale,config)[left-context:]
            good=np.isfinite(f).all(axis=1)
            if good.any():scores[name][np.arange(left,right)[good]]=predict(f[good])
        print('Scored',name,flush=True)
    # Annotation evaluation begins only after both frozen inference passes.
    annotations=load_annotation_table(folder,channels)
    labels=annotation_masks(index,annotations,channels);np.save(DATA/'labels.npy',labels)
    y=((labels&1)!=0).any(axis=1);gaps=((labels&4)!=0).any(axis=1)
    eligible=(index>=start)&~gaps&np.r_[False,~gaps[:-1]]
    eligible &= np.isfinite(scores['frozen'])&np.isfinite(scores['unlabeled_scale_adaptation'])
    summary=[];full={}
    for name,score in scores.items():
        np.save(DATA/f'{name}_scores.npy',score)
        for mode in ['instant','recovery']:
            values=np.where(eligible,score,np.nan)
            alarm=values>policy['entry_threshold'] if mode=='instant' else lifecycle_alarm(values,policy['entry_threshold'],policy['exit_threshold'],policy['quiet_minutes'],policy['max_minutes'])
            key=name+'_'+mode
            events,episodes=event_metrics(index,labels,eligible,alarm,annotations,channels,np.full(len(index),255,dtype=np.uint8),test_start=start.isoformat(),cadence_seconds=60)
            point=confusion(y[eligible],alarm[eligible]);hours=(eligible&~y).sum()/60
            metrics={'variant':key,'events_detected':sum(e['detected'] for e in events),'events_total':len(events),
                'events_scorable':sum(e['scorable'] for e in events),'false_episodes_per_1000_nominal_h':episodes['false_nonoverlapping_episodes']/hours*1000 if hours else None,
                'minute_precision':point['precision'],'minute_recall':point['recall'],'nominal_alarm_fraction':point['false_positive_rate']}
            summary.append(metrics);full[key]={**metrics,'point':point,**episodes}
            pd.DataFrame(events).to_csv(REPORT/f'{key}_events.csv',index=False)
    primary=summary[1]
    acceptable=primary['events_total']>0 and primary['events_detected']/primary['events_total']>=.5 and primary['false_episodes_per_1000_nominal_h']<=10 and primary['nominal_alarm_fraction']<=.001
    result={'source':'REAL MISSION DATA','scope':'Previously unused ESA Mission 2 channels; frozen protocol. Positional channel mapping is not verified physical equivalence.',
        'start':index[0].isoformat(),'evaluation_start':start.isoformat(),'end':index[-1].isoformat(),
        'scored_minutes':int(eligible.sum()),'evaluation_minutes':int((index>=start).sum()),'nominal_hours':float((eligible&~y).sum()/60),
        'channel_stats':stats,'summary':summary,'models':full,'target_scales':global_adapt.tolist(),
        'conclusion':('The frozen model met the illustrative transfer gate, but this single mission subset does not establish operational readiness.' if acceptable else 'The frozen model did not meet the illustrative transfer gate. Cross-mission reliability is not established; keep the deployed model restricted to experimental Mission 1 use.'),
        'gate_note':'Illustrative post-report interpretation: >=50% event detection, <=10 false episodes/1000h, <=0.1% nominal alarm time. Not an operator-approved acceptance standard.',
        'no_retuning':True}
    (REPORT/'metrics.json').write_text(json.dumps(result,indent=2))
    (REPORT/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':run()
