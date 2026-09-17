"""Usable offline inference for the frozen real-ESA context model.
Input: UTC timestamp + channel_41,...,channel_46 numeric observations.
This is anomaly detection, not calibrated failure probability or the Ouranos risk score.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.neural.contextual import ROOT, ContextConfig, context_block, predictor
from src.neural.supervised import CHANNELS
from src.neural.evaluate import sustained_array


class ContextualDetector:
    def __init__(self, model_dir=ROOT):
        model_dir = Path(model_dir)
        self.selection = json.loads((model_dir/'selection.json').read_text())
        self.name = self.selection['winner']
        self.bundle = joblib.load(model_dir/f'{self.name}.joblib')
        self.config = ContextConfig(**self.bundle['config'])
        self.policy = self.selection['candidates'][self.name]
        self.predict = predictor(self.bundle)

    def score_bins(self, index, bins):
        """Score right-closed minute bins, including supplied prior history.
        Nonfinite/warmup rows remain UNSCORED. No annotation labels are required.
        """
        index = pd.DatetimeIndex(index)
        if index.tz is None or index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError('Provide unique, increasing timezone-aware UTC minute timestamps')
        index = index.tz_convert('UTC').as_unit('ns')
        if len(index)>1 and not np.all(np.diff(index.asi8)==60_000_000_000):
            raise ValueError('Bins must be spaced exactly one minute; keep missing bins as NaN')
        bins = np.asarray(bins,dtype=np.float32)
        if bins.shape != (len(index),24):
            raise ValueError('Expected 24 columns: last/min/max/std for each of channels 41–46')
        score = np.full(len(index),np.nan,dtype=np.float32)
        channel = np.full(len(index),None,dtype=object)
        deviation = np.full(len(index),np.nan,dtype=np.float32)
        for start in range(0,len(index),65536):
            end = min(len(index),start+65536)
            left = max(0,start-self.config.slow)
            f = context_block(bins[left:end],self.bundle['global_scale'],self.config)[start-left:]
            good = np.isfinite(f).all(axis=1)
            positions = np.arange(start,end)[good]
            if len(positions):
                score[positions] = self.predict(f[good])
                residuals = np.abs(f[good].reshape(-1,6,10)[:,:,:3]).max(axis=2)
                channel[positions] = np.array(CHANNELS)[residuals.argmax(axis=1)]
                deviation[positions] = residuals.max(axis=1)
        alarm = sustained_array(np.isfinite(score)&(score>self.policy['threshold']),self.policy['dwell'])
        return pd.DataFrame({'timestamp':index,'model_score':score,'alert_threshold':self.policy['threshold'],
            'status':np.where(~np.isfinite(score),'UNSCORED',np.where(alarm,'ANOMALY','NO ALERT')),
            'largest_recent_deviation_channel':channel,'largest_recent_deviation_standardized':deviation,
            'source':'USER-SUPPLIED TELEMETRY — origin not verified'},index=index)

    def score_frame(self, frame):
        missing = set(['timestamp',*CHANNELS])-set(frame.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        index = pd.to_datetime(frame.timestamp,utc=True,format='mixed',errors='raise')
        if index.duplicated().any() or not index.is_monotonic_increasing:
            raise ValueError('Input timestamps must be unique and increasing')
        values = frame[CHANNELS].apply(pd.to_numeric,errors='raise').copy()
        values.index = pd.DatetimeIndex(index)
        if np.isinf(values.to_numpy()).any():
            raise ValueError('Infinite sensor values are invalid; use NaN for missing observations')
        binned = values.resample('60s',closed='right',label='right').agg(['last','min','max','std'])
        for c in CHANNELS:
            binned[(c,'std')] = binned[(c,'std')].fillna(0)
        return self.score_bins(binned.index,binned.to_numpy(dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--recovery',action='store_true',help='Use the calibrated alert recovery policy, retaining instantaneous decisions in a separate column')
    args = parser.parse_args()
    result = ContextualDetector().score_frame(pd.read_csv(args.input))
    if args.recovery:
        from src.neural.lifecycle import apply_recovery
        policy=json.loads((ROOT/'lifecycle.json').read_text())
        result=apply_recovery(result,policy)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(json.dumps({'output':str(args.output),'rows':len(result),'status_counts':result.status.value_counts().to_dict(),
        'note':'Requires about one day of prior telemetry. Scores are not probabilities. Anonymous ESA channel schema only.'},indent=2))


if __name__=='__main__':
    main()
