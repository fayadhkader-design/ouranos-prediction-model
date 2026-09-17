import numpy as np
import pytest

pytest.importorskip('torch')
from src.neural.contextual import (
    ContextConfig, prior_stats, context_block, extract,
    augment_features, calibration_trial, choose_threshold, network, predictor,
)


def telemetry(n=120):
    rng = np.random.default_rng(6)
    values = rng.normal(size=(n, 6)).cumsum(axis=0) * .05 + 10
    raw = np.empty((n, 24), dtype=np.float32)
    raw[:, ::4] = values
    raw[:, 1::4] = values - .1
    raw[:, 2::4] = values + .1
    raw[:, 3::4] = .03
    return raw


def test_prior_statistics_exclude_current_and_future():
    x = np.arange(20, dtype=float)
    mean, std = prior_stats(x, 4, 1.)
    assert mean[4] == 1.5
    assert std[4] == pytest.approx(np.std([0, 1, 2, 3]))
    x[4:] = 1000
    altered, _ = prior_stats(x, 4, 1.)
    assert altered[4] == mean[4]
    assert np.isnan(mean[:4]).all()


def test_context_is_causal_and_constant_level_invariant():
    cfg = ContextConfig(fast=5, slow=20, coverage=1.)
    x = telemetry()
    original = context_block(x, np.ones(6), cfg)
    x[80:] += 100
    np.testing.assert_allclose(context_block(x, np.ones(6), cfg)[:80], original[:80], equal_nan=True)
    shifted = telemetry().astype(np.float64)
    for j in range(6):
        shifted[:, j*4:j*4+3] += (j+1) * 20
    np.testing.assert_allclose(context_block(shifted, np.ones(6), cfg), original, atol=2e-5, equal_nan=True)


def test_chunked_inference_matches_whole_history():
    cfg = ContextConfig(fast=5, slow=20)
    raw = telemetry()
    ids = np.array([119, 70, 40, 21, 33])
    full = context_block(raw, np.ones(6), cfg)[ids]
    batched = extract(raw, ids, np.ones(6), cfg, chunk=32)
    np.testing.assert_allclose(full, batched, atol=2e-5)


def test_missing_history_does_not_become_normal():
    cfg = ContextConfig(fast=5, slow=20, coverage=1.)
    raw = telemetry()
    raw[35:45] = np.nan
    f = context_block(raw, np.ones(6), cfg)
    assert not np.isfinite(f[45:65]).all(axis=1).any()
    assert np.isfinite(f[65:]).all()


def test_augmentation_is_deterministic_channel_coherent_and_nonmutating():
    x = np.ones((10, 60), dtype=np.float32)
    a = augment_features(x, np.random.default_rng(4))
    b = augment_features(x, np.random.default_rng(4))
    np.testing.assert_array_equal(a, b)
    assert (x == 1).all() and (a >= .85).all() and (a <= 1.15).all()
    assert np.all(a.reshape(10, 6, 10).std(axis=2) < 1e-6)


def test_threshold_selection_rejects_always_alarm_and_breaks_dwell_at_gap():
    scores = np.linspace(0, 1, 1000).astype(np.float32)
    y = np.zeros(1000, dtype=bool)
    y[-10:] = True
    eligible = np.ones(1000, dtype=bool)
    groups = [('a', [(990, 1000)])]
    chosen, _ = choose_threshold(scores, eligible, y, groups, ContextConfig())
    assert chosen['detected_events'] == 1
    assert chosen['nominal_alarm_fraction'] <= .001
    assert chosen['false_starts_per_1000h'] <= 10
    eligible[998] = False
    result = calibration_trial(scores, eligible, y, [('b', [(999,1000)])], .9, 3)
    assert result['detected_events'] == 0


def test_serialized_predictor_roundtrip(tmp_path):
    import joblib
    bundle = {'kind':'neural', 'state_dict':network().state_dict(), 'center':np.zeros(60,dtype=np.float32), 'scale':np.ones(60,dtype=np.float32)}
    x = np.random.default_rng(4).normal(size=(4,60)).astype(np.float32)
    path = tmp_path/'model.joblib'
    joblib.dump(bundle,path)
    np.testing.assert_array_equal(predictor(bundle)(x),predictor(joblib.load(path))(x))


def test_public_inference_handles_warmup_missingness_and_input_errors(tmp_path):
    import json
    import joblib
    import pandas as pd
    from dataclasses import asdict
    from src.neural.predict_contextual import ContextualDetector
    cfg=ContextConfig(fast=5,slow=20,coverage=1.)
    bundle={'kind':'neural','state_dict':network().state_dict(),'center':np.zeros(60,dtype=np.float32),'scale':np.ones(60,dtype=np.float32),
            'config':asdict(cfg),'global_scale':np.ones(6)}
    joblib.dump(bundle,tmp_path/'example.joblib')
    (tmp_path/'selection.json').write_text(json.dumps({'winner':'example','candidates':{'example':{'threshold':100.,'dwell':1}}}))
    detector=ContextualDetector(tmp_path)
    index=pd.date_range('2020-01-01',periods=120,freq='min',tz='UTC')
    raw=telemetry()
    out=detector.score_bins(index,raw)
    assert (out.status.iloc[:20]=='UNSCORED').all()
    assert (out.status.iloc[20:]=='NO ALERT').all()
    assert out.largest_recent_deviation_channel.iloc[20:].notna().all()
    with pytest.raises(ValueError,match='one minute'):
        detector.score_bins(index[::2],raw[::2])
    with pytest.raises(ValueError,match='Missing columns'):
        detector.score_frame(pd.DataFrame({'timestamp':index}))
    raw[50:60]=np.nan
    out=detector.score_bins(index,raw)
    assert (out.status.iloc[50:80]=='UNSCORED').all()


def test_csv_inference_accepts_microsecond_datetime_index(tmp_path):
    import json
    import joblib
    import pandas as pd
    from dataclasses import asdict
    from src.neural.predict_contextual import ContextualDetector
    from src.neural.supervised import CHANNELS
    cfg=ContextConfig(fast=5,slow=20,coverage=1.)
    bundle={'kind':'neural','state_dict':network().state_dict(),'center':np.zeros(60,dtype=np.float32),'scale':np.ones(60,dtype=np.float32),
            'config':asdict(cfg),'global_scale':np.ones(6)}
    joblib.dump(bundle,tmp_path/'example.joblib')
    (tmp_path/'selection.json').write_text(json.dumps({'winner':'example','candidates':{'example':{'threshold':100.,'dwell':1}}}))
    frame=pd.DataFrame(telemetry()[:,::4],columns=CHANNELS)
    frame.insert(0,'timestamp',pd.date_range('2020-01-01',periods=len(frame),freq='min',tz='UTC').as_unit('us'))
    result=ContextualDetector(tmp_path).score_frame(frame)
    assert len(result)==len(frame)
    assert result.status.eq('NO ALERT').sum()==100
