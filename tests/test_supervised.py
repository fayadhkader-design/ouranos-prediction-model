import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from src.neural.supervised import bin_statistics, features, classifier, select_examples


def test_bin_keeps_spike_and_never_looks_ahead():
    times = pd.to_datetime(
        [
            "2000-01-01 00:00:10Z",
            "2000-01-01 00:00:20Z",
            "2000-01-01 00:00:50Z",
            "2000-01-01 00:01:10Z",
        ]
    )
    series = pd.Series([1.0, 9.0, 1.0, 100.0], index=times)
    index = pd.date_range("2000-01-01 00:01:00Z", periods=2, freq="60s")
    result = bin_statistics(series, index)
    assert result[0, 0] == 1 and result[0, 1] == 1 and result[0, 2] == 9
    prefix = bin_statistics(series.iloc[:3], index[:1])
    np.testing.assert_allclose(result[:1], prefix)
    assert result[1, 0] == 100


def test_empty_minute_stays_missing():
    index = pd.date_range("2000-01-01 00:01:00Z", periods=3, freq="60s")
    s = pd.Series([1.0, 2.0], index=[index[0], index[2]])
    r = bin_statistics(s, index)
    assert np.isnan(r[1, :3]).all()


def test_supervised_features_are_causal():
    x = np.arange(240, dtype=np.float32).reshape(10, 24)
    a = features(x, np.array([2, 3]), np.zeros(30), np.ones(30) * 100)
    x[4:] = 9999
    b = features(x, np.array([2, 3]), np.zeros(30), np.ones(30) * 100)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, 30) and np.isfinite(a).all()


def test_classifier_has_supervised_gradients():
    model = classifier()
    x = torch.randn(8, 30)
    y = torch.tensor([0.0, 1.0] * 4)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_positive_event_sampling_respects_split_and_caps():
    index = pd.date_range("2000-01-01", periods=50, freq="60s", tz="UTC")
    annotations = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "Category": ["Anomaly", "Anomaly"],
            "StartTime": [index[3], index[30]],
            "EndTime": [index[15], index[35]],
        }
    )
    y = np.zeros(50, dtype=bool)
    y[3:16] = True
    y[30:36] = True
    mask = np.arange(50) < 25
    ids, positives = select_examples(
        mask, y, index, annotations, 7, negative_count=8, per_event=4
    )
    assert positives == 4 and len(ids) == 12 and (ids < 25).all()
