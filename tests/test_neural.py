"""Small fabricated fixtures test mechanics; performance is measured on real ESA data."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from src.neural.config import NeuralConfig
from src.neural.prepare import annotation_masks, split_codes, valid_window_ends
from src.neural.network import WindowAutoencoder, window_batch
from src.neural.train import fit_scaler, checkpoint_model
from src.neural.evaluate import sustained_array, event_metrics
from src.ingestion.esa_download import ArchiveTail


def test_temporal_splits_are_chronological():
    dates = pd.to_datetime(
        [
            "2006-06-30T23:59:30Z",
            "2006-07-01T00:00:00Z",
            "2006-10-01T00:00:00Z",
            "2007-01-01T00:00:00Z",
        ]
    )
    assert split_codes(dates).tolist() == [0, 1, 2, 3]


def test_windows_do_not_bridge_missing_or_excluded_points():
    valid = np.ones(20, dtype=bool)
    valid[9] = False
    ends = valid_window_ends(valid, 4)
    assert set(range(9, 13)).isdisjoint(ends)
    assert 8 in ends and 13 in ends
    assert len(valid_window_ends(np.zeros(10, dtype=bool), 4)) == 0


def test_annotation_bits_preserve_rare_events_and_subcadence_anomalies():
    index = pd.date_range("2001-01-01", periods=6, freq="30s", tz="UTC")
    rows = pd.DataFrame(
        {
            "Channel": ["x", "x", "x"],
            "StartTime": [index[0] + pd.Timedelta(seconds=1), index[1], index[4]],
            "EndTime": [index[0] + pd.Timedelta(seconds=2), index[2], index[5]],
            "Category": ["Anomaly", "Rare Event", "Communication Gap"],
        }
    )
    mask = annotation_masks(index, rows, ["x"])[:, 0]
    assert mask[1] == 3 and mask[2] == 2 and mask[4] == 4
    assert mask[0] == 0


def test_scaler_ignores_test_extremes():
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    allowed = np.arange(20) < 10
    first = fit_scaler(x, allowed)
    x[10:] = 1e8
    second = fit_scaler(x, allowed)
    np.testing.assert_equal(first[0], second[0])
    np.testing.assert_equal(first[1], second[1])


def test_network_gradients_checkpoint_and_causal_windows():
    torch.manual_seed(1)
    x = np.random.default_rng(1).normal(size=(200, 6)).astype(np.float32)
    center = np.zeros(6, dtype=np.float32)
    scale = np.ones(6, dtype=np.float32)
    model = WindowAutoencoder()
    batch = window_batch(x, np.array([63, 100]), 64, center, scale)
    assert batch.shape == (2, 64, 6)
    with torch.no_grad():
        before = model(batch).numpy()
    x[101:] = 10000
    with torch.no_grad():
        after = model(window_batch(x, [63, 100], 64, center, scale)).numpy()
    np.testing.assert_array_equal(before, after)
    loss = (model(batch) - batch).square().mean()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    restored = checkpoint_model(
        {"config": NeuralConfig().to_dict(), "state_dict": model.state_dict()}
    )
    with torch.no_grad():
        np.testing.assert_allclose(restored(batch), model(batch))


def test_sustained_rule_and_breaks():
    assert sustained_array([0, 1, 1, 1, 1, 0, 1, 1], 3).tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]


def test_event_union_does_not_count_alert_in_unannotated_hole():
    index = pd.date_range("2007-01-01", periods=12, freq="30s", tz="UTC")
    annotations = pd.DataFrame(
        {
            "ID": ["event", "event"],
            "Channel": ["x", "x"],
            "Category": ["Anomaly", "Anomaly"],
            "StartTime": [index[1], index[9]],
            "EndTime": [index[2], index[10]],
        }
    )
    labels = annotation_masks(index, annotations, ["x"])
    test = np.ones(12, dtype=bool)
    alarm = np.zeros(12, dtype=bool)
    alarm[5:8] = True
    events, episodes = event_metrics(
        index, labels, test, alarm, annotations, ["x"], np.zeros(12, dtype=np.uint8)
    )
    assert not events[0]["detected"] and episodes["false_nonoverlapping_episodes"] == 1
    alarm[9] = True
    events, episodes = event_metrics(
        index, labels, test, alarm, annotations, ["x"], np.zeros(12, dtype=np.uint8)
    )
    assert events[0]["detected"] and events[0]["detection_delay_minutes"] == 4


def test_archive_tail_rejects_uncached_reads():
    stream = ArchiveTail(100, b"abcdefghij")
    with pytest.raises(ValueError):
        stream.read(2)
    stream.seek(-5, 2)
    assert stream.read(3) == b"fgh"


def test_mixed_iso_timestamp_precision(tmp_path):
    from src.neural.evaluate import read_annotations

    path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "StartTime": [
                "2012-01-01 00:00:00.123000+00:00",
                "2012-01-02 00:00:00+00:00",
            ],
            "EndTime": [
                "2012-01-01 00:01:00+00:00",
                "2012-01-02 00:01:00.456000+00:00",
            ],
        }
    ).to_csv(path, index=False)
    parsed = read_annotations(path)
    assert parsed.StartTime.notna().all() and parsed.EndTime.notna().all()
    assert parsed.StartTime.iloc[0].microsecond == 123000
