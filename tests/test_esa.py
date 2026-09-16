import pandas as pd
import pytest
from src.ingestion.esa import load_preprocessed_csv, load_raw_mission, apply_mapping


def test_esa_csv_preserves_names_and_separates_labels(tmp_path):
    p = tmp_path / "fixture.csv"
    pd.DataFrame(
        {
            "timestamp": ["2001-01-01", "2001-01-02"],
            "channel_12": [1, 2],
            "is_anomaly_channel_12": [0, 1],
        }
    ).to_csv(p, index=False)
    df, labels = load_preprocessed_csv(p)
    assert "channel_12" in df and "is_anomaly_channel_12" not in df
    assert (
        df.source.eq("REAL MISSION DATA").all()
        and labels.is_anomaly_channel_12.iloc[1] == 1
    )
    mapped = apply_mapping(
        df,
        {
            "channel_12": {
                "target": "test_sensor",
                "unit": "V",
                "subsystem": "UNVERIFIED",
                "scale": 2,
                "offset": 1,
            }
        },
    )
    assert mapped.test_sensor.tolist() == [3, 5]


def test_raw_loader_trust_gap_and_annotations(tmp_path):
    with pytest.raises(ValueError, match="pickle"):
        load_raw_mission(tmp_path, ["channel_1"], "2001-01-01", "2001-01-02")
    (tmp_path / "channels").mkdir()
    pd.DataFrame(
        {"channel_1": [1.0, 2.0]},
        index=pd.to_datetime(["2001-01-01 00:01Z", "2001-01-01 00:04Z"]),
    ).to_pickle(tmp_path / "channels/channel_1.zip")
    pd.DataFrame(
        {
            "ID": ["a"],
            "Channel": ["channel_1"],
            "StartTime": ["2001-01-01 00:03Z"],
            "EndTime": ["2001-01-01 00:05Z"],
        }
    ).to_csv(tmp_path / "labels.csv", index=False)
    df, labels = load_raw_mission(
        tmp_path,
        ["channel_1"],
        "2001-01-01",
        "2001-01-01 00:08Z",
        cadence="1min",
        max_hold="1min",
        trusted_pickle=True,
    )
    assert (
        pd.isna(df.channel_1.iloc[0])
        and pd.isna(df.channel_1.iloc[3])
        and pd.isna(df.channel_1.iloc[-1])
    )
    assert df.channel_1.iloc[1] == 1 and df.channel_1.iloc[4] == 2
    assert len(labels) == 1
