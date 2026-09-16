# Optional ESA Anomalies Dataset ingestion

Inspected the official [ESA-ADB repository](https://github.com/kplabs-pl/ESA-ADB) and its [Mission1 preparation source](https://github.com/kplabs-pl/ESA-ADB/blob/main/notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py) on 2026-09-16. The repository is benchmark code; download the separate data from [Zenodo DOI 10.5281/zenodo.12528696](https://doi.org/10.5281/zenodo.12528696). The Zenodo page rate-limited this development session; no mission archive was downloaded and no real-data performance is claimed.

## Supported layouts

The official raw reader uses `channels/<channel>.zip` (pandas DataFrames serialized with pickle), `labels.csv` with `Channel`, `StartTime`, `EndTime`, `ID`, and `anomaly_types.csv`. Place the mission directory under `data/raw/ESA-Mission1/`. The adapter reads selected channels/time intervals, uses past-only bounded zero-order hold, and preserves missing gaps. It reads complete selected channel pickle files before selecting time intervals, so memory must accommodate those files. Pickle can execute code: acknowledge trusted provenance explicitly; never load uploads/untrusted pickle files.

```python
from src.ingestion.esa import load_raw_mission

telemetry, annotations = load_raw_mission(
    "data/raw/ESA-Mission1",
    ["channel_12", "channel_13"],
    start="2001-01-01",
    end="2001-01-02",
    trusted_pickle=True,
)
```

For CSVs created by the official preprocessing pipeline:

```bash
python -m src.ingestion.esa /path/to/84_months.test.csv --out data/processed/esa
```

`is_anomaly_*` columns are separated from measurements and never become model features. Raw annotations preserve categories such as anomaly, rare event and gap; do not collapse rare nominal events into spacecraft failures. The upstream preprocessing includes backward filling; for causal evaluation prefer our bounded raw adapter and explicitly audit missingness.

## Mapping and scientific boundary

Anonymized channel identifiers do not establish physical units or subsystem identity. Preserve names unless mission documentation supports an explicit JSON mapping:

```json
{
  "your_verified_channel": {
    "target": "battery_voltage", "unit": "V", "subsystem": "BATTERY",
    "scale": 1.0, "offset": 0.0
  }
}
```

The example intentionally does not identify a real channel. Add `--mapping path.json` only after verification. Mission-specific context inputs (mode, phase, illumination and commanded load), unit conversions, channel registry, cadence, safety limits, known-healthy training/calibration splits and event labels must be specified before real inference. The V0 synthetic model explicitly rejects REAL MISSION DATA. Loader tests use fabricated file-format fixtures and are not validation on ESA telemetry. Telecommands are not currently ingested; extend the adapter if needed to condition on them.

Do not compare the synthetic evaluation metrics to ESA-ADB benchmark results. ESA annotations support anomaly detection evaluation, not arbitrary future-failure prediction. Honor the dataset's own license/citation terms separately from the benchmark code.
