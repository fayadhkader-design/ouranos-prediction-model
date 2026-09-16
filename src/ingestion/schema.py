"""Strict timestamp checks and bounded, past-only imputation."""

import numpy as np
import pandas as pd
from src.config import CHANNELS, CONTEXT


def validate_telemetry(df, required=None, cadence_minutes=None, max_gap_samples=2):
    if df.empty:
        raise ValueError("Telemetry is empty")
    if "timestamp" not in df:
        raise ValueError("timestamp is required")
    out = df.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, errors="raise")
    if (
        out.timestamp.isna().any()
        or out.timestamp.duplicated().any()
        or not out.timestamp.is_monotonic_increasing
    ):
        raise ValueError("Timestamps must be unique, non-null and increasing")
    required = list(required) if required is not None else CHANNELS + CONTEXT
    missing = set(required) - set(out)
    if missing:
        raise ValueError(f"Missing required channels/context: {sorted(missing)}")
    if cadence_minutes is not None and len(out) > 1:
        if not np.allclose(
            out.timestamp.diff().dropna().dt.total_seconds(), cadence_minutes * 60
        ):
            raise ValueError("Irregular cadence: resample explicitly before scoring")
    numeric = (
        out[required]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
    )
    quality = 1 - numeric.isna().mean(axis=1)
    numeric = numeric.ffill(limit=max_gap_samples)
    if numeric.isna().any().any():
        raise ValueError(
            "Leading missing values or gaps exceed the bounded causal fill limit"
        )
    out[required] = numeric
    return out, quality
