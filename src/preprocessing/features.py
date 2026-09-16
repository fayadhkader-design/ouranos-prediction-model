"""All rolling operations are trailing; never centered, backfilled or future-fitted."""

import numpy as np
import pandas as pd


def rolling_slope(series, window, step_hours):
    x = np.arange(window, dtype=float)
    x -= x.mean()
    values = np.zeros(len(series))
    if len(series) >= window:
        values[window - 1 :] = (
            np.convolve(series.to_numpy(), x[::-1], mode="valid")
            / np.dot(x, x)
            / step_hours
        )
    return pd.Series(values, index=series.index)


def engineer(residuals, windows=(6, 24, 72), step_hours=1 / 12):
    features = {}
    for c in residuals:
        s = residuals[c]
        features[c + "__residual"] = s
        features[c + "__rate"] = s.diff().fillna(0) / step_hours
        for w in windows:
            roll = s.rolling(w, min_periods=w)
            for stat in ("mean", "std", "min", "max"):
                features[f"{c}__{stat}_{w}"] = getattr(roll, stat)().fillna(0)
            features[f"{c}__slope_{w}"] = rolling_slope(s, w, step_hours)
        features[c + "__variance_change"] = (
            s.rolling(windows[0]).var() / (s.rolling(windows[-1]).var() + 0.1)
        ).fillna(1)
    return pd.DataFrame(features, index=residuals.index)


def relationship_signals(df):
    """Diagnostic cross-sensor relationships; learned expected values remove context."""
    out = df.copy()
    for i in (1, 2, 3):
        out[f"wheel_{i}_current_per_krpm"] = df[f"reaction_wheel_{i}_current"] / (
            df[f"reaction_wheel_{i}_rpm"].abs().clip(lower=100) / 1000
        )
    out["battery_voltage_current_relationship"] = (
        df.battery_voltage + 0.08 * df.battery_current
    )
    # Night-time ratio is undefined. A zero residual form stays finite at eclipse.
    out["solar_orbital_relationship"] = df.solar_array_output - 1200 * df.illumination
    return out
