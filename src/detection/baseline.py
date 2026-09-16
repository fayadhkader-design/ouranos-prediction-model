import pandas as pd
from src.config import DEFAULT


def threshold_events(df, config=DEFAULT):
    values = df.copy()
    # Solar uses a context-aware conventional efficiency floor, not a night-time power floor.
    if {"solar_array_output", "illumination"} <= set(df):
        values["solar_efficiency"] = (
            df.solar_array_output / (1200 * df.illumination.clip(lower=0.25))
        ).where(df.illumination > 0.25)
    flags = {}
    for channel, (direction, limit) in config.thresholds.items():
        if channel in values:
            flags[channel] = (
                (values[channel] > limit)
                if direction == "high"
                else (values[channel] < limit)
            ).fillna(False)
    return pd.DataFrame(flags, index=df.index)
