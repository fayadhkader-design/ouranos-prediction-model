import pandas as pd
from src.preprocessing.features import rolling_slope


def trend_evidence(z, config):
    step = config.cadence_minutes / 60
    smooth = z.ewm(span=config.windows[1], adjust=False).mean()
    slopes = pd.DataFrame(
        {c: rolling_slope(z[c], config.windows[-1], step) for c in z}, index=z.index
    )
    # Signed multi-hour direction must agree with smoothed sample changes.
    diff = smooth.diff()
    positive = (
        (diff > 0).rolling(config.windows[-1], min_periods=config.windows[-1]).mean()
    )
    coherence = positive.where(slopes >= 0, 1 - positive).fillna(0)
    persistence = ((coherence - 0.55) / 0.35).clip(0, 1)
    magnitude = ((slopes.abs() - 0.08) / 0.55).clip(0, 1)
    level = (smooth.abs() / 5).clip(0, 1)
    degradation = magnitude * persistence * level
    return {
        "channel_degradation": degradation,
        "slopes": slopes,
        "coherence": coherence,
        "smoothed": smooth,
    }
