"""Compact undercomplete temporal autoencoder; present/past samples only."""

import torch
from torch import nn


class WindowAutoencoder(nn.Module):
    def __init__(self, channels=6, window=64, hidden=64, bottleneck=12):
        super().__init__()
        self.channels, self.window = channels, window
        width = channels * window
        self.encoder = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.GELU(), nn.Linear(hidden, width)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x.flatten(1))).reshape(
            -1, self.window, self.channels
        )


def window_batch(values, ends, window, mean, scale):
    """Explicit terminal indices preserve causal windows, including at inference."""
    import numpy as np

    offsets = np.arange(window - 1, -1, -1)
    windows = values[np.asarray(ends)[:, None] - offsets]
    return torch.from_numpy(
        np.ascontiguousarray((windows - mean) / scale, dtype=np.float32)
    )
