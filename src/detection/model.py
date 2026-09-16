"""Healthy context regression + subsystem multivariate Isolation Forests.
Normalization fitted on independent healthy calibration sequences.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest
from src.config import CONTEXT, SUBSYSTEMS, DEFAULT
from src.preprocessing.features import engineer, relationship_signals
from src.ingestion.schema import validate_telemetry


class NormalBehaviorModel:
    def __init__(self, config=DEFAULT):
        self.config = config
        self.groups = {k: list(v) for k, v in SUBSYSTEMS.items()}
        self.groups["ADCS"] += [f"wheel_{i}_current_per_krpm" for i in (1, 2, 3)]
        self.groups["BATTERY"] += ["battery_voltage_current_relationship"]
        self.groups["POWER"] += ["solar_orbital_relationship"]
        self.channels = list(
            dict.fromkeys(c for cs in self.groups.values() for c in cs)
        )
        self.fitted = False

    def context(self, df):
        x = df[CONTEXT].astype(float)
        # Interactions represent mode-dependent solar/thermal response.
        return np.column_stack(
            [
                x.to_numpy(),
                df.illumination * df.commanded_load,
                df.orbital_sin * df.commanded_load,
            ]
        )

    def residuals(self, df):
        expanded = relationship_signals(df)
        expected = pd.DataFrame(
            self.regression.predict(self.context(df)),
            columns=self.channels,
            index=df.index,
        )
        z = (expanded[self.channels] - expected) / self.scale
        return z, expected

    def features(self, z):
        return engineer(z, self.config.windows, self.config.cadence_minutes / 60)

    def selected(self, features, group):
        # Compact multivariate representation: contextual residuals, level and variability.
        suffixes = (
            "__residual",
            f"__mean_{self.config.windows[1]}",
            f"__std_{self.config.windows[1]}",
            f"__slope_{self.config.windows[-1]}",
        )
        return [c + s for c in self.groups[group] for s in suffixes]

    def fit(self, train, calibration):
        train, _ = validate_telemetry(
            train, cadence_minutes=self.config.cadence_minutes
        )
        calibration, _ = validate_telemetry(
            calibration, cadence_minutes=self.config.cadence_minutes
        )
        if len(train) < 500 or len(calibration) < 300:
            raise ValueError(
                "Need >=500 healthy training and >=300 calibration observations"
            )
        self.regression = Ridge(alpha=1.0).fit(
            self.context(train), relationship_signals(train)[self.channels]
        )
        residual = relationship_signals(train)[self.channels] - self.regression.predict(
            self.context(train)
        )
        self.scale = residual.std().clip(lower=1e-6)
        self.context_min = train[CONTEXT].min()
        self.context_max = train[CONTEXT].max()
        z, _ = self.residuals(train)
        zc, _ = self.residuals(calibration)
        ft = self.features(z)
        fc = self.features(zc)
        self.forests = {}
        self.normalizers = {}
        for group in self.groups:
            cols = self.selected(ft, group)
            forest = IsolationForest(
                n_estimators=64,
                max_samples=512,
                random_state=101,
                contamination="auto",
                n_jobs=1,
            )
            forest.fit(ft[cols].iloc[self.config.warmup :])
            scores = -forest.score_samples(fc[cols].iloc[self.config.warmup :])
            self.forests[group] = forest
            self.normalizers[group] = (
                float(np.quantile(scores, 0.50)),
                float(np.quantile(scores, 0.999)),
            )
        self.training_metadata = {
            "training_rows": len(train),
            "calibration_rows": len(calibration),
            "training_source": str(
                train.get("source", pd.Series(["UNSPECIFIED"])).iloc[0]
            ),
            "features": len(ft.columns),
            "model_version": "0.1.0",
        }
        self.fitted = True
        return self

    def score(self, df):
        if not self.fitted:
            raise ValueError("Train a healthy behavior model before scoring")
        z, expected = self.residuals(df)
        features = self.features(z)
        scores = {}
        for group, forest in self.forests.items():
            low, high = self.normalizers[group]
            raw = -forest.score_samples(features[self.selected(features, group)])
            # q99.9 maps to 0.5; unusual healthy points alone cannot drive warnings.
            scores[group] = np.clip((raw - low) / (2 * max(high - low, 0.01)), 0, 1)
        support = (
            (df[CONTEXT] >= self.context_min - 1e-6)
            & (df[CONTEXT] <= self.context_max + 1e-6)
        ).mean(axis=1)
        return z, expected, features, pd.DataFrame(scores, index=df.index), support
