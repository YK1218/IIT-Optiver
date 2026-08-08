"""Feature encoding, fitted on the training fold only (Intern Guide R3).

Two views over one codebase (Pipeline stage 3):
  * `TreeEncoder`  -- ordinal codes + native NaN, for XGBoost.
  * `DenseEncoder` -- one-hot / frequency codes + median impute + standard scale, for the MLP.

Both learn every statistic (category vocabularies, frequencies, medians, means, stds)
from the rows handed to `fit` and nothing else. Categories unseen at fit time map to NaN
(tree view) or 0 (dense view) rather than silently extending the vocabulary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ONEHOT_MAX_CARDINALITY = 20  # above this a categorical is frequency-encoded instead


class TreeEncoder:
    """Ordinal-encode categoricals; leave numerics (and their NaNs) untouched."""

    def __init__(self, categorical: list[str]):
        self.categorical = categorical
        self.vocab_: dict[str, dict] = {}
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "TreeEncoder":
        self.columns_ = list(X.columns)
        for c in self.categorical:
            if c not in X.columns:
                continue
            cats = pd.Index(X[c].dropna().unique())
            self.vocab_[c] = {v: i for i, v in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.columns_].copy()
        for c, vocab in self.vocab_.items():
            out[c] = out[c].map(vocab).astype(np.float32)  # unseen -> NaN, XGB handles it
        return out.astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class DenseEncoder:
    """Numeric matrix for the MLP: one-hot / frequency codes, median impute, z-score."""

    def __init__(self, categorical: list[str], add_missing_indicators: bool = False):
        self.categorical = categorical
        self.add_missing_indicators = add_missing_indicators
        self.onehot_: dict[str, list] = {}
        self.freq_: dict[str, dict] = {}
        self.numeric_: list[str] = []
        self.missing_cols_: list[str] = []
        self.median_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "DenseEncoder":
        cats = [c for c in self.categorical if c in X.columns]
        self.numeric_ = [c for c in X.columns if c not in cats]

        for c in cats:
            vc = X[c].value_counts(dropna=True)
            if len(vc) <= ONEHOT_MAX_CARDINALITY:
                self.onehot_[c] = list(vc.index)
            else:
                # Frequency encoding: rare-category mass is the signal, not the id itself.
                self.freq_[c] = (vc / len(X)).to_dict()

        if self.add_missing_indicators:
            # Informative absence -- Theme 1.3's lever, off by default so that Theme 1.5
            # measures the loss function and nothing else. Only columns that are actually
            # absent in the training fold, so the column set stays stable at predict time.
            self.missing_cols_ = [c for c in X.columns if X[c].isna().any()]

        Z = self._assemble(X)
        self.median_ = np.nanmedian(Z, axis=0)
        self.median_ = np.where(np.isnan(self.median_), 0.0, self.median_)
        Z = self._impute(Z)
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._impute(self._assemble(X))
        np.subtract(Z, self.mean_, out=Z)
        np.divide(Z, self.std_, out=Z)
        return np.clip(Z, -10.0, 10.0, out=Z)  # tame heavy tails before the MLP sees them

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------ internals
    def _assemble(self, X: pd.DataFrame) -> np.ndarray:
        blocks = [X[self.numeric_].to_numpy(dtype=np.float32, copy=True)]

        for c, levels in self.onehot_.items():
            col = X[c]
            oh = np.zeros((len(X), len(levels)), dtype=np.float32)
            for j, lv in enumerate(levels):
                oh[:, j] = (col == lv).to_numpy(dtype=np.float32)
            blocks.append(oh)

        for c, freq in self.freq_.items():
            blocks.append(X[c].map(freq).to_numpy(dtype=np.float32).reshape(-1, 1))

        if self.missing_cols_:
            miss = X[self.missing_cols_].isna().to_numpy(dtype=np.float32)
            blocks.append(miss)

        return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]

    def _impute(self, Z: np.ndarray) -> np.ndarray:
        bad = ~np.isfinite(Z)
        if bad.any():
            Z[bad] = np.broadcast_to(self.median_, Z.shape)[bad]
        return Z

    @property
    def n_features_(self) -> int:
        return (
            len(self.numeric_)
            + sum(len(v) for v in self.onehot_.values())
            + len(self.freq_)
            + len(self.missing_cols_)
        )
