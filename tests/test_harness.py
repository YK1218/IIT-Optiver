"""Sanity tests for the shared harness and the encoders.

    .venv/Scripts/python.exe -m pytest tests -q

These guard the two ways results silently go wrong: a metric that is subtly mis-defined,
and a transform that has seen the validation fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.features import DenseEncoder, TreeEncoder
from src.eval.harness import evaluate, tune_threshold


def test_evaluate_perfect_separation():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluate(y, s, threshold=0.5)
    assert m.auroc == 1.0
    assert m.auprc == 1.0
    assert m.f1 == 1.0
    assert m.recall == 1.0


def test_evaluate_never_positive_classifier_scores_zero_not_high():
    """The R4 case: a constant-negative model must look bad, not 99% good."""
    y = np.zeros(1000, dtype=int)
    y[:1] = 1                       # 0.1% positive, like PaySim
    s = np.full(1000, 0.01)         # predicts "never fraud"
    m = evaluate(y, s, threshold=0.5)
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert not hasattr(m, "accuracy")


def test_evaluate_is_threshold_sensitive_but_auroc_is_not():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 500), 0, 1)
    lo, hi = evaluate(y, s, 0.2), evaluate(y, s, 0.8)
    assert lo.auroc == pytest.approx(hi.auroc)      # ranking metric, threshold-free
    assert lo.recall > hi.recall                     # a looser threshold catches more


def test_tune_threshold_beats_the_default_on_a_skewed_problem():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    s = np.clip(y * 0.35 + rng.normal(0.15, 0.12, 5000), 0, 1)
    thr, f1 = tune_threshold(y, s)
    assert f1 >= evaluate(y, s, 0.5).f1
    assert 0.0 <= thr <= 1.0


def test_recall_at_p50_respects_the_precision_floor():
    y = np.array([0] * 90 + [1] * 10)
    s = np.concatenate([np.linspace(0.0, 0.5, 90), np.linspace(0.6, 1.0, 10)])
    m = evaluate(y, s, 0.5)
    assert m.recall_at_p50 == 1.0        # perfectly separable, so full recall at any precision


# --------------------------------------------------------------------- R3: no leakage
def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "num": rng.normal(10, 3, n),
        "cat": rng.choice(list("abc"), n),
    })


def test_dense_encoder_standardises_using_train_statistics_only():
    train, val = _frame(200, 0), _frame(50, 1)
    val["num"] = val["num"] + 100.0          # a shifted validation fold
    enc = DenseEncoder(["cat"])
    enc.fit(train)
    mean_before = enc.mean_.copy()
    enc.transform(val)
    assert np.allclose(enc.mean_, mean_before), "transform() must not refit statistics"

    Ztr = enc.transform(train)
    assert abs(Ztr[:, 0].mean()) < 0.1       # train is centred
    Zva = enc.transform(val)
    assert Zva[:, 0].mean() > 1.0            # the shift survives -- it was not re-centred


def test_encoders_map_unseen_categories_to_a_sentinel_not_a_new_code():
    train = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    val = pd.DataFrame({"cat": ["a", "z"]})       # 'z' never seen at fit time

    tree = TreeEncoder(["cat"]).fit(train)
    out = tree.transform(val)["cat"].to_numpy()
    assert np.isnan(out[1]), "unseen category must become NaN, not a fresh integer code"

    dense = DenseEncoder(["cat"]).fit(train)
    Z = dense.transform(val)
    assert Z.shape[1] == dense.n_features_
    assert np.isfinite(Z).all()


def test_dense_encoder_column_count_is_stable_across_folds():
    train, val = _frame(200, 0), _frame(37, 5)
    enc = DenseEncoder(["cat"]).fit(train)
    assert enc.transform(train).shape[1] == enc.transform(val).shape[1]
