"""run_experiment(config) -- train, evaluate, append one row per split to results.csv.

    python -m src.run configs/step0_uci_xgb_baseline.yaml
    python -m src.run configs/t15_*.yaml          # globs work

One config file per experiment (R5); anyone can re-run any number in results.csv with
the command above and nothing else.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import paths
from .data import features, loaders, splits
from .eval import harness


def _resample(X_train, y_train, cfg, dataset):
    """SMOTE on the training fold only (R3). Never touches val/test."""
    from imblearn.over_sampling import SMOTE

    strategy = cfg.get("sampling_strategy", loaders.DATASETS[dataset]["smote_strategy"])
    sm = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=cfg.get("k_neighbors", 5),
        random_state=cfg.get("seed", 42),
    )
    return sm.fit_resample(X_train, y_train)


def _fit_xgboost(X, y, Xv, yv, imb, mcfg, seed):
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=1, reg_lambda=1.0,
        eval_metric="aucpr", early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=seed,
    )
    params.update(mcfg.get("params", {}))
    if imb["method"] == "class_weight":
        n_pos = max(int(y.sum()), 1)
        params["scale_pos_weight"] = (len(y) - n_pos) / n_pos
    clf = XGBClassifier(**params)
    clf.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    return clf, lambda M: clf.predict_proba(M)[:, 1]


def _fit_mlp(X, y, Xv, yv, imb, mcfg, seed):
    from .models.mlp import MLPClassifier

    loss = {"none": "bce", "smote": "bce",
            "class_weight": "weighted_bce", "focal": "focal"}[imb["method"]]
    clf = MLPClassifier(
        loss=loss,
        gamma=imb.get("gamma", 2.0),
        alpha=imb.get("alpha", 0.25),
        hidden=tuple(mcfg.get("hidden", (256, 128))),
        dropout=mcfg.get("dropout", 0.2),
        lr=mcfg.get("lr", 1e-3),
        batch_size=mcfg.get("batch_size", 1024),
        epochs=mcfg.get("epochs", 30),
        patience=mcfg.get("patience", 5),
        seed=seed,
    )
    clf.fit(X, y, Xv, yv)
    return clf, clf.predict_proba


def _fit_logreg(X, y, Xv, yv, imb, mcfg, seed):
    """Conventional reference model from Pipeline stage 4."""
    from sklearn.linear_model import LogisticRegression

    params = dict(max_iter=1000, C=1.0, random_state=seed)
    params.update(mcfg.get("params", {}))
    if imb["method"] == "class_weight":
        params["class_weight"] = "balanced"
    clf = LogisticRegression(**params)
    clf.fit(X, y)
    return clf, lambda M: clf.predict_proba(M)[:, 1]


FITTERS = {"xgboost": _fit_xgboost, "mlp": _fit_mlp, "logreg": _fit_logreg}

# logreg needs the same dense, scaled matrix the MLP gets.
DENSE_MODELS = {"mlp", "logreg"}


def run_experiment(config: dict, config_path: str = "") -> pd.DataFrame:
    t0 = time.time()
    seed = config.get("seed", 42)
    dataset = config["dataset"]
    mcfg = config["model"]
    model_name = mcfg["name"]
    imb = config.get("imbalance", {"method": "none"})

    print(f"\n=== {config['experiment']}  [{dataset} | {model_name} | {imb['method']}] ===",
          flush=True)

    df = loaders.load(dataset)
    # Defaults to the mandated stratified split. `split: <dataset>_grouped` selects the
    # entity-disjoint companion, used to measure how much a random split inflates a
    # score by putting the same customer in train and test.
    split_name = config.get("split", dataset)
    idx = splits.load(split_name)
    feat_cols = loaders.feature_columns(df)
    # `drop_features` supports leakage ablations: e.g. removing PaySim's post-transaction
    # balance columns to check the baseline is not reading the outcome off the ledger.
    drop = set(config.get("drop_features", []))
    if drop:
        feat_cols = [c for c in feat_cols if c not in drop]
        print(f"  dropped {len(drop)} feature(s): {sorted(drop)}", flush=True)
    categorical = [c for c in loaders.DATASETS[dataset]["categorical"] if c in feat_cols]

    y = df["y"].to_numpy().astype(np.int8)
    y_tr, y_va, y_te = y[idx["train"]], y[idx["val"]], y[idx["test"]]

    # --- encode: fitted on the training fold only -----------------------------------
    if model_name in DENSE_MODELS:
        enc = features.DenseEncoder(
            categorical,
            add_missing_indicators=config.get("missing_indicators", False),
        )
    else:
        enc = features.TreeEncoder(categorical)
    X_tr = enc.fit_transform(df.iloc[idx["train"]][feat_cols])
    X_va = enc.transform(df.iloc[idx["val"]][feat_cols])
    X_te = enc.transform(df.iloc[idx["test"]][feat_cols])
    del df

    # SMOTE cannot interpolate across NaNs, so the tree view has to be imputed first.
    # The imputation must then apply to val/test too -- otherwise the model learns splits
    # on imputed values and meets raw NaNs at inference. Medians come from the training
    # fold only (R3). `impute_nans` lets an ablation isolate this effect from SMOTE's.
    if model_name == "xgboost" and (imb["method"] == "smote"
                                    or config.get("impute_nans", False)):
        med = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(med).fillna(0.0)
        X_va = X_va.fillna(med).fillna(0.0)
        X_te = X_te.fillna(med).fillna(0.0)
        print("  median-imputed NaNs (train medians) across all folds", flush=True)

    if imb["method"] == "smote":
        n_before = len(y_tr)
        X_tr, y_tr = _resample(X_tr, y_tr, imb, dataset)
        print(f"  SMOTE: {n_before:,} -> {len(y_tr):,} rows "
              f"(pos {y_tr.sum():,} = {y_tr.mean():.3%})", flush=True)

    # --- train ----------------------------------------------------------------------
    clf, score = FITTERS[model_name](X_tr, y_tr, X_va, y_va, imb, mcfg, seed)
    train_seconds = time.time() - t0

    # --- threshold tuned on validation, then applied unchanged to test (R2) ----------
    s_va = score(X_va)
    thr, _ = harness.tune_threshold(y_va, s_va)
    m_va = harness.evaluate(y_va, s_va, thr)
    m_te = harness.evaluate(y_te, score(X_te), thr)
    print(f"  val  AUROC={m_va.auroc:.4f} AUPRC={m_va.auprc:.4f} "
          f"F1={m_va.f1:.4f} recall={m_va.recall:.4f} thr={thr:.4g}", flush=True)
    print(f"  TEST AUROC={m_te.auroc:.4f} AUPRC={m_te.auprc:.4f} "
          f"F1={m_te.f1:.4f} recall={m_te.recall:.4f}", flush=True)

    run_id = uuid.uuid4().hex[:8]
    rows = []
    for split_name, m in (("val", m_va), ("test", m_te)):
        row = {
            "run_id": run_id, "timestamp": harness.now(),
            "experiment": config["experiment"], "theme": str(config.get("theme", "")),
            "dataset": dataset, "model": model_name,
            "imbalance": _imb_label(imb), "seed": seed, "split": split_name,
            "train_seconds": round(train_seconds, 1),
            "config": config_path, "git_sha": harness.git_sha(),
            **m.as_row(),
        }
        harness.append_result(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _imb_label(imb: dict) -> str:
    if imb["method"] == "focal":
        return f"focal_g{imb.get('gamma', 2.0):g}"
    return imb["method"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one or more experiment configs.")
    ap.add_argument("configs", nargs="+", help="config yaml path(s) or glob(s)")
    args = ap.parse_args(argv)

    files: list[str] = []
    for pattern in args.configs:
        hits = sorted(glob.glob(pattern))
        files.extend(hits if hits else [pattern])

    failed = []
    for f in files:
        p = Path(f).resolve()
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        try:
            run_experiment(cfg, config_path=p.relative_to(paths.ROOT).as_posix())
        except Exception as exc:  # a bad config must not kill the batch
            print(f"  !! FAILED {f}: {type(exc).__name__}: {exc}", flush=True)
            failed.append(f)
    if failed:
        print(f"\n{len(failed)} config(s) failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
