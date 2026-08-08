"""The shared evaluation harness. Every reported number comes through here (R1).

`evaluate` deliberately does not return accuracy (R4): on PaySim, "never fraud" scores
99.87% accurate and catches nothing.
"""
from __future__ import annotations

import csv
import datetime as dt
import subprocess
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .. import paths

RESULT_COLUMNS = [
    "run_id", "timestamp", "experiment", "theme", "dataset", "model", "imbalance",
    "seed", "split", "n", "n_pos", "pos_rate",
    "auroc", "auprc", "f1", "recall", "precision", "threshold",
    "recall_at_p50", "precision_at_r50", "train_seconds", "config", "git_sha",
]


@dataclass
class Metrics:
    auroc: float
    auprc: float
    f1: float
    recall: float
    precision: float
    threshold: float
    n: int
    n_pos: int
    recall_at_p50: float = float("nan")
    precision_at_r50: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d["pos_rate"] = self.n_pos / self.n if self.n else float("nan")
        return d


def evaluate(y_true, y_scores, threshold: float = 0.5) -> Metrics:
    """AUROC, AUPRC, F1, recall (positive class) at a given decision threshold.

    AUROC/AUPRC are threshold-free; F1/recall/precision are read at `threshold`, which
    the caller is expected to have tuned on validation, never on test.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    y_pred = (y_scores >= threshold).astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return Metrics(
        auroc=float(roc_auc_score(y_true, y_scores)),
        auprc=float(average_precision_score(y_true, y_scores)),
        f1=float(f1),
        recall=float(rec),
        precision=float(prec),
        threshold=float(threshold),
        n=int(len(y_true)),
        n_pos=int(y_true.sum()),
        recall_at_p50=_recall_at_precision(y_true, y_scores, 0.50),
        precision_at_r50=_precision_at_recall(y_true, y_scores, 0.50),
    )


def tune_threshold(y_true, y_scores) -> tuple[float, float]:
    """Threshold that maximises F1 on the supplied (validation) fold. Returns (thr, f1)."""
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    denom = prec[:-1] + rec[:-1]
    f1 = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / np.where(denom > 0, denom, 1), 0.0)
    if len(f1) == 0:
        return 0.5, 0.0
    best = int(np.nanargmax(f1))
    return float(thr[best]), float(f1[best])


def _recall_at_precision(y_true, y_scores, target: float) -> float:
    """Operating-point view the lab actually cares about: recall at fixed precision."""
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


def _precision_at_recall(y_true, y_scores, target: float) -> float:
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    ok = rec >= target
    return float(prec[ok].max()) if ok.any() else 0.0


# --------------------------------------------------------------------- results.csv
def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "nogit"


def append_result(row: dict) -> None:
    """One row per (experiment, split). results.csv is the single source of truth."""
    path = paths.RESULTS_CSV
    new = not path.exists()
    full = {c: row.get(c, "") for c in RESULT_COLUMNS}
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(full)


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")
