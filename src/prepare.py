"""Step 0: raw -> processed parquet -> frozen splits -> audits -> sequence view.

    python -m src.prepare              # all datasets
    python -m src.prepare uci ieee     # a subset

Runs the three audits the pipeline names as the definition of Themes 1.3 / 1.5 / 1.1:
missingness, imbalance, sparsity. Output lands in reports/audits.md.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import paths
from .data import loaders, splits

ALL = ["uci", "ieee", "paysim"]


def prepare(name: str, force: bool = False) -> dict:
    print(f"\n--- {name} ---", flush=True)
    df = loaders.build(name, force=force)
    print(f"  rows={len(df):,}  cols={df.shape[1]}  "
          f"pos={int(df['y'].sum()):,} ({df['y'].mean():.4%})", flush=True)

    y = df["y"].to_numpy()
    idx = splits.make(name, y, force=force)
    for k in ("train", "val", "test"):
        s = idx[k]
        print(f"  {k:<5} n={len(s):>9,}  pos_rate={y[s].mean():.4%}", flush=True)

    # Group-aware companion split for the sequence themes (1.1 / 1.2): whole customers
    # are kept in one fold so a sequence model cannot memorise an entity across folds.
    if loaders.DATASETS[name]["temporal"]:
        splits.make(f"{name}_grouped", y, groups=df["entity_id"].to_numpy(), force=force)
        print("  + grouped split written (entity-disjoint, for themes 1.1/1.2)", flush=True)

    return audit(name, df, idx)


# ------------------------------------------------------------------------- audits
def audit(name: str, df: pd.DataFrame, idx: dict) -> dict:
    feat = loaders.feature_columns(df)
    miss = df[feat].isna().mean()
    counts = df.groupby("entity_id", observed=True).size()

    a = {
        "dataset": name,
        "rows": len(df),
        "features": len(feat),
        # imbalance audit -> Theme 1.5
        "pos_rate": float(df["y"].mean()),
        "imbalance_ratio": float((1 - df["y"].mean()) / max(df["y"].mean(), 1e-12)),
        # missingness audit -> Theme 1.3
        "cols_any_missing": int((miss > 0).sum()),
        "cols_over_50pct_missing": int((miss > 0.50).sum()),
        "mean_missing_rate": float(miss.mean()),
        # sparsity audit -> Theme 1.1
        "entities": int(counts.size),
        "median_events_per_entity": float(counts.median()),
        "p95_events_per_entity": float(counts.quantile(0.95)),
        "single_event_entities_pct": float((counts == 1).mean()),
    }

    if loaders.DATASETS[name]["temporal"]:
        gaps = _time_gaps(df)
        a["median_gap"] = float(np.nanmedian(gaps)) if len(gaps) else float("nan")
        a["p95_gap"] = float(np.nanpercentile(gaps, 95)) if len(gaps) else float("nan")

    for k in ("train", "val", "test"):
        a[f"n_{k}"] = int(len(idx[k]))
    return a


def _time_gaps(df: pd.DataFrame) -> np.ndarray:
    """Delta-t between consecutive events of the same entity -- the Theme 1.1 channel."""
    s = df[["entity_id", "event_time"]].sort_values(["entity_id", "event_time"])
    g = s.groupby("entity_id", observed=True)["event_time"].diff()
    return g.dropna().to_numpy()


def build_sequences(name: str) -> None:
    """Step 0.5: per-entity events in time order, each carrying its gap from the previous.

    Saved as a flat parquet (row_id, entity_id, event_time, dt_prev, event_rank) that the
    Theme 1.1 loader joins back onto the feature table -- no feature duplication.
    """
    df = loaders.load(name)[["row_id", "entity_id", "event_time"]]
    df = df.sort_values(["entity_id", "event_time"], kind="mergesort")
    g = df.groupby("entity_id", observed=True)
    df["dt_prev"] = g["event_time"].diff()
    df["event_rank"] = g.cumcount()
    df["log_dt_prev"] = np.log1p(df["dt_prev"])  # log-scaled, as the guide specifies
    out = paths.SEQUENCES / f"{name}_events.parquet"
    df.to_parquet(out, index=False)
    print(f"  sequence view -> {out.name}  "
          f"(median dt={df['dt_prev'].median():.3g}, max rank={int(df['event_rank'].max())})",
          flush=True)


def write_audits(rows: list[dict]) -> None:
    t = pd.DataFrame(rows).set_index("dataset").T
    md = ["# Step 0 — data audits\n",
          "Three audits define the themes: **missingness → 1.3**, **imbalance → 1.5**, "
          "**sparsity → 1.1**.\n",
          t.to_markdown(floatfmt=".4g"), ""]
    (paths.REPORTS / "audits.md").write_text("\n".join(md), encoding="utf-8")
    t.to_csv(paths.REPORTS / "audits.csv")
    print(f"\naudits -> {paths.REPORTS / 'audits.md'}", flush=True)


def main(argv=None):
    names = argv or sys.argv[1:] or ALL
    rows = [prepare(n) for n in names]
    for n in names:
        if loaders.DATASETS[n]["temporal"]:
            build_sequences(n)
    write_audits(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
