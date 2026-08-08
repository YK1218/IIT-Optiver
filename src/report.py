"""Build the Theme 1.5 deliverable from results.csv: comparison table + headline figure.

    python -m src.report

Everything here reads results.csv only -- no model is re-run and no metric is
recomputed by hand (R1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths

# Categorical slots 1-3 of the lab palette; validated all-pairs (worst CVD dE 9.2,
# normal-vision dE 24.0) so the three model families stay distinguishable under CVD.
MODEL_COLOR = {"logreg": "#2a78d6", "xgboost": "#eb6834", "mlp": "#1baf7a"}
MODEL_LABEL = {"logreg": "Logistic regression", "xgboost": "XGBoost", "mlp": "MLP"}

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BASELINE_RULE = "#c3c2b7"

VARIANT_ORDER = ["plain", "class_weight", "smote", "focal_g1", "focal_g2", "focal_g5"]
VARIANT_LABEL = {
    "plain": "plain BCE",
    "class_weight": "class-weighted",
    "smote": "SMOTE",
    "focal_g1": "focal γ=1",
    "focal_g2": "focal γ=2",
    "focal_g5": "focal γ=5",
}
DATASET_LABEL = {
    "uci": "UCI Credit Default\n22.1% positive",
    "ieee": "IEEE-CIS Fraud\n3.50% positive",
    "paysim": "PaySim\n0.11% positive",
}
METRICS = ["auroc", "auprc", "f1", "recall", "precision", "recall_at_p50"]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(paths.RESULTS_CSV)
    df["imbalance"] = df["imbalance"].replace({"none": "plain"})
    # Keep the newest run per (experiment, split) so re-runs supersede rather than stack.
    df = df.sort_values("timestamp").drop_duplicates(["experiment", "split"], keep="last")
    return df


def comparison_table(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    t = df[(df["theme"].astype(str) == "1.5") & (df["split"] == split)].copy()
    t["variant"] = pd.Categorical(t["imbalance"], VARIANT_ORDER, ordered=True)
    t["dataset"] = pd.Categorical(t["dataset"], ["uci", "ieee", "paysim"], ordered=True)
    t = t.sort_values(["dataset", "model", "variant"])
    cols = ["dataset", "model", "imbalance", *METRICS, "threshold"]
    return t[cols].reset_index(drop=True)


def delta_vs_plain(tbl: pd.DataFrame) -> pd.DataFrame:
    """Every variant expressed as a change from that model's own plain-BCE run.

    This is the comparison the theme actually asks for: does the imbalance recipe beat
    doing nothing, holding the architecture fixed?
    """
    base = tbl.loc[tbl["imbalance"] == "plain", ["dataset", "model", *METRICS]]
    out = tbl.merge(base, on=["dataset", "model"], how="left", suffixes=("", "_base"))
    for m in METRICS:
        out[f"d_{m}"] = out[m] - out[f"{m}_base"]
    return out.drop(columns=[f"{m}_base" for m in METRICS])


def write_tables(tbl: pd.DataFrame, delta: pd.DataFrame) -> None:
    tbl.to_csv(paths.RESULTS / "theme_1_5_comparison.csv", index=False)

    pretty = tbl.copy()
    pretty["method"] = pretty["imbalance"].map(VARIANT_LABEL)
    lines = ["# Theme 1.5 — imbalance-aware training: method × dataset × metrics\n",
             "Test-fold numbers. Decision threshold tuned for F1 on **validation** and "
             "applied unchanged to test. AUROC/AUPRC are threshold-free.\n",
             "`recall_at_p50` = recall achievable while holding precision ≥ 0.50 — the "
             "operating point the pipeline document asks for.\n"]
    for ds in ["uci", "ieee", "paysim"]:
        sub = pretty[pretty["dataset"] == ds]
        if sub.empty:
            continue
        lines.append(f"\n## {ds}\n")
        show = sub[["model", "method", *METRICS, "threshold"]]
        lines.append(show.to_markdown(index=False, floatfmt=".4f"))
    (paths.REPORTS / "theme_1_5_comparison.md").write_text("\n".join(lines) + "\n",
                                                           encoding="utf-8")

    d = delta[["dataset", "model", "imbalance", "d_auprc", "d_recall", "d_f1",
               "d_recall_at_p50"]]
    d.to_csv(paths.RESULTS / "theme_1_5_delta_vs_plain.csv", index=False)


# ------------------------------------------------------------------------- figure
def headline_figure(tbl: pd.DataFrame, metric: str = "auprc") -> None:
    """Dot plot per dataset: every recipe on a common scale, plain BCE marked.

    A dot plot rather than bars -- the differences between recipes are a few points of
    AUPRC, and bars anchored at zero would render them invisible. Dots carry no
    zero-baseline obligation, so each facet is scaled to its own data range.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    datasets = [d for d in ["uci", "ieee", "paysim"] if d in set(tbl["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 5.4),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)

    rows = [(m, v) for m in ["logreg", "xgboost", "mlp"] for v in VARIANT_ORDER]

    for ax, ds in zip(axes, datasets):
        sub = tbl[tbl["dataset"] == ds]
        present = [r for r in rows if not sub[(sub["model"] == r[0])
                                              & (sub["imbalance"] == r[1])].empty]
        ypos = {r: i for i, r in enumerate(reversed(present))}

        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE_RULE)
        ax.spines["bottom"].set_linewidth(1)
        ax.grid(axis="x", color=GRID, linewidth=1, linestyle="-")
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)

        # Reference: the frozen plain-BCE XGBoost baseline every theme must beat.
        b = sub[(sub["model"] == "xgboost") & (sub["imbalance"] == "plain")]
        if not b.empty:
            ax.axvline(float(b[metric].iloc[0]), color=BASELINE_RULE, linewidth=1.5,
                       zorder=1)

        best = sub.loc[sub[metric].idxmax()]
        for (model, variant), y in ypos.items():
            r = sub[(sub["model"] == model) & (sub["imbalance"] == variant)]
            if r.empty:
                continue
            x = float(r[metric].iloc[0])
            ax.plot([x], [y], marker="o", markersize=9,
                    color=MODEL_COLOR[model], markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=3, linestyle="none")

        # Direct-label the winner only; the table view carries every other value.
        bkey = (best["model"], best["imbalance"])
        if bkey in ypos:
            ax.annotate(f"{best[metric]:.3f}",
                        (float(best[metric]), ypos[bkey]),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=9, color=INK, fontweight="bold")

        ax.margins(x=0.10)   # keep edge dots and their labels clear of the axis
        ax.set_yticks(list(ypos.values()))
        ax.set_yticklabels([f"{MODEL_LABEL[m].split()[0].lower()} · {VARIANT_LABEL[v]}"
                            for m, v in ypos.keys()], fontsize=8.5, color=INK)
        ax.set_ylim(-0.8, len(ypos) - 0.2)
        ax.set_title(DATASET_LABEL[ds], fontsize=11, color=INK, pad=12, loc="left")
        ax.set_xlabel(metric.upper().replace("_", " "), fontsize=9, color=MUTED)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=9,
                      color=MODEL_COLOR[m], markeredgecolor=SURFACE, markeredgewidth=2,
                      label=MODEL_LABEL[m]) for m in ["logreg", "xgboost", "mlp"]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 1.0),
               ncol=3, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.suptitle(
        f"Theme 1.5 — {metric.upper()} on the held-out test fold, by training recipe",
        fontsize=13, color=INK, x=0.005, ha="left", y=1.055,
    )
    fig.text(0.005, -0.035,
             "Grey rule = frozen plain-BCE XGBoost baseline.  Threshold tuned on "
             "validation only.  Facets use independent x-scales.",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout()
    out = paths.FIGURES / f"theme_1_5_{metric}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {out}")


def winners(tbl: pd.DataFrame) -> pd.DataFrame:
    """Best recipe per (dataset, model) on AUPRC -- the input to declaring the standard loss."""
    i = tbl.groupby(["dataset", "model"], observed=True)["auprc"].idxmax()
    return tbl.loc[i, ["dataset", "model", "imbalance", "auprc", "recall",
                       "recall_at_p50", "f1"]].reset_index(drop=True)


def main() -> int:
    df = load_results()
    tbl = comparison_table(df, "test")
    if tbl.empty:
        print("no theme 1.5 rows in results.csv yet")
        return 1
    delta = delta_vs_plain(tbl)
    write_tables(tbl, delta)
    for m in ("auprc", "recall_at_p50"):
        headline_figure(tbl, m)
    w = winners(tbl)
    w.to_csv(paths.RESULTS / "theme_1_5_winners.csv", index=False)
    print("\nBest recipe per (dataset, model) by test AUPRC:")
    print(w.to_string(index=False))
    print(f"\ntables -> {paths.REPORTS / 'theme_1_5_comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
