"""Emit one committed YAML config per experiment (R5).

    python -m src.make_configs

Theme 1.5 grid: {logreg, xgboost, mlp} x {plain, class-weighted, focal g1/2/5, SMOTE}
x {uci, ieee, paysim}. Focal loss only applies to the MLP; class weighting for XGBoost is
`scale_pos_weight`, for logreg `class_weight='balanced'`.
"""
from __future__ import annotations

import yaml

from . import paths

SEED = 42

# Per-dataset model sizing. Larger datasets get bigger batches and fewer epochs so that
# every cell of the grid finishes in a comparable wall-clock budget.
MLP_CFG = {
    "uci":    {"hidden": [128, 64],  "batch_size": 512,  "epochs": 40, "patience": 6},
    "ieee":   {"hidden": [256, 128], "batch_size": 1024, "epochs": 30, "patience": 5},
    "paysim": {"hidden": [256, 128], "batch_size": 4096, "epochs": 20, "patience": 4},
}
XGB_CFG = {
    "uci":    {"n_estimators": 600, "max_depth": 5, "learning_rate": 0.05},
    "ieee":   {"n_estimators": 800, "max_depth": 8, "learning_rate": 0.05},
    "paysim": {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.10},
}

IMBALANCE_VARIANTS = [
    ("plain",        {"method": "none"}),
    ("weighted",     {"method": "class_weight"}),
    ("smote",        {"method": "smote"}),
]
FOCAL_VARIANTS = [(f"focal_g{g:g}", {"method": "focal", "gamma": g, "alpha": 0.25})
                  for g in (1, 2, 5)]


def write(cfg: dict) -> str:
    path = paths.CONFIGS / f"{cfg['experiment']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path.name


def main() -> int:
    written = []

    # --- Step 0: the frozen conventional baselines (Pipeline stage 4) ----------------
    for ds in ("uci", "ieee", "paysim"):
        written.append(write({
            "experiment": f"step0_{ds}_xgb_baseline",
            "theme": "0",
            "dataset": ds,
            "seed": SEED,
            "model": {"name": "xgboost", "params": XGB_CFG[ds]},
            "imbalance": {"method": "none"},
            "notes": "Frozen conventional baseline. No imbalance or sequence handling.",
        }))
        written.append(write({
            "experiment": f"step0_{ds}_logreg_baseline",
            "theme": "0",
            "dataset": ds,
            "seed": SEED,
            "model": {"name": "logreg", "params": {"max_iter": 1000}},
            "imbalance": {"method": "none"},
            "notes": "Linear reference point.",
        }))

    # --- Theme 1.5 grid --------------------------------------------------------------
    for ds in ("uci", "ieee", "paysim"):
        for model in ("logreg", "xgboost", "mlp"):
            variants = list(IMBALANCE_VARIANTS)
            if model == "mlp":
                variants += FOCAL_VARIANTS
            if model == "logreg":
                variants = [v for v in variants if v[0] != "smote"]  # covered by weighted
            for tag, imb in variants:
                mcfg: dict = {"name": model}
                if model == "xgboost":
                    mcfg["params"] = XGB_CFG[ds]
                elif model == "mlp":
                    mcfg.update(MLP_CFG[ds])
                else:
                    mcfg["params"] = {"max_iter": 1000}
                written.append(write({
                    "experiment": f"t15_{ds}_{model}_{tag}",
                    "theme": "1.5",
                    "dataset": ds,
                    "seed": SEED,
                    "model": mcfg,
                    "imbalance": imb,
                    "missing_indicators": False,
                }))

    print(f"wrote {len(written)} configs -> {paths.CONFIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
