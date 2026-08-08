# Project 1 — Learning from Sparse, Skewed, or Irregular Data

IITB – Optiver AI Innovation Lab · PI: Prof. Siddhartha Duttagupta

Benchmark codebase for the five research themes. Implementation order follows the
pipeline document: **1.5 → 1.3 → 1.1 → 1.2 → 1.4**, one dataset at a time.

| | status |
|---|---|
| Step 0 — setup, datasets, splits, harness | done — UCI reproduced at AUROC 0.7774 vs 0.774 target (`reports/step0_setup.md`) |
| Theme 1.5 — extreme class imbalance | done — standard loss declared: plain BCE + validation-tuned threshold, SMOTE below ~1% positives (`reports/theme_1_5_summary.md`) |
| Theme 1.3 — missing features | not started |
| Theme 1.1 — sparse event sequences | not started (sequence view is built) |
| Theme 1.2 — few-shot / zero-shot | not started |
| Theme 1.4 — generative reconstruction | not started |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe numpy pandas pyarrow scikit-learn xgboost imbalanced-learn matplotlib xlrd openpyxl pyyaml tqdm tabulate
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch
```

Raw datasets live in `IITB_01_datasets/` and are **read-only**.

## Running

```bash
python -m src.prepare              # raw -> processed parquet -> frozen splits -> audits
python -m src.make_configs         # emit one YAML per experiment
python -m src.run configs/step0_uci_xgb_baseline.yaml
python -m src.run "configs/t15_*.yaml"
python -m src.report               # comparison tables + headline figures
```

Every experiment is one committed config file, and every number in `results/results.csv`
is reproducible from its `config` column with a single `python -m src.run` (R5).

## Ground rules encoded in the code

| rule | where it lives |
|---|---|
| R1 — all metrics from the shared harness | `src/eval/harness.py`; nothing else computes a metric |
| R2 — fixed 70/15/15 splits, seed 42, test untouched | `src/data/splits.py` refuses to overwrite a frozen split; the threshold is tuned on validation only |
| R3 — fit learnables on train only | `src/data/features.py` encoders; SMOTE runs inside `run_experiment` after the split |
| R4 — never report accuracy | `evaluate()` does not return it |
| R5 — one config per experiment | `configs/`, emitted by `src/make_configs.py` |

## Reports

- `reports/step0_setup.md` — Step 0 hand-in, including two data problems found
- `reports/audits.md` — missingness / imbalance / sparsity audits
- `reports/theme_1_5_comparison.md` — method × dataset × metrics table
- `reports/theme_1_5_summary.md` — the declared standard loss
- `reports/figures/` — headline figures
