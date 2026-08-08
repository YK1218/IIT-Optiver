# Step 0 — Setup, datasets, splits, harness

**Hand-in:** working environment, processed datasets, frozen splits, harness verified
against the UCI number.
**Status: complete.** UCI baseline reproduced through the harness at **test AUROC 0.7774**
against the required ≈ 0.774.

---

## 1. Environment

Python 3.12 venv at `.venv/`, created with `uv`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe numpy pandas pyarrow scikit-learn xgboost imbalanced-learn matplotlib xlrd openpyxl pyyaml tqdm tabulate
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch
```

Key versions: pandas 2.x, scikit-learn 1.8, xgboost 3.4.0, torch (CPU build), imbalanced-learn.

## 2. Repository layout

```
src/
  paths.py            canonical paths; raw data is read-only
  prepare.py          Step 0 driver: raw -> parquet -> splits -> audits -> sequences
  make_configs.py     emits one committed YAML per experiment (R5)
  run.py              run_experiment(config); appends to results.csv
  report.py           builds the Theme 1.5 table + headline figures from results.csv
  data/loaders.py     one loader per dataset -> label column `y`
  data/splits.py      frozen 70/15/15 stratified splits, seed 42
  data/features.py    TreeEncoder / DenseEncoder, both fitted on the train fold only
  models/mlp.py       MLP + FocalLoss + weighted BCE
  eval/harness.py     evaluate(), tune_threshold(), append_result()
configs/              40 experiment configs
data/processed/       uci.parquet, ieee.parquet, paysim.parquet
data/splits/          <dataset>.npz (+ <dataset>_grouped.npz for themes 1.1/1.2)
data/sequences/       <dataset>_events.parquet — ordered events with a Δt channel
results/results.csv   the single source of truth for every number
```

Reproduce anything in `results.csv` with one command (R5):

```bash
.venv/Scripts/python.exe -m src.run configs/<experiment>.yaml
```

## 3. Datasets processed

| | UCI Credit Default | IEEE-CIS Fraud | PaySim |
|---|---|---|---|
| rows | 30,000 | 590,540 | 1,048,575 |
| features | 23 | 431 | 13 |
| positive rate | 22.12% | 3.499% | 0.109% |
| imbalance ratio | 3.5 : 1 | 27.6 : 1 | 917 : 1 |
| entity key | account id | `card1_addr1_(day − D1)` | `nameDest` |
| matches the guide? | yes (22.1%) | yes (590K, 3.5%) | **no — see §6** |

IEEE-CIS is `train_transaction` left-joined to `train_identity`; identity is present for
only ~24% of rows, and that whole-channel absence is the raw material for Theme 1.3.

## 4. Frozen splits (R2)

70/15/15, stratified on the label, `random_state=42`, written once to
`data/splits/<name>.npz`. `splits.make()` refuses to overwrite an existing file unless
explicitly forced — a silent re-split would invalidate every row already in `results.csv`.

| dataset | train | val | test | val pos-rate | test pos-rate |
|---|---|---|---|---|---|
| uci | 21,000 | 4,500 | 4,500 | 22.11% | 22.13% |
| ieee | 413,378 | 88,581 | 88,581 | 3.500% | 3.499% |
| paysim | 734,002 | 157,286 | 157,287 | 0.109% | 0.109% |

A second, **entity-disjoint** split (`<name>_grouped.npz`) is written for the sequence
themes so that a Theme 1.1 model cannot see the same customer in train and test. The
mandated stratified split remains the one every reported number uses.

## 5. Audits (Pipeline stage 2)

Full table in `reports/audits.md`. The three audits and what they set up:

- **Imbalance → Theme 1.5.** 917:1 on PaySim, 27.6:1 on IEEE-CIS. A "never fraud"
  classifier is 99.89% accurate on PaySim; this is why R4 bans accuracy.
- **Missingness → Theme 1.3.** IEEE-CIS: 414 of 431 columns have missing values, 214 are
  over 50% missing, mean missing rate 45.4%. UCI and PaySim have none. IEEE-CIS is the
  only viable Theme 1.3 dataset in the current pool (MIMIC-III is not yet downloaded).
- **Sparsity → Theme 1.1.** IEEE-CIS: median 1 event per entity, p95 = 9, median inter-event
  gap 2.63e5 s (≈ 3.0 days), p95 gap 3.97e6 s (≈ 46 days). Genuinely irregular — good.

## 6. Two data problems found — both need a decision

### 6a. The PaySim file in `IITB_01_datasets/` is truncated

`PS_20174392719_1491204439457_log.csv` contains exactly **1,048,575 data rows** — that
is 2²⁰ − 1, the Excel worksheet row limit. The file covers simulation **steps 1–95**;
full PaySim is **6,362,620 rows over steps 1–744**. It has been round-tripped through
Excel at some point and silently cut at ~16% of its length.

Consequences: the positive rate reads 0.109% instead of the documented 0.13%, and the
time axis is ~4 simulated days instead of ~31. The Theme 1.5 result below is still valid
(917:1 is still extreme skew), but **Theme 1.1 needs the full file** — 4 days of history
cannot demonstrate dormancy.

**Action: re-download PaySim from Kaggle before Theme 1.1 starts (weeks 3–6).**

### 6b. PaySim's `nameOrig` cannot be the sequence entity key

| key | entities | median events | p95 | max | single-event |
|---|---|---|---|---|---|
| `nameOrig` | 1,048,317 | 1 | 1 | 2 | **99.98%** |
| `nameDest` | 449,635 | 1 | 10 | 98 | 82.6% |

PaySim generates a near-unique origin account per transaction, so keying customer
sequences on `nameOrig` yields sequences of length 1 — Theme 1.1's "last N=100 events"
would be empty. This is a property of the simulator, not of the truncation.

**Action taken:** `entity_id` for PaySim is set to `nameDest`. Theme 1.1 should model the
receiving account's event stream. Flag at the weekly meeting — if the lab wants
origin-side sequences, PaySim is the wrong dataset for 1.1 and IEEE-CIS should carry it.

## 7. Harness (R1, R4)

`src/eval/harness.py`:

- `evaluate(y_true, y_scores, threshold)` → AUROC, AUPRC, F1, recall, precision,
  plus `recall_at_p50` (recall while holding precision ≥ 0.50) and `precision_at_r50`.
  **Accuracy is deliberately not returned.**
- `tune_threshold(y_true, y_scores)` → the F1-maximising threshold, called on
  validation only; the resulting threshold is applied unchanged to test.
- `run_experiment(config)` in `src/run.py` trains, evaluates and appends one row per
  split to `results.csv`, tagged with the config path and the git SHA.

### Verification against the UCI number

| model | split | AUROC | AUPRC | F1 | recall |
|---|---|---|---|---|---|
| XGBoost (frozen baseline) | val | 0.7764 | 0.5535 | 0.5356 | 0.6050 |
| **XGBoost (frozen baseline)** | **test** | **0.7774** | 0.5521 | 0.5376 | 0.6175 |
| Logistic regression | test | 0.7197 | 0.4961 | 0.5045 | 0.4819 |

Target was AUROC ≈ 0.774; reproduced at **0.7774** (+0.003). The harness is verified.

## 8. Sequence view (Step 0.5)

`data/sequences/<name>_events.parquet` holds `row_id, entity_id, event_time, dt_prev,
log_dt_prev, event_rank` — per entity, events in time order, each carrying its gap from
the previous event, log-scaled as the guide specifies. It joins back onto the feature
table on `row_id`, so no feature is duplicated. This is the input Theme 1.1 consumes.

## 9. Leakage handling

- PaySim: `isFlaggedFraud` (the simulator's own rule-based alarm) is dropped;
  `nameOrig`/`nameDest` are used as keys only, never as features.
- IEEE-CIS: `TransactionID` dropped; `TransactionDT` kept only as the ordering key.
- All encoders, imputers, scalers and resamplers are fitted on the training fold and
  applied to val/test (R3). `DenseEncoder` maps categories unseen at fit time to 0
  rather than extending its vocabulary.
- One open question, tested in Theme 1.5: PaySim's post-transaction balance columns.
  See `configs/t15_paysim_xgboost_weighted_noleak.yaml`.
