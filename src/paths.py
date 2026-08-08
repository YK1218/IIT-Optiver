"""Canonical project paths. Raw data is read-only (Intern Guide, Step 0.2)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "IITB_01_datasets"          # immutable copy of the raw datasets
PROCESSED = ROOT / "data" / "processed"  # clean parquet, one label column `y`
SPLITS = ROOT / "data" / "splits"        # frozen 70/15/15 stratified indices, seed 42
SEQUENCES = ROOT / "data" / "sequences"  # per-entity event sequences with a dt channel
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RESULTS_CSV = RESULTS / "results.csv"

RAW_UCI = RAW / "UCI Credit Card Default Dataset" / "default of credit card clients.xls"
RAW_IEEE_TRANSACTION = RAW / "IEEE-CIS Fraud Detection" / "train_transaction.csv"
RAW_IEEE_IDENTITY = RAW / "IEEE-CIS Fraud Detection" / "train_identity.csv"
RAW_PAYSIM = RAW / "PaySim Synthetic Financial Dataset" / "PS_20174392719_1491204439457_log.csv"

for _d in (PROCESSED, SPLITS, SEQUENCES, MODELS, RESULTS, REPORTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
