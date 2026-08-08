"""Dataset loaders: raw files -> clean parquet in data/processed/ with a label column `y`.

Intern Guide Step 0.3. Raw files are never modified. Every loader returns a frame whose
columns are: `y` (int8 label), `entity_id` (string, the entity key), `event_time` (numeric,
for temporal ordering) and the feature columns. Column roles are declared in DATASETS so
the rest of the pipeline never has to hard-code dataset specifics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths

# Columns that are never features: identifiers, the label, and the ordering key.
NON_FEATURE = ("y", "entity_id", "event_time", "row_id")


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    """Common tail-end: stable row id, label as int8, features downcast to save memory."""
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    df["y"] = df["y"].astype(np.int8)
    for c in df.columns:
        if c in NON_FEATURE:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(np.float32)  # keeps NaN representable downstream
    return df


# --------------------------------------------------------------------------- UCI
def load_uci() -> pd.DataFrame:
    """UCI Credit Card Default: 30K accounts, ~22% positive. Pipeline proving ground.

    The .xls has a two-row header (X1..X23 then the real names); the real names are row 2.
    """
    df = pd.read_excel(paths.RAW_UCI, header=1)
    df = df.rename(columns={"default payment next month": "y", "PAY_0": "PAY_1"})
    df["entity_id"] = "acct_" + df["ID"].astype(str)
    df["event_time"] = 0.0  # cross-sectional: one row per account, no time axis
    df = df.drop(columns=["ID"])
    return _finish(df)


# ------------------------------------------------------------------------ IEEE-CIS
def load_ieee() -> pd.DataFrame:
    """IEEE-CIS Fraud: ~590K transactions, ~3.5% fraud, heavy structured missingness.

    train_transaction is left-joined with train_identity (identity is present for only
    ~24% of rows -- that whole-channel absence is the raw material for Theme 1.3).
    """
    trans = pd.read_csv(paths.RAW_IEEE_TRANSACTION)
    ident = pd.read_csv(paths.RAW_IEEE_IDENTITY)
    df = trans.merge(ident, on="TransactionID", how="left")
    del trans, ident

    df = df.rename(columns={"isFraud": "y"})

    # Entity key. IEEE-CIS ships no customer id; the community-standard proxy is
    # card1 + addr1 + (TransactionDay - D1), which pins a card to an account-open date.
    day = df["TransactionDT"] / (24 * 60 * 60)
    uid = (
        df["card1"].astype("Int64").astype(str)
        + "_" + df["addr1"].astype("Int64").astype(str)
        + "_" + np.floor(day - df["D1"].fillna(-1)).astype("Int64").astype(str)
    )
    df["entity_id"] = uid
    df["event_time"] = df["TransactionDT"].astype(np.float64)  # seconds from a fixed origin
    df = df.drop(columns=["TransactionID", "TransactionDT"])
    return _finish(df)


# -------------------------------------------------------------------------- PaySim
def load_paysim() -> pd.DataFrame:
    """PaySim: 6.36M simulated mobile-money transactions, ~0.13% fraud.

    Leakage removal (Pipeline stage 2):
      * `isFlaggedFraud` is the simulator's own rule-based alarm, not an observable
        feature -- dropped.
      * `nameOrig`/`nameDest` are raw account ids; kept only as the entity key and as a
        derived `dest_is_merchant` flag, never as model features.
    """
    df = pd.read_csv(paths.RAW_PAYSIM)
    df = df.rename(columns={"isFraud": "y"})

    # Entity key is the DESTINATION account, not the origin. PaySim mints a near-unique
    # nameOrig per transaction (99.98% of origins appear exactly once), so origin-keyed
    # sequences have length 1 and Theme 1.1 would have nothing to attend over. nameDest
    # gives 449,635 entities with p95 = 10 events. See reports/step0_setup.md §6b.
    df["entity_id"] = df["nameDest"]
    df["event_time"] = df["step"].astype(np.float64)  # simulation hour

    # Balance-consistency features: PaySim's fraud pattern drains an account, so the
    # residual between declared and implied balances is the strongest simple signal.
    df["err_balance_orig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["err_balance_dest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["dest_is_merchant"] = (df["nameDest"].str[0] == "M").astype(np.int8)
    df["hour_of_day"] = df["step"] % 24
    df["day"] = df["step"] // 24
    df["amount_to_balance"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)

    df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])
    return _finish(df)


# ------------------------------------------------------------------------- registry
DATASETS = {
    "uci": {
        "loader": load_uci,
        "label_rate": 0.221,
        "categorical": ["SEX", "EDUCATION", "MARRIAGE"],
        "temporal": False,
        "smote_strategy": "auto",   # 22% positive -> full rebalance is sane
    },
    "ieee": {
        "loader": load_ieee,
        "label_rate": 0.035,
        "categorical": [
            "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "P_emaildomain", "R_emaildomain",
            "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
            "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19",
            "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27",
            "id_28", "id_29", "id_30", "id_31", "id_32", "id_33", "id_34", "id_35",
            "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo",
        ],
        "temporal": True,
        "smote_strategy": 0.1,      # full rebalance would triple the matrix
    },
    "paysim": {
        "loader": load_paysim,
        "label_rate": 0.0013,
        "categorical": ["type"],
        "temporal": True,
        "smote_strategy": 0.1,      # 0.13% positive -> 1:1 would synthesise 4.4M rows
    },
}


def processed_path(name: str):
    return paths.PROCESSED / f"{name}.parquet"


def build(name: str, force: bool = False) -> pd.DataFrame:
    """Materialise data/processed/<name>.parquet, or read the cached copy."""
    out = processed_path(name)
    if out.exists() and not force:
        return pd.read_parquet(out)
    df = DATASETS[name]["loader"]()
    df.to_parquet(out, index=False)
    return df


def load(name: str) -> pd.DataFrame:
    out = processed_path(name)
    if not out.exists():
        raise FileNotFoundError(f"{out} missing -- run `python -m src.prepare` first.")
    return pd.read_parquet(out)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE]
