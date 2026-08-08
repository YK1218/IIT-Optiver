"""Frozen 70/15/15 stratified splits, seed 42 (Intern Guide R2 + Step 0.4).

Written once to data/splits/<dataset>.npz and never regenerated: `make()` refuses to
overwrite an existing file unless explicitly forced, because a silent re-split
invalidates every number already in results.csv.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from .. import paths

SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)


def split_path(name: str):
    return paths.SPLITS / f"{name}.npz"


def make(name: str, y: np.ndarray, groups: np.ndarray | None = None, force: bool = False) -> dict:
    """Create and freeze the split for `name`.

    Stratified on the label, as mandated. When `groups` (entity ids) is supplied the
    split is additionally group-aware: all events of one customer land in a single fold,
    so a sequence model in Theme 1.1 cannot see the same customer in train and test.
    """
    out = split_path(name)
    if out.exists() and not force:
        return load(name)

    idx = np.arange(len(y))
    if groups is None:
        tr, tmp = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y)
        va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=y[tmp])
    else:
        tr, va, te = _grouped_stratified(idx, y, groups)

    np.savez_compressed(out, train=tr, val=va, test=te, seed=SEED)
    return {"train": tr, "val": va, "test": te}


def _grouped_stratified(idx, y, groups):
    """Split whole entities, stratified by whether the entity ever has a positive event."""
    codes, uniq_pos = _entity_table(y, groups)
    ents = np.arange(len(uniq_pos))
    e_tr, e_tmp = train_test_split(ents, test_size=0.30, random_state=SEED, stratify=uniq_pos)
    e_va, e_te = train_test_split(
        e_tmp, test_size=0.50, random_state=SEED, stratify=uniq_pos[e_tmp]
    )
    member = np.full(len(uniq_pos), -1, dtype=np.int8)
    member[e_tr], member[e_va], member[e_te] = 0, 1, 2
    fold = member[codes]
    return idx[fold == 0], idx[fold == 1], idx[fold == 2]


def _entity_table(y, groups):
    codes, _ = _factorize(groups)
    n = codes.max() + 1
    any_pos = np.zeros(n, dtype=np.int8)
    np.maximum.at(any_pos, codes, y.astype(np.int8))
    return codes, any_pos


def _factorize(groups):
    import pandas as pd

    codes, uniques = pd.factorize(groups, sort=False)
    return codes.astype(np.int64), uniques


def load(name: str) -> dict:
    f = split_path(name)
    if not f.exists():
        raise FileNotFoundError(f"{f} missing -- run `python -m src.prepare` first.")
    z = np.load(f)
    return {"train": z["train"], "val": z["val"], "test": z["test"]}
