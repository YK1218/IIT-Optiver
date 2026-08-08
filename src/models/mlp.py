"""Small MLP classifier with pluggable imbalance handling (Theme 1.5).

Three training recipes share one architecture so the comparison isolates the loss:
  * `bce`            -- plain binary cross-entropy
  * `weighted_bce`   -- cost-sensitive: positives weighted by N_neg/N_pos
  * `focal`          -- focal loss, alpha-balanced, gamma in {1, 2, 5}
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Lin et al. 2017, alpha-balanced, computed on logits for numerical stability.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma down-weights easy negatives so the gradient is dominated by the hard, rare class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t).pow(self.gamma) * ce).mean()


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden=(256, 128), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        d = n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier:
    """sklearn-shaped wrapper: fit(X, y, X_val, y_val) / predict_proba(X)."""

    def __init__(
        self,
        loss: str = "bce",
        gamma: float = 2.0,
        alpha: float = 0.25,
        hidden=(256, 128),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        epochs: int = 30,
        patience: int = 5,
        seed: int = 42,
        threads: int = 0,
        verbose: bool = True,
    ):
        self.loss, self.gamma, self.alpha = loss, gamma, alpha
        self.hidden, self.dropout = tuple(hidden), dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.seed, self.verbose = seed, verbose
        if threads:
            torch.set_num_threads(threads)
        self.model_: MLP | None = None
        self.history_: list[dict] = []

    def _criterion(self, y_train: np.ndarray):
        if self.loss == "focal":
            return FocalLoss(gamma=self.gamma, alpha=self.alpha)
        if self.loss == "weighted_bce":
            n_pos = max(int(y_train.sum()), 1)
            pos_weight = torch.tensor([(len(y_train) - n_pos) / n_pos], dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss()

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.metrics import average_precision_score

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.model_ = MLP(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        crit = self._criterion(y.numpy())

        n = len(X)
        best_auprc, best_state, stale = -1.0, None, 0
        g = torch.Generator().manual_seed(self.seed)

        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                if len(idx) < 2:      # BatchNorm needs >1 sample
                    continue
                opt.zero_grad(set_to_none=True)
                loss = crit(self.model_(X[idx]), y[idx])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)

            row = {"epoch": epoch, "train_loss": total / n}
            if X_val is not None:
                # Early stopping on validation AUPRC -- the metric that actually moves
                # under extreme skew; AUROC saturates and hides differences.
                s = self.predict_proba(X_val)
                row["val_auprc"] = float(average_precision_score(y_val, s))
                if row["val_auprc"] > best_auprc + 1e-5:
                    best_auprc, stale = row["val_auprc"], 0
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                else:
                    stale += 1
            self.history_.append(row)
            if self.verbose:
                print(f"    epoch {epoch:>2}  loss={row['train_loss']:.5f}"
                      + (f"  val_auprc={row.get('val_auprc', float('nan')):.4f}" if X_val is not None else ""),
                      flush=True)
            if stale >= self.patience:
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        self.model_.eval()
        X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        out = [torch.sigmoid(self.model_(X[i : i + 8192])).numpy()
               for i in range(0, len(X), 8192)]
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)
