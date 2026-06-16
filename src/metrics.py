from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class BinaryMetrics:
    acc: float
    auc: float
    f1: float
    precision: float
    recall: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "acc": self.acc,
            "auc": self.auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_binary_metrics(labels: list[int] | np.ndarray, probs: list[float] | np.ndarray) -> BinaryMetrics:
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs).astype(float)
    pred = (p >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    return BinaryMetrics(
        acc=float(accuracy_score(y, pred)),
        auc=auc,
        f1=float(f1_score(y, pred, zero_division=0)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).astype(int).tolist(),
    )


def logits_to_fake_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]
