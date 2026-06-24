"""
Calibration helpers — metrics, temperature scaling, data loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

COMP_DIR = Path("results/comprehensive")

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def nll(probs: np.ndarray, true_labels: np.ndarray) -> float:
    p = probs[np.arange(len(true_labels)), true_labels]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())

def brier(probs: np.ndarray, true_labels: np.ndarray) -> float:
    oh = np.zeros_like(probs)
    oh[np.arange(len(true_labels)), true_labels] = 1.0
    return float(((probs - oh) ** 2).sum(axis=1).mean())

def ece(probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 15) -> float:
    conf    = probs.max(axis=1)
    pred    = probs.argmax(axis=1)
    correct = (pred == true_labels).astype(float)
    val = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        val += mask.sum() * abs(correct[mask].mean() - conf[mask].mean())
    return float(val / len(true_labels))

def accuracy(probs: np.ndarray, true_labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == true_labels).mean())

def all_metrics(probs: np.ndarray, true_labels: np.ndarray) -> dict:
    from sklearn.metrics import f1_score
    pred = probs.argmax(axis=1)
    return dict(
        accuracy  = accuracy(probs, true_labels),
        f1_macro  = float(f1_score(true_labels, pred, average="macro",    zero_division=0)),
        f1_weighted = float(f1_score(true_labels, pred, average="weighted", zero_division=0)),
        nll       = nll(probs, true_labels),
        brier     = brier(probs, true_labels),
        ece       = ece(probs, true_labels),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────

def _ts_probs(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-10, 1.0))
    scaled = logits / T
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

def fit_temperature(probs: np.ndarray, true_labels: np.ndarray) -> float:
    def obj(T):
        return nll(_ts_probs(probs, T), true_labels)
    return float(minimize_scalar(obj, bounds=(0.05, 20.0), method="bounded").x)

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(model: str, dataset: str, pool: str):
    path = COMP_DIR / f"{model}__{dataset}__{pool}.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)

def load_jsonl(model: str, dataset: str, pool: str) -> list[dict]:
    path = COMP_DIR / f"{model}__{dataset}__{pool}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines()]
