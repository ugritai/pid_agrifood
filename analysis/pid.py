"""
Prompt-Induced Dirichlet (PID) — quality-gated epistemic uncertainty.

The ZPE ensemble gives P probability vectors q_p(x) per sample x.
The BMA mean μ(x) = Σ_p w_p^pred · q_p(x) is the point prediction.

We model per-sample uncertainty as:

    θ(x) ~ Dir(α(x))    where  α(x) = κ(x) · μ(x)
    κ(x) = c / D(x)

    D(x) = Σ_p  w_p^agree · KL(q_p(x) ‖ μ(x))    quality-gated disagreement

The two weight vectors use different temperatures:

    w_p^pred  = softmax(s_norm_p / τ)      # standard ZPE-norm weights (accuracy-tuned)
    w_p^agree = softmax(s_norm_p / τ_a)    # agreement weights — sharper to suppress bad prompts

    τ_a < τ   → only top prompts' disagreement counts  (quality-gated)
    τ_a = τ   → recovers plain mutual information
    τ_a → 0   → only the single best prompt defines agreement
    τ_a → ∞   → all prompts count equally (noisy)

Two scalars (c, τ_a) are fitted on a validation split.
Point predictions are identical to ZPE-norm; only confidence/uncertainty changes.

What is evaluated
-----------------
  1. Selective prediction  — accuracy vs coverage curves
                             reject low-κ(x) samples first; compare methods
  2. AUROC                 — of 1/D(x) as binary predictor of correctness
  3. Calibration           — ECE/NLL with κ-adjusted confidence vs ZPE-norm
  4. τ_a ablation          — AUROC across the τ_a grid to find optimal

Requires
--------
  NPZ files must contain `single_probs (P, N, C)` — re-run zpe_full.py
  with --force if your NPZ files pre-date this change.

Usage
-----
    python analysis/pid.py --model siglip-so400m --dataset food101_full --pool 247
    python analysis/pid.py                                 # all combos
    python analysis/pid.py --table                        # summary table
    python analysis/pid.py --val-frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product as iproduct
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

COMP_DIR = Path("results/comprehensive")
OUT_DIR  = Path("results/pid")

ALL_MODELS = [
    "siglip-so400m",
    "clip-vit-large",
]
ALL_DATASETS = ["food101_full", "food11", "agriculture", "beans"]
DATASET_POOLS = {
    "food101_full": ["247", "426", "food", "247+food"],
    "food11":       ["247", "426", "food", "247+food"],
    "agriculture":  ["247", "426", "agri", "247+agri"],
    "beans":        ["247", "426", "agri", "247+agri"],
}

# τ_a grid — from very sharp (quality-gated) to flat (all prompts equal)
TAU_A_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

# ─────────────────────────────────────────────────────────────────────────────
# Core: quality-gated disagreement D(x)
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def compute_bma(
    single_probs: np.ndarray,  # (P, N, C) float16
    scores_norm:  np.ndarray,  # (P,)
    tau_pred:     float,
) -> np.ndarray:               # (N, C) float32
    """BMA mean: probability-averaged (not logit-averaged)."""
    w = _softmax(scores_norm, tau_pred)              # (P,)
    probs = single_probs.astype(np.float32)          # (P, N, C)
    return (probs * w[:, None, None]).sum(0)         # (N, C)


def compute_D(
    single_probs: np.ndarray,  # (P, N, C) float16
    scores_norm:  np.ndarray,  # (P,)
    mu:           np.ndarray,  # (N, C) float32  — BMA mean
    tau_agree:    float,
) -> np.ndarray:               # (N,) — quality-gated disagreement
    """
    D(x) = Σ_p  w_p^agree · KL(q_p(x) ‖ μ(x))

    KL(q ‖ μ) = Σ_c q_c log(q_c / μ_c)
    """
    w_agree = _softmax(scores_norm, tau_agree)       # (P,)
    probs   = np.clip(single_probs.astype(np.float32), 1e-10, 1.0)  # (P, N, C)
    mu_safe = np.clip(mu, 1e-10, 1.0)               # (N, C)

    # KL per prompt per sample: (P, N)
    kl = (probs * np.log(probs / mu_safe[None])).sum(-1)  # (P, N)

    # Quality-weighted sum over prompts: (N,)
    D = (kl * w_agree[:, None]).sum(0)
    return np.clip(D, 1e-10, None)


def kappa(D: np.ndarray, c: float) -> np.ndarray:
    """Dirichlet concentration: κ(x) = c / D(x)."""
    return c / D


# ─────────────────────────────────────────────────────────────────────────────
# Fitting (c, τ_a) on validation split
# ─────────────────────────────────────────────────────────────────────────────

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of `scores` as predictor of binary `labels` (1=correct, 0=wrong)."""
    from sklearn.metrics import roc_auc_score
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fit_tau_a(
    single_probs: np.ndarray,  # (P, N_val, C)
    scores_norm:  np.ndarray,  # (P,)
    true_labels:  np.ndarray,  # (N_val,)
    tau_pred:     float,
    tau_a_grid:   list[float] = TAU_A_GRID,
) -> tuple[float, dict]:
    """
    Find τ_a that maximises AUROC of 1/D(x) as a correctness predictor.
    Returns (best_tau_a, {tau_a: auroc} dict).
    """
    mu = compute_bma(single_probs, scores_norm, tau_pred)
    correct = (mu.argmax(axis=1) == true_labels).astype(int)

    results = {}
    for ta in tau_a_grid:
        D  = compute_D(single_probs, scores_norm, mu, ta)
        auc = auroc(1.0 / D, correct)
        results[ta] = auc

    best_ta = max(results, key=lambda k: results[k] if not np.isnan(results[k]) else -1)
    return best_ta, results


def fit_c(
    D_val:       np.ndarray,   # (N_val,) disagreement on val split
    mu_val:      np.ndarray,   # (N_val, C)
    true_labels: np.ndarray,   # (N_val,)
) -> float:
    """
    Fit scalar c to minimise NLL under the Dirichlet-Categorical model.
    p(y|x) = E_{θ~Dir(c/D·μ)}[Cat(y|θ)] = μ_y   (same point prediction)

    NLL = -log μ_y  is independent of c, so instead we minimise
    Brier score with κ-shrunk probabilities as a calibration objective.

    Concretely: p_cal(y|x) = softmax(log μ · κ(x)) — a per-sample TS.
    We fit c on val NLL.
    """
    from scipy.special import gammaln

    def nll_dirichlet(log_c: float) -> float:
        c = np.exp(log_c)
        alpha = np.clip((c / D_val)[:, None] * mu_val, 1e-6, None)  # (N, C)
        alpha_y  = alpha[np.arange(len(true_labels)), true_labels]
        alpha_0  = alpha.sum(axis=1)
        # Dirichlet-Categorical log-likelihood: log Γ(α_y+1) - log Γ(α_0+1) + log Γ(α_0) - log Γ(α_y)
        ll = gammaln(alpha_y + 1) - gammaln(alpha_y) + gammaln(alpha_0) - gammaln(alpha_0 + 1)
        return float(-ll.mean())

    res = minimize_scalar(nll_dirichlet, bounds=(-5, 10), method="bounded")
    return float(np.exp(res.x))


# ─────────────────────────────────────────────────────────────────────────────
# Selective prediction
# ─────────────────────────────────────────────────────────────────────────────

def selective_prediction_curve(
    scores:      np.ndarray,  # (N,) higher = more confident
    correct:     np.ndarray,  # (N,) binary
    n_points:    int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (coverage, accuracy) arrays.
    At each coverage fraction, keep the top-fraction highest-score samples
    and compute accuracy on them.
    """
    order    = np.argsort(-scores)  # descending confidence
    correct  = correct[order]
    coverages = np.linspace(1.0 / n_points, 1.0, n_points)
    accs = []
    for cov in coverages:
        k = max(1, int(cov * len(correct)))
        accs.append(correct[:k].mean())
    return coverages, np.array(accs)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation for one (model, dataset, pool) combo
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:    str,
    dataset:  str,
    pool:     str,
    val_frac: float = 0.2,
    seed:     int   = 42,
    tau_pred: float | None = None,   # if None, use best τ from JSONL
    verbose:  bool  = True,
) -> dict | None:
    """
    Returns a dict with metrics for all methods and τ_a values.
    Returns None if data is missing or single_probs not present.
    """
    npz_path = COMP_DIR / f"{model}__{dataset}__{pool}.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=True)

    if "single_probs" not in data:
        print(f"  [SKIP] {model}/{dataset}/{pool}: single_probs missing — "
              f"re-run zpe_full.py --force to regenerate NPZ files.")
        return None

    single_probs = data["single_probs"]   # (P, N, C) float16
    scores_norm  = data["scores_norm"]    # (P,)
    true_labels  = data["true_labels"]    # (N,)
    P, N, C      = single_probs.shape

    # ── τ_pred: best τ for ZPE-norm from JSONL ────────────────────────────────
    if tau_pred is None:
        jsonl_path = COMP_DIR / f"{model}__{dataset}__{pool}.jsonl"
        records    = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
        zpe_norm   = [r for r in records
                      if r.get("method") == "zpe_norm" and r.get("mode") == "ensemble"]
        if zpe_norm:
            tau_pred = max(zpe_norm, key=lambda r: r["accuracy"])["temperature"]
        else:
            tau_pred = 1.0

    # ── Train/val/test split ──────────────────────────────────────────────────
    rng      = np.random.default_rng(seed)
    idx      = rng.permutation(N)
    n_val    = max(1, int(N * val_frac))
    val_idx  = idx[:n_val]
    test_idx = idx[n_val:]

    sp_val   = single_probs[:, val_idx, :]
    sp_test  = single_probs[:, test_idx, :]
    lab_val  = true_labels[val_idx]
    lab_test = true_labels[test_idx]

    # ── Fit τ_a on val split ──────────────────────────────────────────────────
    if verbose:
        print(f"  Fitting τ_a on {n_val} val samples…")

    best_ta, ta_auroc_map = fit_tau_a(sp_val, scores_norm, lab_val, tau_pred)

    # ── Compute μ and D on test split with best τ_a ───────────────────────────
    mu_test  = compute_bma(sp_test, scores_norm, tau_pred)           # (N_test, C)
    D_test_best = compute_D(sp_test, scores_norm, mu_test, best_ta)  # (N_test,)

    # Fit c on val split
    mu_val_arr = compute_bma(sp_val, scores_norm, tau_pred)
    D_val_best = compute_D(sp_val, scores_norm, mu_val_arr, best_ta)
    c_fitted   = fit_c(D_val_best, mu_val_arr, lab_val)

    kappa_test = kappa(D_test_best, c_fitted)                        # (N_test,)
    correct_test = (mu_test.argmax(1) == lab_test).astype(int)

    # ── Baselines: ZPE-raw and ZPE-norm confidence (max softmax) ────────────────
    max_conf_norm = mu_test.max(axis=1)                              # (N_test,)
    # ZPE-raw: ensemble with raw scores
    scores_raw = data["scores_raw"]
    mu_raw_test = compute_bma(sp_test, scores_raw, tau_pred)
    correct_raw = (mu_raw_test.argmax(1) == lab_test).astype(int)
    max_conf_raw = mu_raw_test.max(axis=1)

    # ── Selective prediction ──────────────────────────────────────────────────
    cov_pid,  acc_pid  = selective_prediction_curve(kappa_test,    correct_test)
    cov_zpe,  acc_zpe  = selective_prediction_curve(max_conf_norm, correct_test)

    # AUC-selective = mean accuracy across coverage points (higher = better)
    auc_sel_pid = float(acc_pid.mean())
    auc_sel_zpe = float(acc_zpe.mean())

    # ── AUROC ─────────────────────────────────────────────────────────────────
    auroc_pid     = auroc(kappa_test,    correct_test)
    auroc_zpe_norm= auroc(max_conf_norm, correct_test)
    auroc_zpe_raw = auroc(max_conf_raw,  correct_raw)
    auroc_maxconf = auroc_zpe_norm   # kept for backwards compat

    # ── D(x) vs plain MI (τ_a = τ_pred) ──────────────────────────────────────
    D_mi      = compute_D(sp_test, scores_norm, mu_test, tau_pred)   # plain MI
    auroc_mi  = auroc(1.0 / D_mi, correct_test)

    # ── τ_a ablation on test ──────────────────────────────────────────────────
    ta_test_auroc = {}
    for ta in TAU_A_GRID:
        D_ta = compute_D(sp_test, scores_norm, mu_test, ta)
        ta_test_auroc[ta] = auroc(1.0 / D_ta, correct_test)

    # ── Spearman: D(x) vs correctness ─────────────────────────────────────────
    rho, _ = spearmanr(-D_test_best, correct_test)

    result = dict(
        model=model, dataset=dataset, pool=pool,
        n_test=len(test_idx), n_val=n_val,
        tau_pred=tau_pred, best_tau_a=best_ta, c=c_fitted,
        # Selective prediction AUC
        auc_sel_pid=auc_sel_pid,
        auc_sel_zpe=auc_sel_zpe,
        auc_sel_gain=auc_sel_pid - auc_sel_zpe,
        # AUROC correctness prediction
        auroc_pid=auroc_pid,
        auroc_mi=auroc_mi,
        auroc_zpe_raw=auroc_zpe_raw,
        auroc_zpe_norm=auroc_zpe_norm,
        auroc_maxconf=auroc_maxconf,   # = auroc_zpe_norm, kept for compat
        # Spearman
        spearman_rho=float(rho),
        # τ_a maps (for ablation)
        ta_val_auroc=ta_auroc_map,
        ta_test_auroc=ta_test_auroc,
        # Curves (for plotting)
        sel_cov=cov_pid.tolist(),
        sel_acc_pid=acc_pid.tolist(),
        sel_acc_zpe=acc_zpe.tolist(),
    )

    if verbose:
        print(f"  τ_pred={tau_pred}  best_τ_a={best_ta}  c={c_fitted:.2f}")
        print(f"  AUROC  — PID: {auroc_pid:.4f}  |  ZPE-raw: {auroc_zpe_raw:.4f}  |  "
              f"ZPE-norm: {auroc_zpe_norm:.4f}  |  MI: {auroc_mi:.4f}")
        print(f"  Sel-AUC — PID: {auc_sel_pid:.4f}  |  ZPE-norm: {auc_sel_zpe:.4f}  "
              f"(Δ={auc_sel_pid - auc_sel_zpe:+.4f})")
        print(f"  Spearman ρ(−D, correct): {rho:.4f}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

DATASET_LABELS = {
    "food101_full": "Food-101",
    "food11":       "Food-11",
    "agriculture":  "Agriculture",
    "beans":        "Beans",
}


def print_summary_table(results: list[dict], pool: str = "247"):
    sub = [r for r in results if r["pool"] == pool]
    if not sub:
        print(f"No results for pool={pool}")
        return

    models   = [m for m in ALL_MODELS if any(r["model"] == m for r in sub)]
    datasets = [d for d in ALL_DATASETS if any(r["dataset"] == d for r in sub)]

    try:
        from tabulate import tabulate
        use_tab = True
    except ImportError:
        use_tab = False

    print(f"\n{'═'*110}")
    print(f"  PID Uncertainty Evaluation — Pool: {pool}")
    print(f"  Columns: AUROC(PID) | AUROC(ZPE-raw) | AUROC(ZPE-norm) | Sel-AUC gain | best τ_a")
    print(f"{'═'*110}")

    header = ["Model"] + [DATASET_LABELS.get(d, d) for d in datasets for _ in range(4)]
    sub_header = [""] + ["PID↑", "ZPE-raw", "ZPE-norm", "τ_a*"] * len(datasets)

    rows = []
    for model in models:
        row = [model]
        for dataset in datasets:
            r = next((x for x in sub if x["model"] == model and x["dataset"] == dataset), None)
            if r is None:
                row += ["—", "—", "—", "—"]
            else:
                row += [
                    f"{r['auroc_pid']:.4f}",
                    f"{r.get('auroc_zpe_raw', float('nan')):.4f}",
                    f"{r.get('auroc_zpe_norm', r['auroc_maxconf']):.4f}",
                    f"{r['best_tau_a']}",
                ]
        rows.append(row)

    if use_tab:
        print(tabulate(rows, headers=sub_header, tablefmt="simple"))
    else:
        for r in [sub_header] + rows:
            print("  ".join(f"{str(v):<12}" for v in r))


def print_tau_a_ablation(results: list[dict], model: str, dataset: str, pool: str = "247"):
    r = next((x for x in results
              if x["model"] == model and x["dataset"] == dataset and x["pool"] == pool), None)
    if r is None:
        print("No result found.")
        return

    print(f"\n  τ_a ablation — {model} / {dataset} / pool-{pool}")
    print(f"  {'τ_a':<10} {'AUROC val':<12} {'AUROC test':<12}")
    print(f"  {'-'*34}")
    for ta in TAU_A_GRID:
        val_auc  = r["ta_val_auroc"].get(ta, float("nan"))
        test_auc = r["ta_test_auroc"].get(ta, float("nan"))
        marker   = " ← best" if ta == r["best_tau_a"] else ""
        print(f"  {ta:<10} {val_auc:<12.4f} {test_auc:<12.4f}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def save_figures(results: list[dict], out_dir: Path):
    """Save PID figures: selective prediction curves + AUROC comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig_dir = out_dir.parent / "figures" / "pid"
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASET_LABELS = {
        "food101_full": "Food-101",
        "food11":       "Food-11",
        "agriculture":  "Agriculture",
        "beans":        "Beans",
    }
    POOL_LABELS = {
        "247":      "Pool-247",
        "426":      "Pool-426",
        "food":     "Pool-Food",
        "agri":     "Pool-Agri",
        "247+food": "Pool-247+Food",
        "247+agri": "Pool-247+Agri",
    }
    MODEL_LABELS = {
        "siglip-base":    "SigLIP-B",
        "siglip-so400m":  "SigLIP-SO400M",
        "siglip-large":   "SigLIP-L",
        "clip-vit-b32":   "CLIP-B/32",
        "clip-vit-b16":   "CLIP-B/16",
        "clip-vit-large": "CLIP-L",
    }
    COLORS = {
        "siglip-base":    "#1f77b4",
        "siglip-so400m":  "#ff7f0e",
        "siglip-large":   "#2ca02c",
        "clip-vit-b32":   "#9467bd",
        "clip-vit-b16":   "#8c564b",
        "clip-vit-large": "#e377c2",
    }

    # Pool groups layout: each row = (row_label, [(pool, dataset), ...])
    # Rows 1-2: full pools (4 datasets each)
    # Row 3: domain pools (food 2 ds + agri 2 ds)
    # Row 4: combined pools (247+food 2 ds + 247+agri 2 ds)
    ROW_GROUPS = [
        ("Pool-247",         [("247",  "food101_full"), ("247",  "food11"),
                              ("247",  "agriculture"),  ("247",  "beans")]),
        ("Pool-Food / Agri", [("food", "food101_full"), ("food", "food11"),
                              ("agri", "agriculture"),  ("agri", "beans")]),
    ]
    N_ROWS = len(ROW_GROUPS)
    N_COLS = 4

    # ── Fig A: Selective prediction — all pools, all datasets ──────────────
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3.5, N_ROWS * 3),
                             squeeze=False)
    for row_idx, (row_label, cells) in enumerate(ROW_GROUPS):
        for col_idx, (pool, dataset) in enumerate(cells):
            ax = axes[row_idx][col_idx]
            sub = [r for r in results if r["pool"] == pool and r["dataset"] == dataset
                   and r.get("sel_cov")]
            if sub:
                for r in sub:
                    model = r["model"]
                    cov = r["sel_cov"]
                    ax.plot(cov, r["sel_acc_pid"], color=COLORS[model],
                            linewidth=1.4)
                    ax.plot(cov, r["sel_acc_zpe"], color=COLORS[model],
                            linestyle="--", linewidth=0.9, alpha=0.55)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.25)
            else:
                ax.set_visible(False)
                continue
            if row_idx == 0:
                ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(row_label + "\nAccuracy", fontsize=7.5)
            if row_idx == N_ROWS - 1:
                ax.set_xlabel("Coverage", fontsize=8)
    # shared legend
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=c, linewidth=1.5, label=MODEL_LABELS[m])
                      for m, c in COLORS.items()]
    legend_handles += [
        Line2D([0], [0], color="k", linewidth=1.5, label="PID (solid)"),
        Line2D([0], [0], color="k", linewidth=1.0, linestyle="--", alpha=0.6, label="ZPE-norm (dashed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Selective Prediction Accuracy vs Coverage\n(solid = PID, dashed = ZPE-norm)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ("pdf", "png"):
        plt.savefig(fig_dir / f"figA_sel_pred_all.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figA (selective prediction — all pools)")

    # ── Fig B: AUROC bar chart — all pools ────────────────────────────────
    pools_present = [p for p in ["247", "426", "food", "agri", "247+food", "247+agri"]
                     if any(r["pool"] == p for r in results)]
    models_present = [m for m in ALL_MODELS if any(r["model"] == m for r in results)]
    n_pools = len(pools_present)
    n_ds    = len(ALL_DATASETS)
    n_mod   = len(models_present)

    fig, axes = plt.subplots(n_pools, n_ds, figsize=(n_ds * 3.0, n_pools * 2.5),
                             squeeze=False, sharey="row")
    for p_idx, pool in enumerate(pools_present):
        sub_pool = [r for r in results if r["pool"] == pool]
        ds_for_pool = [d for d in ALL_DATASETS if any(r["dataset"] == d for r in sub_pool)]
        for d_idx, ds in enumerate(ALL_DATASETS):
            ax = axes[p_idx][d_idx]
            rows = [r for r in sub_pool if r["dataset"] == ds]
            if not rows or ds not in ds_for_pool:
                ax.set_visible(False)
                continue
            x   = np.arange(n_mod)
            w   = 0.25
            pid_vals  = [next((r["auroc_pid"]                              for r in rows if r["model"] == m), np.nan) for m in models_present]
            raw_vals  = [next((r.get("auroc_zpe_raw", r["auroc_maxconf"])  for r in rows if r["model"] == m), np.nan) for m in models_present]
            norm_vals = [next((r.get("auroc_zpe_norm", r["auroc_maxconf"]) for r in rows if r["model"] == m), np.nan) for m in models_present]
            ax.bar(x - w,  pid_vals,  width=w, label="PID",      color="#1f77b4", alpha=0.85)
            ax.bar(x,      raw_vals,  width=w, label="ZPE-raw",  color="#ff7f0e", alpha=0.85)
            ax.bar(x + w,  norm_vals, width=w, label="ZPE-norm", color="#2ca02c", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([MODEL_LABELS[m] for m in models_present],
                               rotation=30, ha="right", fontsize=6.5)
            ax.set_ylim(0.45, 1.0)
            ax.grid(axis="y", alpha=0.3)
            if p_idx == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=9)
            if d_idx == 0:
                ax.set_ylabel(POOL_LABELS.get(pool, pool) + "\nAUROC", fontsize=7.5)
            if p_idx == 0 and d_idx == 0:
                ax.legend(fontsize=6.5)
    fig.suptitle("Correctness Prediction AUROC — PID vs ZPE-raw vs ZPE-norm", fontsize=11)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(fig_dir / f"figB_auroc_bar_all.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figB (AUROC bar chart — all pools)")

    # ── Fig C: τ_a ablation — all pools × datasets × models ─────────────
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 3.5, N_ROWS * 3),
                             squeeze=False)
    legend_handles_c = [Line2D([0], [0], color=c, linewidth=1.5,
                                marker="o", markersize=4, label=MODEL_LABELS[m])
                        for m, c in COLORS.items()]
    fig.legend(handles=legend_handles_c, loc="upper center",
               ncol=len(legend_handles_c), fontsize=8, bbox_to_anchor=(0.5, 1.0),
               frameon=True, borderpad=0.4)

    for row_idx, (row_label, cells) in enumerate(ROW_GROUPS):
        for col_idx, (pool, dataset) in enumerate(cells):
            ax = axes[row_idx][col_idx]
            sub = [r for r in results if r["pool"] == pool and r["dataset"] == dataset
                   and r.get("ta_test_auroc")]
            if not sub:
                ax.set_visible(False)
                continue
            for r in sub:
                model = r["model"]
                ta_vals = sorted(float(k) for k in r["ta_test_auroc"].keys())
                auroc_t = [r["ta_test_auroc"][str(t)] for t in ta_vals]
                best_ta  = r["best_tau_a"]
                best_auc = r["ta_test_auroc"][str(best_ta)]
                ax.semilogx(ta_vals, auroc_t, color=COLORS[model],
                            linewidth=1.3, marker="o", markersize=4)
                ax.plot(best_ta, best_auc, "o", color=COLORS[model],
                        markersize=10, markeredgecolor="white", markeredgewidth=1.0,
                        zorder=5)
            ax.grid(True, alpha=0.25)
            ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=9)
            row_pools = [p for p, _ in cells]
            if row_pools.index(pool) == col_idx:
                ax.set_ylabel(POOL_LABELS.get(pool, pool) + "\nAUROC (test)", fontsize=7.5)
            ax.set_xlabel("τ_a", fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        plt.savefig(fig_dir / f"figC_ta_ablation_all.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figC (τ_a ablation — all pools)")


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty figure  (scatter AUROC-PID vs AUROC-MaxConf + ΔSel-AUC bars)
# ─────────────────────────────────────────────────────────────────────────────

def save_uncertainty_figure(results: list[dict]):
    """Two-row figure, one column per pool.

    Row 1 — Scatter: AUROC Max-conf (x) vs AUROC PID (y).
             color = dataset, marker = model, diagonal y=x.
    Row 2 — Bar chart: ΔSel-AUC = auc_sel_pid − auc_sel_zpe per model.
             color = dataset, x = model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig_dir = OUT_DIR.parent / "figures" / "pid"
    fig_dir.mkdir(parents=True, exist_ok=True)

    DS_LABELS  = {"food101_full": "Food-101", "food11": "Food-11",
                  "agriculture": "Agriculture", "beans": "Beans"}
    DS_COLORS  = {"food101_full": "#1f77b4", "food11": "#ff7f0e",
                  "agriculture":  "#2ca02c",  "beans": "#d62728"}
    MOD_LABELS = {"siglip-base": "SigLIP-B", "siglip-so400m": "SigLIP-SO400M",
                  "siglip-large": "SigLIP-L", "clip-vit-b32": "CLIP-B/32",
                  "clip-vit-b16": "CLIP-B/16", "clip-vit-large": "CLIP-L"}
    MOD_MARKERS= {"siglip-base": "o", "siglip-so400m": "s", "siglip-large": "^",
                  "clip-vit-b32": "D", "clip-vit-b16": "v", "clip-vit-large": "P"}
    MOD_ORDER  = ["siglip-base", "siglip-so400m", "siglip-large",
                  "clip-vit-b32", "clip-vit-b16", "clip-vit-large"]

    POOLS = ["247", "426", "food", "agri"]
    POOL_LABELS = {"247": "Pool-247", "426": "Pool-426",
                   "food": "Pool-food", "agri": "Pool-agri"}

    pools_present  = [p for p in POOLS if any(r["pool"] == p for r in results)]
    models_present = [m for m in MOD_ORDER if any(r["model"] == m for r in results)]
    n_pools = len(pools_present)

    fig, axes = plt.subplots(2, n_pools,
                             figsize=(n_pools * 3.4, 7.0),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.38, wspace=0.25)

    # ── Row 1: scatter ────────────────────────────────────────────────────────
    for c_idx, pool in enumerate(pools_present):
        ax  = axes[0][c_idx]
        sub = [r for r in results if r["pool"] == pool]

        lo, hi = 0.28, 1.02
        ax.plot([lo, hi], [lo, hi], "--", color="#aaaaaa", linewidth=0.9, zorder=0)

        for r in sub:
            x = r.get("auroc_maxconf", r.get("auroc_zpe_norm", float("nan")))
            y = r["auroc_pid"]
            ds, model = r["dataset"], r["model"]
            ax.scatter(x, y,
                       color=DS_COLORS.get(ds, "gray"),
                       marker=MOD_MARKERS.get(model, "o"),
                       s=55, zorder=3,
                       edgecolors="white", linewidths=0.4)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("AUROC — Max-conf", fontsize=8.5)
        if c_idx == 0:
            ax.set_ylabel("AUROC — PID", fontsize=8.5)
        ax.set_title(POOL_LABELS[pool], fontsize=10, fontweight="bold")
        ax.text(0.04, 0.96, "PID↑", transform=ax.transAxes,
                fontsize=7.5, color="#555555", va="top")
        ax.text(0.96, 0.04, "Conf↑", transform=ax.transAxes,
                fontsize=7.5, color="#555555", ha="right")
        ax.grid(True, alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Row 2: ΔSel-AUC bar chart ─────────────────────────────────────────────
    for c_idx, pool in enumerate(pools_present):
        ax  = axes[1][c_idx]
        sub = [r for r in results if r["pool"] == pool]
        datasets_here = sorted({r["dataset"] for r in sub},
                                key=lambda d: list(DS_LABELS.keys()).index(d))

        n_mod = len(models_present)
        n_ds  = len(datasets_here)
        bar_w = 0.8 / max(n_ds, 1)
        x     = np.arange(n_mod)

        for di, ds in enumerate(datasets_here):
            vals = []
            for model in models_present:
                row = next((r for r in sub if r["model"] == model
                            and r["dataset"] == ds), None)
                vals.append(row["auc_sel_gain"] if row else np.nan)
            offset = (di - (n_ds - 1) / 2) * bar_w
            ax.bar(x + offset, vals, width=bar_w,
                   color=DS_COLORS[ds], label=DS_LABELS[ds], alpha=0.88)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([MOD_LABELS.get(m, m) for m in models_present],
                           fontsize=7.5, rotation=30, ha="right")
        if c_idx == 0:
            ax.set_ylabel("ΔSel-AUC (PID − ZPE-norm)", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)


    # ── Shared top legend: datasets (color) + models (marker) ─────────────────
    ds_handles  = [Patch(color=DS_COLORS[d], label=DS_LABELS[d])
                   for d in DS_LABELS if any(r["dataset"] == d for r in results)]
    mod_handles = [Line2D([0], [0], color="#555555",
                          marker=MOD_MARKERS[m], markersize=6,
                          linestyle="none", label=MOD_LABELS[m])
                   for m in models_present]
    fig.legend(handles=ds_handles + mod_handles,
               loc="upper center", ncol=len(ds_handles) + len(mod_handles),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.02),
               borderpad=0.5)

    fig.suptitle("PID uncertainty evaluation — all pools", fontsize=11, y=1.06)

    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"fig_uncertainty_all_pools.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → results/figures/pid/fig_uncertainty_all_pools.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    nargs="+", default=ALL_MODELS,   dest="models")
    p.add_argument("--dataset",  nargs="+", default=ALL_DATASETS, dest="datasets")
    p.add_argument("--pool",     nargs="+", default=None)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--table",           action="store_true", help="Print summary table")
    p.add_argument("--ablation",        action="store_true", help="Print τ_a ablation for each combo")
    p.add_argument("--save",            action="store_true", help="Save results to results/pid/")
    p.add_argument("--uncertainty-fig", action="store_true",
                   help="Generate uncertainty figure from saved pid_results.json")
    return p.parse_args()


def main():
    import json as _json
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.uncertainty_fig:
        saved = OUT_DIR / "pid_results.json"
        if not saved.exists():
            print(f"ERROR: {saved} not found. Run with --save first.")
            return
        results = _json.loads(saved.read_text())
        print(f"Loaded {len(results)} records from {saved}")
        save_uncertainty_figure(results)
        return

    all_results = []

    for model, dataset in iproduct(args.models, args.datasets):
        valid_pools = DATASET_POOLS.get(dataset, [])
        pools = [p for p in valid_pools if args.pool is None or p in args.pool]
        for pool in pools:
            print(f"\n{'─'*60}")
            print(f"  {model} | {dataset} | pool={pool}")
            print(f"{'─'*60}")
            r = evaluate(
                model, dataset, pool,
                val_frac=args.val_frac,
                seed=args.seed,
                verbose=True,
            )
            if r is not None:
                all_results.append(r)
                if args.ablation:
                    print_tau_a_ablation(all_results, model, dataset, pool)

    if not all_results:
        print("\nNo results — re-run zpe_full.py --force to regenerate NPZ files "
              "with single_probs.")
        return

    if args.table:
        pools_present = sorted({r["pool"] for r in all_results})
        for pool in pools_present:
            print_summary_table(all_results, pool=pool)

    if args.save:
        import json as _json
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / "pid_results.json"
        # strip non-serialisable numpy types
        serialisable = []
        for r in all_results:
            row = {k: (v.tolist() if hasattr(v, "tolist") else v)
                   for k, v in r.items()}
            serialisable.append(row)
        out.write_text(_json.dumps(serialisable, indent=2))
        print(f"\nSaved → {out}")

        print("\nGenerating figures…")
        save_figures(all_results, OUT_DIR)


if __name__ == "__main__":
    main()
