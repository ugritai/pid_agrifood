"""
Lexical analysis of prompts.

Figures
-------
  1  ZPE score vs single-prompt accuracy  (scatter + Spearman ρ)
     → validates ZPE score as a label-free proxy for prompt quality

  2  Top-K / Bottom-K prompt table
     → shows ZPE identifies out-of-domain language automatically

  3  Word discriminability
     → words over-represented in top-25% vs bottom-25% prompts by ZPE score
     → coloured bar chart (positive = top-quartile words, negative = bottom)

  4  Prompt length vs ZPE score  (scatter)
     → does verbosity help or hurt?

  5  Lexical diversity of pool vs disagreement
     → for each pool: avg pairwise token-Jaccard distance (diversity)
        vs mean D(x) across samples (uncertainty signal quality)
     → motivates: diverse pools → more informative uncertainty

  6  ZPE score stability across datasets
     → Spearman ρ of prompt rankings between all pairs of datasets
     → do "good prompts" transfer across domains?

Usage
-----
    python analysis/lexical.py                        # all figures
    python analysis/lexical.py --fig 1 3              # specific figures
    python analysis/lexical.py --model siglip-so400m  # one model
    python analysis/lexical.py --pool 247
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

COMP_DIR = Path("results/comprehensive")
FIG_DIR  = Path("results/figures/lexical")

ALL_MODELS = [
    "siglip-so400m",
    "clip-vit-large",
]
ALL_DATASETS = ["food101_full", "food11", "agriculture", "beans"]
DATASET_LABELS = {
    "food101_full": "Food-101",
    "food11":       "Food-11",
    "agriculture":  "Agriculture",
    "beans":        "Beans",
}
MODEL_COLORS = {
    "siglip-base":   "#1f77b4",
    "siglip-so400m": "#ff7f0e",
    "siglip-large":  "#9467bd",
    "clip-vit-b32":  "#2ca02c",
    "clip-vit-b16":  "#98df8a",
    "clip-vit-large":"#d62728",
}

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load(model: str, dataset: str, pool: str):
    npz_path   = COMP_DIR / f"{model}__{dataset}__{pool}.npz"
    jsonl_path = COMP_DIR / f"{model}__{dataset}__{pool}.jsonl"
    if not npz_path.exists() or not jsonl_path.exists():
        return None
    npz     = np.load(npz_path, allow_pickle=True)
    records = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
    single  = [r for r in records if r["mode"] == "single"]
    # index by prompt_idx
    acc_map = {r["prompt_idx"]: r["accuracy"] for r in single}
    ece_map = {r["prompt_idx"]: r["ece"]      for r in single}
    return dict(
        prompts     = list(npz["prompts"]),
        scores_norm = npz["scores_norm"].astype(float),
        acc         = np.array([acc_map.get(i, np.nan) for i in range(len(npz["prompts"]))]),
        ece         = np.array([ece_map.get(i, np.nan) for i in range(len(npz["prompts"]))]),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Lexical utilities
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "is", "it",
              "this", "that", "with", "for", "to", "and", "or", "by",
              "as", "be", "are", "was", "were", "has", "have", "its"}

def tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords and placeholder {}."""
    text = text.lower().replace("{}", "").replace(".", "").replace(",", "")
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pairwise_diversity(prompts: list[str]) -> float:
    """Average pairwise token-Jaccard DISTANCE (1 - Jaccard) across all prompt pairs."""
    sets = [set(tokenise(p)) for p in prompts]
    dists = [1.0 - jaccard(sets[i], sets[j])
             for i, j in combinations(range(len(sets)), 2)]
    return float(np.mean(dists)) if dists else 0.0


def word_discriminability(
    prompts: list[str],
    scores:  np.ndarray,
    top_frac: float = 0.25,
) -> list[tuple[str, float]]:
    """
    For each word, compute log-ratio of frequency in top vs bottom quartile.
    Returns list of (word, log_ratio) sorted by |log_ratio|.
    """
    n = len(prompts)
    order    = np.argsort(-scores)
    top_idx  = set(order[:max(1, int(n * top_frac))])
    bot_idx  = set(order[-max(1, int(n * top_frac)):])

    top_words = Counter(w for i in top_idx for w in tokenise(prompts[i]))
    bot_words = Counter(w for i in bot_idx for w in tokenise(prompts[i]))

    all_words = set(top_words) | set(bot_words)
    results   = []
    total_top = max(1, sum(top_words.values()))
    total_bot = max(1, sum(bot_words.values()))

    for word in all_words:
        f_top = (top_words[word] + 0.5) / total_top
        f_bot = (bot_words[word] + 0.5) / total_bot
        log_r = np.log2(f_top / f_bot)
        results.append((word, log_r))

    return sorted(results, key=lambda x: abs(x[1]), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        path = FIG_DIR / (name + ext)
        fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved → {FIG_DIR / name}.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — ZPE score vs single-prompt accuracy
# ─────────────────────────────────────────────────────────────────────────────

def fig_score_vs_accuracy(model: str, datasets: list[str], pool: str = "247"):
    """
    Scatter: x = ZPE-norm score, y = single-prompt accuracy.
    One panel per dataset. Spearman ρ annotated.
    """
    valid = [(d, load(model, d, pool)) for d in datasets]
    valid = [(d, dat) for d, dat in valid if dat is not None]
    if not valid:
        print("  No data for fig 1."); return

    ncols = len(valid)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.5), squeeze=False)

    for col, (dataset, dat) in enumerate(valid):
        ax  = axes[0][col]
        s   = dat["scores_norm"]
        acc = dat["acc"]

        mask = ~np.isnan(acc)
        rho, pval = spearmanr(s[mask], acc[mask])

        ax.scatter(s[mask], acc[mask], s=8, alpha=0.5,
                   color=MODEL_COLORS.get(model, "#555555"), linewidths=0)

        # Annotate top-3 and bottom-3
        order = np.argsort(-s)
        for i in order[:3]:
            ax.annotate(dat["prompts"][i].replace("{}", "·")[:30],
                        (s[i], acc[i]), fontsize=5, ha="left",
                        xytext=(3, 0), textcoords="offset points", color="#d62728")
        for i in order[-3:]:
            ax.annotate(dat["prompts"][i].replace("{}", "·")[:30],
                        (s[i], acc[i]), fontsize=5, ha="left",
                        xytext=(3, 0), textcoords="offset points", color="#1f77b4")

        ax.set_xlabel("ZPE-norm score", fontsize=9)
        if col == 0:
            ax.set_ylabel("Single-prompt accuracy", fontsize=9)
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=10)
        ax.annotate(f"ρ = {rho:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=9, va="top",
                    color="green" if pval < 0.01 else "orange")

    fig.suptitle(f"ZPE-norm score vs prompt accuracy  ({model}, Pool-{pool})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"fig1_score_vs_acc_{model}_{pool}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Top-K / Bottom-K prompt tables
# ─────────────────────────────────────────────────────────────────────────────

def fig_top_bottom_prompts(
    model: str, dataset: str, pool: str = "247", K: int = 10
):
    dat = load(model, dataset, pool)
    if dat is None:
        print("  No data for fig 2."); return

    scores  = dat["scores_norm"]
    prompts = dat["prompts"]
    accs    = dat["acc"]
    order   = np.argsort(-scores)

    fig, (ax_top, ax_bot) = plt.subplots(1, 2, figsize=(14, 3.5))

    for ax, indices, title, color in [
        (ax_top, order[:K],    f"Top-{K} prompts (ZPE-norm ↑)", "#d62728"),
        (ax_bot, order[-K:],   f"Bottom-{K} prompts (ZPE-norm ↓)", "#1f77b4"),
    ]:
        ax.axis("off")
        rows = []
        for rank, i in enumerate(indices, 1):
            p = prompts[i].replace("{}", "{class}")
            if len(p) > 55:
                p = p[:52] + "…"
            rows.append([str(rank), p,
                         f"{scores[i]:.2f}",
                         f"{accs[i]:.4f}"])

        col_labels = ["Rank", "Prompt", "Score", "Acc"]
        col_widths = [0.05, 0.65, 0.15, 0.15]
        tbl = ax.table(
            cellText=rows, colLabels=col_labels,
            loc="center", cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.4)

        # Header colour
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor(color)
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=8)

    fig.suptitle(
        f"Prompt ranking by ZPE-norm score  "
        f"({model}, {DATASET_LABELS.get(dataset, dataset)}, Pool-{pool})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, f"fig2_top_bottom_{model}_{dataset}_{pool}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Word discriminability
# ─────────────────────────────────────────────────────────────────────────────

def fig_word_discriminability(
    model: str, datasets: list[str], pool: str = "247", top_n: int = 20,
    models: list[str] | None = None,
):
    """
    Horizontal bar chart: words with highest |log2-ratio| of frequency
    in top-25% vs bottom-25% prompts by ZPE score.
    Positive (red) = overrepresented in high-scoring prompts.
    Negative (blue) = overrepresented in low-scoring prompts.

    If `models` is provided, generates a grid rows=models × cols=datasets
    in a single figure.
    """
    from matplotlib.patches import Patch

    if models is None:
        models = [model]

    # filter to models × datasets that have data
    grid = {}
    for m in models:
        for d in datasets:
            dat = load(m, d, pool)
            if dat is not None:
                grid[(m, d)] = dat

    if not grid:
        print("  No data for fig 3."); return

    valid_models   = [m for m in models   if any(k[0] == m for k in grid)]
    valid_datasets = [d for d in datasets if any(k[1] == d for k in grid)]
    nrows = len(valid_models)
    ncols = len(valid_datasets)

    MODEL_LABELS_SHORT = {
        "siglip-base":    "SigLIP-B",    "siglip-so400m": "SigLIP-SO400M",
        "siglip-large":   "SigLIP-L",    "clip-vit-b32":  "CLIP-B/32",
        "clip-vit-b16":   "CLIP-B/16",   "clip-vit-large":"CLIP-L",
    }

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.2 * nrows),
                             squeeze=False)

    for row, m in enumerate(valid_models):
        for col, d in enumerate(valid_datasets):
            ax = axes[row][col]
            dat = grid.get((m, d))
            if dat is None:
                ax.set_visible(False); continue

            disc    = word_discriminability(dat["prompts"], dat["scores_norm"])[:top_n]
            words   = [w for w, _ in disc]
            lograts = [r for _, r in disc]
            colors  = ["#d62728" if r > 0 else "#1f77b4" for r in lograts]
            y_pos   = np.arange(len(words))

            ax.barh(y_pos, lograts, color=colors, edgecolor="none", height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=7.5 if nrows > 1 else 8)
            ax.axvline(0, color="black", lw=0.8)
            ax.invert_yaxis()
            if row == nrows - 1:
                ax.set_xlabel("log₂(top / bottom)", fontsize=7.5)
            if row == 0:
                ax.set_title(DATASET_LABELS.get(d, d), fontsize=9)
            if col == 0:
                ax.set_ylabel(MODEL_LABELS_SHORT.get(m, m), fontsize=8)

    handles = [Patch(color="#d62728", label="Top-25% prompts"),
               Patch(color="#1f77b4", label="Bottom-25% prompts")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Word discriminability — ZPE-norm top vs bottom quartile  (Pool-{pool})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    suffix = "all" if len(valid_models) > 1 else model
    _save(fig, f"fig3_word_disc_{suffix}_{pool}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Prompt length vs ZPE score
# ─────────────────────────────────────────────────────────────────────────────

def fig_length_vs_score(model: str, datasets: list[str], pool: str = "247"):
    valid = [(d, load(model, d, pool)) for d in datasets]
    valid = [(d, dat) for d, dat in valid if dat is not None]
    if not valid:
        print("  No data for fig 4."); return

    ncols = len(valid)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.2), squeeze=False)

    for col, (dataset, dat) in enumerate(valid):
        ax = axes[0][col]
        lengths = np.array([len(tokenise(p)) for p in dat["prompts"]])
        scores  = dat["scores_norm"]
        rho, _  = spearmanr(lengths, scores)

        ax.scatter(lengths, scores, s=10, alpha=0.5,
                   color=MODEL_COLORS.get(model, "#555555"), linewidths=0)
        ax.set_xlabel("Prompt length (tokens, no stopwords)", fontsize=8)
        if col == 0:
            ax.set_ylabel("ZPE-norm score", fontsize=8)
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=10)
        ax.annotate(f"ρ = {rho:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                    fontsize=9, va="top")

    fig.suptitle(f"Prompt length vs ZPE-norm score  ({model}, Pool-{pool})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"fig4_length_vs_score_{model}_{pool}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Pool diversity vs disagreement quality
# ─────────────────────────────────────────────────────────────────────────────

def fig_diversity_vs_disagreement(models: list[str], datasets: list[str]):
    """
    For each (model, dataset, pool): compute
      x = lexical diversity of pool (avg pairwise Jaccard distance)
      y = Spearman ρ between ZPE score and single-prompt accuracy
          (proxy for how well ZPE score discriminates good prompts)

    Point = one (model, dataset, pool) combo.
    Motivates: diverse pools → ZPE score is more discriminative.
    """
    from itertools import product as iproduct
    DATASET_POOLS = {
        "food101_full": ["247", "426", "food", "247+food"],
        "food11":       ["247", "426", "food", "247+food"],
        "agriculture":  ["247", "426", "agri", "247+agri"],
        "beans":        ["247", "426", "agri", "247+agri"],
    }
    POOL_MARKERS = {"247": "o", "426": "s", "food": "^", "agri": "D",
                    "247+food": "P", "247+agri": "X"}
    POOL_LABELS  = {"247": "Pool-247", "426": "Pool-426",
                    "food": "Pool-Food", "agri": "Pool-Agri",
                    "247+food": "Pool-247+Food", "247+agri": "Pool-247+Agri"}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    pool_handles = {}
    for model, dataset in iproduct(models, datasets):
        color  = MODEL_COLORS.get(model, "#888888")
        for pool in DATASET_POOLS.get(dataset, []):
            dat = load(model, dataset, pool)
            if dat is None:
                continue

            div  = pairwise_diversity(dat["prompts"])
            mask = ~np.isnan(dat["acc"])
            rho, _ = spearmanr(dat["scores_norm"][mask], dat["acc"][mask])

            m = POOL_MARKERS.get(pool, "o")
            sc = ax.scatter(div, rho, s=55, color=color, marker=m,
                            alpha=0.75, edgecolors="white", linewidths=0.5,
                            label=POOL_LABELS.get(pool, pool))
            if pool not in pool_handles:
                pool_handles[pool] = plt.scatter(
                    [], [], s=55, color="grey", marker=m,
                    label=POOL_LABELS.get(pool, pool))

    ax.set_xlabel("Lexical diversity of pool\n(avg pairwise token-Jaccard distance)", fontsize=9)
    ax.set_ylabel("Spearman ρ  (ZPE score vs prompt accuracy)", fontsize=9)
    ax.set_title("More lexically diverse pools → ZPE score is a better quality proxy",
                 fontsize=10, fontweight="bold")

    # Pool legend (shape)
    pool_legend = ax.legend(handles=list(pool_handles.values()),
                            title="Pool", fontsize=8, loc="lower right")
    ax.add_artist(pool_legend)

    # Model legend (colour)
    model_handles = [
        plt.scatter([], [], s=55, color=MODEL_COLORS[m], marker="o", label=m)
        for m in models if any(
            load(m, d, p) is not None
            for d in datasets
            for p in DATASET_POOLS.get(d, [])
        )
    ]
    ax.legend(handles=model_handles, title="Model", fontsize=7,
              loc="upper left", ncol=2)

    fig.tight_layout()
    _save(fig, "fig5_diversity_vs_disagreement")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Prompt score stability across datasets
# ─────────────────────────────────────────────────────────────────────────────

def fig_score_stability(model: str, datasets: list[str], pool: str = "247"):
    """
    Heatmap of Spearman ρ of ZPE-norm prompt rankings across dataset pairs.
    High ρ = the same prompts score well on both datasets.
    """
    data_by_ds = {}
    for d in datasets:
        dat = load(model, d, pool)
        if dat is not None:
            data_by_ds[d] = dat["scores_norm"]

    ds_list = list(data_by_ds.keys())
    n       = len(ds_list)
    if n < 2:
        print("  Need ≥2 datasets for fig 6."); return

    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = data_by_ds[ds_list[i]], data_by_ds[ds_list[j]]
            rho, _ = spearmanr(s1, s2)
            mat[i, j] = mat[j, i] = rho

    labels = [DATASET_LABELS.get(d, d) for d in ds_list]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(mat[i, j]) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title(f"Prompt ranking stability across datasets\n({model}, Pool-{pool})",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"fig6_score_stability_{model}_{pool}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Cross-domain: food vs agriculture prompt scores
# ─────────────────────────────────────────────────────────────────────────────

FOOD_DATASETS = ["food101_full", "food11"]
AGRI_DATASETS = ["agriculture", "beans"]

def fig_cross_domain(model: str, pool: str = "247", top_k: int = 30):
    """
    Three panels:

    Left  — Scatter: avg ZPE score on food datasets vs avg ZPE score on agri
             datasets. Each point = one prompt. Quadrants: domain-general (top-right),
             food-specific (top-left), agri-specific (bottom-right), bad everywhere.
             Annotate outliers.

    Middle — Word discriminability: words over-represented in food-top vs agri-top
             prompts. Red = food domain words, blue = agri domain words.

    Right  — Venn counts + sample prompts for the three groups:
             food-only top, agri-only top, shared top.
    """
    # ── load and average scores across domain datasets ────────────────────────
    def avg_scores(datasets):
        all_s = []
        for d in datasets:
            dat = load(model, d, pool)
            if dat is not None:
                all_s.append(dat["scores_norm"])
        if not all_s:
            return None, None
        mat = np.vstack(all_s)
        return mat.mean(0), mat

    s_food, mat_food = avg_scores(FOOD_DATASETS)
    s_agri, mat_agri = avg_scores(AGRI_DATASETS)
    if s_food is None or s_agri is None:
        print("  Not enough data for fig 7."); return

    dat0    = load(model, FOOD_DATASETS[0], pool)
    prompts = dat0["prompts"]
    P       = len(prompts)

    from scipy.stats import spearmanr as _sp
    rho, _ = _sp(s_food, s_agri)

    # Quadrant assignment (top-K by each domain)
    top_food_idx = set(np.argsort(-s_food)[:top_k])
    top_agri_idx = set(np.argsort(-s_agri)[:top_k])
    shared       = top_food_idx & top_agri_idx
    food_only    = top_food_idx - top_agri_idx
    agri_only    = top_agri_idx - top_food_idx

    # ── word discriminability across domains ─────────────────────────────────
    def domain_word_disc(top_idx_a, top_idx_b, top_n=18):
        """Words over-represented in top_idx_a vs top_idx_b."""
        words_a = Counter(w for i in top_idx_a for w in tokenise(prompts[i]))
        words_b = Counter(w for i in top_idx_b for w in tokenise(prompts[i]))
        total_a = max(1, sum(words_a.values()))
        total_b = max(1, sum(words_b.values()))
        all_w   = set(words_a) | set(words_b)
        disc    = []
        for w in all_w:
            fa = (words_a[w] + 0.5) / total_a
            fb = (words_b[w] + 0.5) / total_b
            disc.append((w, np.log2(fa / fb)))
        disc.sort(key=lambda x: x[1], reverse=True)
        # top_n food words + top_n agri words
        return disc[:top_n] + disc[-top_n:]

    disc = domain_word_disc(top_food_idx, top_agri_idx)
    words   = [w for w, _ in disc]
    lograts = [r for _, r in disc]

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1.4, 1.0, 0.9], wspace=0.35)
    ax_scatter = fig.add_subplot(gs[0])
    ax_words   = fig.add_subplot(gs[1])
    ax_venn    = fig.add_subplot(gs[2])

    # ── Panel 1: scatter ──────────────────────────────────────────────────────
    colors_pts = []
    for i in range(P):
        if i in shared:
            colors_pts.append("#2ca02c")   # green = domain-general
        elif i in food_only:
            colors_pts.append("#ff7f0e")   # orange = food-specific
        elif i in agri_only:
            colors_pts.append("#1f77b4")   # blue = agri-specific
        else:
            colors_pts.append("#cccccc")   # grey = low both

    ax_scatter.scatter(s_food, s_agri, c=colors_pts, s=14, alpha=0.75, linewidths=0)

    # Annotate interesting outliers
    def annotate_pts(indices, n=4, color="black"):
        # pick the most extreme (farthest from diagonal) within the set
        subset = list(indices)
        if not subset: return
        diffs = [abs(s_food[i] - s_agri[i]) for i in subset]
        for i in sorted(subset, key=lambda x: abs(s_food[x] - s_agri[x]), reverse=True)[:n]:
            txt = prompts[i].replace("{}", "·")[:35]
            ax_scatter.annotate(txt, (s_food[i], s_agri[i]),
                                fontsize=5.5, color=color,
                                xytext=(4, 0), textcoords="offset points")

    annotate_pts(food_only, 3, color="#d65f00")
    annotate_pts(agri_only, 3, color="#1560a8")
    annotate_pts(shared,    3, color="#1a7a1a")

    ax_scatter.set_xlabel(f"Avg ZPE-norm score — Food ({', '.join(FOOD_DATASETS)})", fontsize=8)
    ax_scatter.set_ylabel(f"Avg ZPE-norm score — Agriculture ({', '.join(AGRI_DATASETS)})", fontsize=8)
    ax_scatter.set_title(f"Prompt score: food vs agriculture\nSpearman ρ = {rho:.3f}", fontsize=9,
                         fontweight="bold")

    # Quadrant labels
    xlim = ax_scatter.get_xlim(); ylim = ax_scatter.get_ylim()
    ax_scatter.text(0.97, 0.97, "domain-general", transform=ax_scatter.transAxes,
                    ha="right", va="top", fontsize=7, color="#2ca02c", alpha=0.8)
    ax_scatter.text(0.97, 0.03, "food-specific", transform=ax_scatter.transAxes,
                    ha="right", va="bottom", fontsize=7, color="#ff7f0e", alpha=0.8)
    ax_scatter.text(0.03, 0.97, "agri-specific", transform=ax_scatter.transAxes,
                    ha="left", va="top", fontsize=7, color="#1f77b4", alpha=0.8)

    # Legend
    from matplotlib.patches import Patch
    ax_scatter.legend(handles=[
        Patch(color="#2ca02c", label=f"Shared top-{top_k}"),
        Patch(color="#ff7f0e", label=f"Food-only top-{top_k}"),
        Patch(color="#1f77b4", label=f"Agri-only top-{top_k}"),
        Patch(color="#cccccc", label="Low both"),
    ], fontsize=7, loc="lower right")

    # ── Panel 2: word discriminability food vs agri ───────────────────────────
    colors_w = ["#ff7f0e" if r > 0 else "#1f77b4" for r in lograts]
    y_pos    = np.arange(len(words))
    ax_words.barh(y_pos, lograts, color=colors_w, edgecolor="none", height=0.75)
    ax_words.set_yticks(y_pos)
    ax_words.set_yticklabels(words, fontsize=7.5)
    ax_words.axvline(0, color="black", lw=0.8)
    ax_words.set_xlabel("log₂(freq_food_top / freq_agri_top)", fontsize=8)
    ax_words.set_title("Word discriminability\nfood vs agriculture top prompts",
                       fontsize=9, fontweight="bold")
    ax_words.invert_yaxis()
    ax_words.legend(handles=[
        Patch(color="#ff7f0e", label="Food-domain words"),
        Patch(color="#1f77b4", label="Agri-domain words"),
    ], fontsize=7, loc="lower right")

    # ── Panel 3: example prompts per group ───────────────────────────────────
    ax_venn.axis("off")

    def sample_prompts(indices, n=4):
        # pick by highest avg score
        ranked = sorted(indices, key=lambda i: (s_food[i] + s_agri[i]) / 2, reverse=True)
        return [prompts[i].replace("{}", "{c}") for i in ranked[:n]]

    groups = [
        ("Domain-general", "#2ca02c", shared),
        ("Food-specific",  "#ff7f0e", food_only),
        ("Agri-specific",  "#1f77b4", agri_only),
    ]

    y = 0.97
    for label, color, idx in groups:
        ax_venn.text(0.0, y, f"{label}  (n={len(idx)})",
                     transform=ax_venn.transAxes,
                     fontsize=8.5, fontweight="bold", color=color, va="top")
        y -= 0.06
        for p in sample_prompts(idx, 4):
            short = p[:48] + ("…" if len(p) > 48 else "")
            ax_venn.text(0.03, y, f"· {short}",
                         transform=ax_venn.transAxes,
                         fontsize=6.5, va="top", color="#333333")
            y -= 0.055
        y -= 0.03

    ax_venn.set_title(f"Sample prompts by domain alignment\n(top-{top_k} per domain)",
                      fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Cross-domain prompt analysis: food vs agriculture  ({model}, Pool-{pool})",
        fontsize=11, fontweight="bold", y=1.01,
    )
    _save(fig, f"fig7_cross_domain_{model}_{pool}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="siglip-so400m", choices=ALL_MODELS)
    p.add_argument("--dataset", nargs="+", default=ALL_DATASETS, dest="datasets")
    p.add_argument("--pool",    default="247")
    p.add_argument("--fig",     nargs="+", type=int, default=list(range(1, 8)),
                   help="Figures to generate (1-7)")
    p.add_argument("--topk",    type=int, default=10, help="K for top/bottom table")
    return p.parse_args()


def main():
    args = parse_args()
    figs = set(args.fig)
    print(f"Model: {args.model}  |  Pool: {args.pool}  |  "
          f"Datasets: {args.datasets}")

    if 1 in figs:
        print("\n[Fig 1] ZPE score vs accuracy…")
        fig_score_vs_accuracy(args.model, args.datasets, args.pool)

    if 2 in figs:
        print("\n[Fig 2] Top/Bottom prompt tables…")
        for dataset in args.datasets:
            fig_top_bottom_prompts(args.model, dataset, args.pool, K=args.topk)

    if 3 in figs:
        print("\n[Fig 3] Word discriminability…")
        fig_word_discriminability(args.model, args.datasets, args.pool,
                                  models=ALL_MODELS)

    if 4 in figs:
        print("\n[Fig 4] Prompt length vs score…")
        fig_length_vs_score(args.model, args.datasets, args.pool)

    if 5 in figs:
        print("\n[Fig 5] Pool diversity vs disagreement quality…")
        fig_diversity_vs_disagreement(ALL_MODELS, args.datasets)

    if 6 in figs:
        print("\n[Fig 6] Score stability across datasets…")
        fig_score_stability(args.model, args.datasets, args.pool)

    if 7 in figs:
        print("\n[Fig 7] Cross-domain: food vs agriculture…")
        fig_cross_domain(args.model, args.pool)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
