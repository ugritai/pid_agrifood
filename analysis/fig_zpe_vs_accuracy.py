"""
Figure: scatter plot of per-prompt ZPE score vs single-prompt accuracy.

Each point is one prompt.  Columns = datasets, rows = models (or a single
panel when --model / --dataset are given).  A Pearson-r annotation is shown
per panel.

Usage
-----
    python analysis/fig_zpe_vs_accuracy.py
    python analysis/fig_zpe_vs_accuracy.py --pool 426
    python analysis/fig_zpe_vs_accuracy.py --models siglip-so400m clip-vit-large
    python analysis/fig_zpe_vs_accuracy.py --zpe norm   # use scores_norm instead of scores_raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMP_DIR = Path("results/comprehensive")
FIG_DIR  = Path("results/figures/reviewer")

DATASETS = ["beans", "agriculture", "food11", "food101_full"]
DS_LABELS = {
    "beans":        "Beans",
    "agriculture":  "Agriculture",
    "food11":       "Food-11",
    "food101_full": "Food-101",
}

ALL_MODELS = [
    "siglip-base", "siglip-large", "siglip-so400m",
    "clip-vit-b32", "clip-vit-b16", "clip-vit-large",
]

MODEL_COLOR = {
    "siglip-so400m":  "#d95f02",
    "siglip-large":   "#f4a563",
    "siglip-base":    "#fdd0a2",
    "clip-vit-large": "#1b7ab3",
    "clip-vit-b16":   "#74b9d4",
    "clip-vit-b32":   "#c6dbef",
}
MODEL_LABEL = {
    "siglip-so400m":  "SigLIP SO400M",
    "siglip-large":   "SigLIP Large",
    "siglip-base":    "SigLIP Base",
    "clip-vit-large": "CLIP ViT-L/14",
    "clip-vit-b16":   "CLIP ViT-B/16",
    "clip-vit-b32":   "CLIP ViT-B/32",
}


def load_scatter_data(model: str, dataset: str, pool: str, zpe_key: str) -> dict | None:
    path = COMP_DIR / f"{model}__{dataset}__{pool}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    zpe_scores = data[zpe_key].astype(float)           # (n_prompts,)
    true_labels = data["true_labels"]                  # (n_samples,)
    single_pred = data["single_pred"].astype(int)      # (n_prompts, n_samples)
    accuracy = (single_pred == true_labels[None, :]).mean(axis=1)  # (n_prompts,)
    return {"zpe": zpe_scores, "accuracy": accuracy}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool",   default="247", choices=["247", "426", "247+agri", "247+food"])
    ap.add_argument("--models", nargs="+", default=["siglip-so400m", "clip-vit-large"],
                    choices=ALL_MODELS)
    ap.add_argument("--zpe",   default="raw", choices=["raw", "norm"],
                    help="Which ZPE score to use: 'raw' = scores_raw, 'norm' = scores_norm")
    ap.add_argument("--out",   default=None)
    args = ap.parse_args()

    zpe_key   = "scores_raw" if args.zpe == "raw" else "scores_norm"
    zpe_label = "ZPE score (raw)" if args.zpe == "raw" else "ZPE score (normalised)"

    models = args.models
    n_mod  = len(models)
    n_ds   = len(DATASETS)

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,
    })

    fig, axes = plt.subplots(
        n_mod, n_ds,
        figsize=(2.8 * n_ds, 2.6 * n_mod),
        squeeze=False,
    )

    for row, model in enumerate(models):
        color = MODEL_COLOR[model]
        for col, dataset in enumerate(DATASETS):
            ax = axes[row][col]
            data = load_scatter_data(model, dataset, args.pool, zpe_key)

            if data is None:
                ax.set_visible(False)
                continue

            zpe = data["zpe"]
            acc = data["accuracy"]

            ax.scatter(zpe, acc, color=color, alpha=0.35, s=8, linewidths=0,
                       rasterized=True)

            # Regression line
            slope, intercept, r, p, _ = stats.linregress(zpe, acc)
            x_line = np.linspace(zpe.min(), zpe.max(), 200)
            ax.plot(x_line, slope * x_line + intercept, color=color,
                    linewidth=1.4, alpha=0.9)

            # Pearson-r annotation
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            ax.text(0.97, 0.05, f"r = {r:.2f}{star}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color=color)

            if row == 0:
                ax.set_title(DS_LABELS[dataset], fontsize=9, fontweight="bold", pad=4)
            if col == 0:
                ax.set_ylabel(f"{MODEL_LABEL[model]}\nAccuracy", fontsize=7)
            if row == n_mod - 1:
                ax.set_xlabel(zpe_label, fontsize=7)

    fig.suptitle(
        f"ZPE score vs single-prompt accuracy — pool-{args.pool}",
        fontsize=9, y=1.01,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models_tag = "_".join(m.replace("/", "-") for m in models)
    out = args.out or str(FIG_DIR / f"fig_zpe_vs_accuracy_{args.pool}_{args.zpe}_{models_tag}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved → {out}")

    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
