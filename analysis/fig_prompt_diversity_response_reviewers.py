"""
Figure: single-prompt accuracy distributions per dataset — reviewer response.

Shows that pool-247 contains both high-quality and low-quality prompts across
all datasets (Reviewer #2, Concern #3). Each panel is one dataset; violins show
the distribution of per-prompt accuracy over all 247 prompts; horizontal lines
mark the uniform ensemble and the best ZPE-norm ensemble.

Usage
-----
    python analysis/fig_prompt_diversity.py
    python analysis/fig_prompt_diversity.py --pool 426
    python analysis/fig_prompt_diversity.py --models siglip-so400m clip-vit-large
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMP_DIR = Path("results/comprehensive")
FIG_DIR  = Path("results/figures/reviewer")

DATASETS = ["beans", "agriculture", "food11", "food101_full"]
DS_LABELS = {
    "beans":       "Beans",
    "agriculture": "Agriculture",
    "food11":      "Food-11",
    "food101_full":"Food-101",
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


def load_data(model: str, dataset: str, pool: str) -> dict | None:
    path = COMP_DIR / f"{model}__{dataset}__{pool}.jsonl"
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    records = [json.loads(l) for l in lines]
    singles = [r for r in records if r["mode"] == "single"]
    ens     = [r for r in records if r["mode"] == "ensemble"]
    if not singles:
        return None
    accs = np.array([r["accuracy"] for r in singles])
    uniform = next((r["accuracy"] for r in ens if r.get("method") == "uniform"), None)
    zpe_norm = next(
        (r["accuracy"] for r in ens if r.get("method") == "zpe_norm_best_tau"), None
    )
    return {"accs": accs, "uniform": uniform, "zpe_norm": zpe_norm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool",   default="247", choices=["247", "426", "247+agri", "247+food"])
    ap.add_argument("--models", nargs="+", default=["siglip-so400m", "clip-vit-large"],
                    choices=ALL_MODELS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models  = args.models
    n_ds    = len(DATASETS)
    n_mod   = len(models)

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
        1, n_ds,
        figsize=(2.6 * n_ds, 3.4),
        sharey=False,
    )
    if n_ds == 1:
        axes = [axes]

    x_positions = np.arange(1, n_mod + 1)

    for col, dataset in enumerate(DATASETS):
        ax = axes[col]

        for x_pos, model in zip(x_positions, models):
            data = load_data(model, dataset, args.pool)
            if data is None:
                continue

            accs    = data["accs"]
            color   = MODEL_COLOR[model]

            # Violin
            parts = ax.violinplot(
                accs,
                positions=[x_pos],
                widths=0.7,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("none")
                pc.set_alpha(0.55)

            # IQR box
            q25, q50, q75 = np.percentile(accs, [25, 50, 75])
            ax.vlines(x_pos, q25, q75, color=color, linewidth=2.5, zorder=3)
            ax.scatter([x_pos], [q50], color=color, s=18, zorder=4)

            # Min / max ticks
            ax.hlines([accs.min(), accs.max()], x_pos - 0.15, x_pos + 0.15,
                      color=color, linewidth=1.2, zorder=3)

            # Uniform line
            if data["uniform"] is not None:
                ax.hlines(data["uniform"], x_pos - 0.32, x_pos + 0.32,
                          colors="#555555", linewidths=1.2, linestyles="--", zorder=5)

            # ZPE-norm best-τ line
            if data["zpe_norm"] is not None:
                ax.hlines(data["zpe_norm"], x_pos - 0.32, x_pos + 0.32,
                          colors="#111111", linewidths=1.6, linestyles="-", zorder=5)

        ax.set_title(DS_LABELS[dataset], fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [MODEL_LABEL[m].replace(" ", "\n") for m in models],
            fontsize=6.5,
        )
        ax.set_xlim(0.3, n_mod + 0.7)

        if col == 0:
            ax.set_ylabel("Single-prompt accuracy", fontsize=8)

        # Annotate range for the first model (siglip-so400m or first available)
        for x_pos, model in zip(x_positions, models):
            data = load_data(model, dataset, args.pool)
            if data is None:
                continue
            rng = data["accs"].max() - data["accs"].min()
            ymax = data["accs"].max()
            ax.text(x_pos, ymax + 0.005, f"Δ={rng:.2f}",
                    ha="center", va="bottom", fontsize=6, color="#444")

    # Legend
    violin_handles = [
        mpatches.Patch(facecolor=MODEL_COLOR[m], alpha=0.7, label=MODEL_LABEL[m])
        for m in models
    ]
    line_handles = [
        Line2D([0], [0], color="#555555", lw=1.2, ls="--", label="Uniform ensemble"),
        Line2D([0], [0], color="#111111", lw=1.6, ls="-",  label="ZPE-norm (best τ)"),
        Line2D([0], [0], color="gray",    lw=2.5, ls="-",  label="IQR (box) + median"),
    ]
    fig.legend(
        handles=violin_handles + line_handles,
        loc="lower center",
        ncol=len(models) + 3,
        fontsize=6.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(
        f"Distribution of single-prompt accuracy — pool-{args.pool} ({247 if args.pool=='247' else args.pool} prompts)",
        fontsize=9, y=1.01,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models_tag = "_".join(m.replace("/", "-") for m in models)
    out = args.out or str(FIG_DIR / f"fig_prompt_diversity_{args.pool}_{models_tag}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved → {out}")

    # Also save PNG for quick inspection
    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
