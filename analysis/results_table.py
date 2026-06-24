"""
Results table for ZPE study — full datasets only.

Loads pre-computed JSONL files and produces a comparison table across:
  - Single-prompt:  mean ± std, best, worst  (over all prompts in the pool)
  - Uniform ensemble
  - ZPE-raw         (best τ by accuracy)
  - ZPE-norm        (best τ by accuracy)

Metrics reported: Accuracy, F1-macro, NLL, Brier, ECE

Usage
-----
    python analysis/results_table.py                     # all models/datasets/pools
    python analysis/results_table.py --pool 247          # one pool
    python analysis/results_table.py --dataset food101_full beans
    python analysis/results_table.py --latex             # print LaTeX tabular
    python analysis/results_table.py --csv results/tables/main.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from itertools import product as iproduct

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

COMP_DIR = Path("results/comprehensive")

ALL_MODELS = [
    "siglip-so400m",
    "clip-vit-large",
]

ALL_DATASETS = ["food101_full", "food11", "agriculture", "beans"]

# Pools valid per dataset
DATASET_POOLS = {
    "food101_full": ["247", "426", "food"],
    "food11":       ["247", "426", "food"],
    "agriculture":  ["247", "426", "agri"],
    "beans":        ["247", "426", "agri"],
}

# For figures: pool key that is "domain-specific" for each dataset
DOMAIN_POOL = {
    "food101_full": "food",
    "food11":       "food",
    "agriculture":  "agri",
    "beans":        "agri",
}

DATASET_LABELS = {
    "food101_full": "Food-101",
    "food11":       "Food-11",
    "agriculture":  "Agriculture",
    "beans":        "Beans",
}

METRICS = ["accuracy", "f1_macro", "nll", "brier", "ece"]
METRIC_FMT = {
    "accuracy":   "{:.4f}",
    "f1_macro":   "{:.4f}",
    "nll":        "{:.4f}",
    "brier":      "{:.4f}",
    "ece":        "{:.4f}",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(model: str, dataset: str, pool: str) -> list[dict]:
    path = COMP_DIR / f"{model}__{dataset}__{pool}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines()]


# ─────────────────────────────────────────────────────────────────────────────
# Row extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_rows(model: str, dataset: str, pool: str) -> list[dict]:
    records = load_jsonl(model, dataset, pool)
    if not records:
        return []

    base = dict(model=model, dataset=dataset, pool=pool)

    rows: list[dict] = []

    single = [r for r in records if r["mode"] == "single"]
    ensemble = [r for r in records if r["mode"] == "ensemble"]

    # ── Single-prompt statistics ──────────────────────────────────────────────
    if single:
        for metric in METRICS:
            vals = np.array([r[metric] for r in single if metric in r])
            if len(vals) == 0:
                continue
            # collect into arrays keyed by metric (done below per-metric)

        # Build per-metric stats
        stat_row_mean = {**base, "method": "single_mean"}
        stat_row_std  = {**base, "method": "single_std"}
        stat_row_best = {**base, "method": "single_best"}
        stat_row_worst= {**base, "method": "single_worst"}

        best_rec  = max(single, key=lambda r: r["accuracy"])
        worst_rec = min(single, key=lambda r: r["accuracy"])

        for metric in METRICS:
            vals = np.array([r[metric] for r in single if metric in r])
            stat_row_mean[metric] = float(vals.mean())
            stat_row_std[metric]  = float(vals.std())
            stat_row_best[metric] = float(best_rec[metric]) if metric in best_rec else float("nan")
            stat_row_worst[metric]= float(worst_rec[metric]) if metric in worst_rec else float("nan")

        # Best prompt text
        stat_row_best["prompt"]  = best_rec.get("prompt", "")
        stat_row_worst["prompt"] = worst_rec.get("prompt", "")
        stat_row_mean["n_prompts"] = len(single)
        stat_row_best["n_prompts"] = len(single)

        rows += [stat_row_mean, stat_row_std, stat_row_best, stat_row_worst]

    # ── Ensemble methods ──────────────────────────────────────────────────────

    # Uniform
    uniform = next((r for r in ensemble if r.get("method") == "uniform"), None)
    if uniform:
        row = {**base, "method": "uniform"}
        for metric in METRICS:
            row[metric] = uniform.get(metric, float("nan"))
        rows.append(row)

    # ZPE-raw and ZPE-norm: pick best τ by accuracy
    for zpe_key in ("zpe_raw", "zpe_norm"):
        candidates = [r for r in ensemble if r.get("method") == zpe_key]
        if not candidates:
            # fall back to pre-computed best_tau record
            best_tau_rec = next(
                (r for r in ensemble if r.get("method") == f"{zpe_key}_best_tau"), None
            )
            if best_tau_rec:
                row = {**base, "method": zpe_key,
                       "temperature": best_tau_rec.get("temperature")}
                for metric in METRICS:
                    row[metric] = best_tau_rec.get(metric, float("nan"))
                rows.append(row)
        else:
            best = max(candidates, key=lambda r: r.get("accuracy", -1))
            row = {**base, "method": zpe_key, "temperature": best.get("temperature")}
            for metric in METRICS:
                row[metric] = best.get(metric, float("nan"))
            rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Build full DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_dataframe(
    models: list[str],
    datasets: list[str],
    pools: list[str] | None = None,
) -> pd.DataFrame:
    all_rows = []
    for model, dataset in iproduct(models, datasets):
        valid_pools = DATASET_POOLS.get(dataset, [])
        for pool in valid_pools:
            if pools and pool not in pools:
                continue
            all_rows.extend(extract_rows(model, dataset, pool))

    if not all_rows:
        print("No data found.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

METHOD_ORDER = ["uniform", "zpe_raw", "zpe_norm"]

METHOD_LABELS = {
    "single_best":  "Single (best)",
    "single_mean":  "Single (mean)",
    "single_std":   "Single (std)",
    "single_worst": "Single (worst)",
    "uniform":      "Uniform",
    "zpe_raw":      "ZPE-raw",
    "zpe_norm":     "ZPE-norm",
}


def print_console_table(df: pd.DataFrame, metric: str = "accuracy"):
    """Print a compact console table: rows = model × method, cols = dataset."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    datasets = [d for d in ALL_DATASETS if d in df["dataset"].values]
    models   = [m for m in ALL_MODELS   if m in df["model"].values]
    pools    = sorted(df["pool"].unique())

    for pool in pools:
        sub = df[df["pool"] == pool]
        print(f"\n{'═'*100}")
        print(f"  Pool: {pool}   Metric: {metric}")
        print(f"{'═'*100}")

        header = ["Model", "Method"] + [DATASET_LABELS.get(d, d) for d in datasets]
        table_rows = []

        for model in models:
            for method in METHOD_ORDER:
                row_vals = [model, METHOD_LABELS.get(method, method)]
                for dataset in datasets:
                    cell = sub[
                        (sub["model"] == model) &
                        (sub["dataset"] == dataset) &
                        (sub["method"] == method)
                    ]
                    if cell.empty or metric not in cell.columns:
                        row_vals.append("—")
                    else:
                        val = cell.iloc[0][metric]
                        if np.isnan(val):
                            row_vals.append("—")
                        else:
                            row_vals.append(METRIC_FMT[metric].format(val))
                table_rows.append(row_vals)
            table_rows.append([""] * len(header))  # blank separator between models

        if use_tabulate:
            print(tabulate(table_rows, headers=header, tablefmt="simple"))
        else:
            col_w = [max(len(str(r[i])) for r in [header] + table_rows)
                     for i in range(len(header))]
            fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
            print(fmt.format(*header))
            print("  ".join("-" * w for w in col_w))
            for r in table_rows:
                print(fmt.format(*r))


def print_latex_table(df: pd.DataFrame, pool: str = "247"):
    """
    LaTeX table: rows = model × method, col groups = dataset.
    Reports Accuracy / ECE / NLL for each dataset.
    Only the 4 key methods: single_best, single_mean±std, uniform, zpe_raw, zpe_norm.
    """
    sub = df[df["pool"] == pool].copy()
    datasets = [d for d in ALL_DATASETS if d in sub["dataset"].values]
    models   = [m for m in ALL_MODELS   if m in sub["model"].values]
    show_metrics = ["accuracy", "ece", "nll"]

    n_ds = len(datasets)
    n_m  = len(show_metrics)

    # Header
    col_spec = "ll" + ("".join(["r" * n_m + "|"] * n_ds)).rstrip("|")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{" + col_spec + r"}")
    print(r"\toprule")

    # Dataset headers (multicolumn)
    ds_header = " & " * 2
    ds_header += " & ".join(
        rf"\multicolumn{{{n_m}}}{{c}}{{{DATASET_LABELS.get(d, d)}}}"
        for d in datasets
    )
    print(ds_header + r" \\")

    # Cmidrules
    cmidrules = []
    for i, _ in enumerate(datasets):
        start = 3 + i * n_m
        end   = start + n_m - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    print(" ".join(cmidrules))

    # Metric sub-header
    metric_labels = {"accuracy": "Acc", "ece": "ECE", "nll": "NLL"}
    sub_header = "Model & Method"
    for _ in datasets:
        for m in show_metrics:
            sub_header += " & " + metric_labels[m]
    print(sub_header + r" \\")
    print(r"\midrule")

    key_methods = ["uniform", "zpe_raw", "zpe_norm"]

    for mi, model in enumerate(models):
        if mi > 0:
            print(r"\midrule")
        model_label = (model.replace("clip-vit-", "CLIP-ViT-")
                            .replace("siglip-", "SigLIP-")
                            .replace("b32", "B/32").replace("b16", "B/16")
                            .replace("large", "L").replace("base", "B")
                            .replace("so400m", "SO/400M"))
        for ki, method in enumerate(key_methods):
            method_label = METHOD_LABELS.get(method, method)
            # For single_mean, append ± std in the accuracy cell specially
            row_str = model_label if ki == 0 else ""
            row_str += " & " + method_label

            for dataset in datasets:
                cell = sub[
                    (sub["model"] == model) &
                    (sub["dataset"] == dataset) &
                    (sub["method"] == method)
                ]
                for metric in show_metrics:
                    if cell.empty:
                        row_str += " & —"
                        continue
                    val = cell.iloc[0].get(metric, float("nan"))
                    if isinstance(val, float) and np.isnan(val):
                        row_str += " & —"
                        continue
                    # For mean accuracy, append ± std
                    if method == "single_mean" and metric == "accuracy":
                        std_cell = sub[
                            (sub["model"] == model) &
                            (sub["dataset"] == dataset) &
                            (sub["method"] == "single_std")
                        ]
                        if not std_cell.empty:
                            std_val = std_cell.iloc[0].get(metric, float("nan"))
                            row_str += rf" & {val:.4f}\small{{$\pm${std_val:.4f}}}"
                        else:
                            row_str += f" & {val:.4f}"
                    else:
                        row_str += f" & {val:.4f}"

            print(row_str + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Zero-shot classification results on full datasets. "
          r"Pool-" + pool + r". "
          r"Single (best): best individual prompt; "
          r"Single (mean): average over all prompts ($\pm$ std); "
          r"Uniform: unweighted ensemble; "
          r"ZPE-raw/norm: ZPE-weighted ensemble at best~$\tau$.}")
    print(r"\label{tab:main_results}")
    print(r"\end{table*}")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration figures
# ─────────────────────────────────────────────────────────────────────────────

def save_calibration_figures(df: pd.DataFrame, style: str = "lines"):
    """
    style: "bars"     — mean bars (no errbar), anonymous dots per model
           "bars_err" — mean bars with errbar + dots
           "lines"    — dot-lines per model (color) × method (linestyle): shows both
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig_dir = Path("results/figures/calibration")
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASET_LABELS = {
        "food101_full": "Food-101",
        "food11":       "Food-11",
        "agriculture":  "Agriculture",
        "beans":        "Beans",
    }
    MODEL_COLORS = {
        "siglip-base":    "#1f77b4",
        "siglip-so400m":  "#ff7f0e",
        "siglip-large":   "#2ca02c",
        "clip-vit-b32":   "#9467bd",
        "clip-vit-b16":   "#8c564b",
        "clip-vit-large": "#e377c2",
    }
    MODEL_LABELS = {
        "siglip-base":    "SigLIP-B",
        "siglip-so400m":  "SigLIP-SO400M",
        "siglip-large":   "SigLIP-L",
        "clip-vit-b32":   "CLIP-B/32",
        "clip-vit-b16":   "CLIP-B/16",
        "clip-vit-large": "CLIP-L",
    }
    METHOD_COLORS = {"uniform": "#4c72b0", "zpe_raw": "#dd8452", "zpe_norm": "#55a868"}
    METHOD_LABELS = {"uniform": "Uniform",  "zpe_raw": "ZPE-raw", "zpe_norm": "ZPE-norm"}
    METHOD_LS     = {"uniform": ":",        "zpe_raw": "--",       "zpe_norm": "-"}
    METHOD_MK     = {"uniform": "^",        "zpe_raw": "s",        "zpe_norm": "o"}
    METRICS_PLOT  = [
        ("ece",      "ECE ↓"),
        ("brier",    "Brier ↓"),
        ("nll",      "NLL ↓"),
        ("accuracy", "Accuracy ↑"),
    ]
    ALL_POOLS_ORD     = ["247", "426", "food", "agri"]
    POOL_LABELS_SHORT = {
        "247": "247", "426": "426", "food": "Food", "agri": "Agri",
    }

    methods = ["uniform", "zpe_raw", "zpe_norm"]
    sub = df[df["method"].isin(methods)].copy()
    if sub.empty:
        print("  No calibration data"); return

    datasets    = [d for d in ALL_DATASETS if d in sub["dataset"].unique()]
    models      = [m for m in ALL_MODELS   if m in sub["model"].unique()]
    pools_by_ds = {ds: [p for p in ALL_POOLS_ORD
                        if not sub[(sub["dataset"]==ds) & (sub["pool"]==p)].empty]
                   for ds in datasets}
    n_metrics = len(METRICS_PLOT)
    n_ds      = len(datasets)

    fig, axes = plt.subplots(n_metrics, n_ds,
                             figsize=(n_ds * 3.8, n_metrics * 2.8),
                             squeeze=False, sharey="row")
    rng = np.random.default_rng(0)

    for m_idx, (metric, mlabel) in enumerate(METRICS_PLOT):
        for d_idx, ds in enumerate(datasets):
            ax     = axes[m_idx][d_idx]
            ds_pools = pools_by_ds[ds]
            x        = np.arange(len(ds_pools))

            if style in ("bars", "bars_err"):
                w = 0.22
                for i, method in enumerate(methods):
                    means, stds = [], []
                    for p in ds_pools:
                        vals = [float(sub[(sub["dataset"]==ds) & (sub["pool"]==p) &
                                         (sub["model"]==model) & (sub["method"]==method)
                                        ][metric].iloc[0])
                                for model in models
                                if not sub[(sub["dataset"]==ds) & (sub["pool"]==p) &
                                           (sub["model"]==model) & (sub["method"]==method)].empty]
                        means.append(np.mean(vals) if vals else np.nan)
                        stds.append(np.std(vals)   if vals else np.nan)
                    offset = (i - 1) * w
                    ax.bar(x + offset, means, width=w,
                           yerr=stds if style == "bars_err" else None, capsize=2,
                           label=METHOD_LABELS[method],
                           color=METHOD_COLORS[method], alpha=0.80,
                           error_kw={"linewidth": 0.8})
                    for p_idx, p in enumerate(ds_pools):
                        for model in models:
                            row = sub[(sub["dataset"]==ds) & (sub["pool"]==p) &
                                      (sub["model"]==model) & (sub["method"]==method)]
                            if not row.empty:
                                ax.plot(p_idx + offset + rng.uniform(-0.04, 0.04),
                                        float(row[metric].iloc[0]),
                                        ".", color="k", markersize=3, alpha=0.45, zorder=5)

            elif style == "lines":
                for model in models:
                    for method in methods:
                        vals = []
                        for p in ds_pools:
                            row = sub[(sub["dataset"]==ds) & (sub["pool"]==p) &
                                      (sub["model"]==model) & (sub["method"]==method)]
                            vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
                        ax.plot(x, vals,
                                color=MODEL_COLORS[model],
                                linestyle=METHOD_LS[method],
                                marker=METHOD_MK[method],
                                markersize=4, linewidth=1.1, alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels([POOL_LABELS_SHORT[p] for p in ds_pools], fontsize=7.5)
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(labelsize=7)
            if m_idx == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=9)
            if d_idx == 0:
                ax.set_ylabel(mlabel, fontsize=8)

    if style == "lines":
        model_h  = [Line2D([0],[0], color=MODEL_COLORS[m], linewidth=1.5,
                            marker="o", markersize=4, label=MODEL_LABELS[m])
                    for m in models]
        method_h = [Line2D([0],[0], color="k", linewidth=1.2,
                            linestyle=METHOD_LS[m], marker=METHOD_MK[m],
                            markersize=4, label=METHOD_LABELS[m])
                    for m in methods]
        fig.legend(handles=model_h + method_h,
                   loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        axes[0][0].legend(fontsize=7)
        plt.tight_layout()

    fig.suptitle("Calibration — all pools  (Uniform vs ZPE-raw vs ZPE-norm)", fontsize=10)
    for ext in ("pdf", "png"):
        plt.savefig(fig_dir / f"fig_calibration_{style}.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → results/figures/calibration/fig_calibration_{style}.pdf")


def save_calibration_heatmaps(df: pd.DataFrame):
    """Heatmap grid: rows=models, cols=pools, value annotated inside each cell.
    One figure per method, layout = metrics × datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig_dir = Path("results/figures/calibration")
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASET_LABELS = {"food101_full": "Food-101", "food11": "Food-11",
                      "agriculture": "Agriculture", "beans": "Beans"}
    MODEL_LABELS   = {"siglip-base": "SigLIP-B", "siglip-so400m": "SigLIP-SO400M",
                      "siglip-large": "SigLIP-L", "clip-vit-b32": "CLIP-B/32",
                      "clip-vit-b16": "CLIP-B/16", "clip-vit-large": "CLIP-L"}
    METRICS_PLOT   = [("ece", "ECE ↓", True), ("brier", "Brier ↓", True),
                      ("nll", "NLL ↓", True), ("accuracy", "Accuracy ↑", False)]
    ALL_POOLS_ORD  = ["247", "426", "food", "agri"]
    POOL_LABELS_S  = {"247": "247", "426": "426", "food": "Food", "agri": "Agri"}
    methods = ["uniform", "zpe_raw", "zpe_norm"]
    METHOD_TITLES  = {"uniform": "Uniform", "zpe_raw": "ZPE-raw", "zpe_norm": "ZPE-norm"}

    sub      = df[df["method"].isin(methods)].copy()
    datasets = [d for d in ALL_DATASETS if d in sub["dataset"].unique()]
    models   = [m for m in ALL_MODELS   if m in sub["model"].unique()]
    pools_by_ds = {ds: [p for p in ALL_POOLS_ORD
                        if not sub[(sub["dataset"]==ds) & (sub["pool"]==p)].empty]
                   for ds in datasets}

    for method in methods:
        n_metrics = len(METRICS_PLOT)
        n_ds      = len(datasets)
        fig, axes = plt.subplots(n_metrics, n_ds,
                                 figsize=(n_ds * 3.5, n_metrics * 2.2),
                                 squeeze=False)

        for m_idx, (metric, mlabel, lower_better) in enumerate(METRICS_PLOT):
            # compute vmin/vmax across all datasets for shared colorscale
            all_vals = []
            for ds in datasets:
                for p in pools_by_ds[ds]:
                    for model in models:
                        row = sub[(sub["dataset"]==ds) & (sub["pool"]==p) &
                                  (sub["model"]==model) & (sub["method"]==method)]
                        if not row.empty:
                            all_vals.append(float(row[metric].iloc[0]))
            vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
            cmap = "RdYlGn_r" if lower_better else "RdYlGn"

            for d_idx, ds in enumerate(datasets):
                ax = axes[m_idx][d_idx]
                ds_pools = pools_by_ds[ds]
                n_pools  = len(ds_pools)
                n_models = len(models)

                mat = np.full((n_models, n_pools), np.nan)
                for pi, p in enumerate(ds_pools):
                    for mi, model in enumerate(models):
                        row = sub[(sub["dataset"]==ds) & (sub["pool"]==p) &
                                  (sub["model"]==model) & (sub["method"]==method)]
                        if not row.empty:
                            mat[mi, pi] = float(row[metric].iloc[0])

                im = ax.imshow(mat, aspect="auto", cmap=cmap,
                               vmin=vmin, vmax=vmax)

                # annotate values
                for mi in range(n_models):
                    for pi in range(n_pools):
                        v = mat[mi, pi]
                        if not np.isnan(v):
                            # pick text color for contrast
                            norm_v = (v - vmin) / (vmax - vmin + 1e-9)
                            txt_color = "white" if (norm_v < 0.25 or norm_v > 0.75) else "black"
                            ax.text(pi, mi, f"{v:.3f}", ha="center", va="center",
                                    fontsize=8.5, color=txt_color, fontweight="bold")

                ax.set_xticks(range(n_pools))
                ax.set_xticklabels([POOL_LABELS_S[p] for p in ds_pools], fontsize=7.5)
                ax.set_yticks(range(n_models))
                ax.set_yticklabels([MODEL_LABELS[m] for m in models] if d_idx == 0
                                   else [], fontsize=7.5)
                if m_idx == 0:
                    ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=9)
                if d_idx == n_ds - 1:
                    ax.set_ylabel(mlabel, fontsize=8, labelpad=4)
                    ax.yaxis.set_label_position("right")

        fig.suptitle(f"Calibration heatmap — {METHOD_TITLES[method]}", fontsize=11)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(fig_dir / f"fig_calibration_heatmap_{method}.{ext}",
                        dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → results/figures/calibration/fig_calibration_heatmap_{method}.pdf")


def save_calibration_comparison(df: pd.DataFrame, pool: str = "247"):
    """Single heatmap per pool per metric.
    rows = models, cols = datasets × methods (3 adjacent cols per dataset).
    Separators between dataset groups make method comparison direct."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig_dir = Path("results/figures/calibration")
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASET_LABELS = {"food101_full": "Food-101", "food11": "Food-11",
                      "agriculture": "Agriculture", "beans": "Beans"}
    MODEL_LABELS   = {"siglip-base": "SigLIP-B", "siglip-so400m": "SigLIP-SO400M",
                      "siglip-large": "SigLIP-L", "clip-vit-b32": "CLIP-B/32",
                      "clip-vit-b16": "CLIP-B/16", "clip-vit-large": "CLIP-L"}
    METRICS_PLOT   = [("ece", "ECE ↓", True), ("brier", "Brier ↓", True),
                      ("nll", "NLL ↓", True), ("accuracy", "Accuracy ↑", False)]
    methods        = ["uniform", "zpe_raw", "zpe_norm"]
    METHOD_LABELS  = {"uniform": "Uniform", "zpe_raw": "ZPE-raw", "zpe_norm": "ZPE-norm"}

    sub = df[(df["pool"] == pool) & (df["method"].isin(methods))].copy()
    if sub.empty:
        return

    datasets = [d for d in ALL_DATASETS if d in sub["dataset"].unique()]
    models   = [m for m in ALL_MODELS   if m in sub["model"].unique()]
    if not datasets or not models:
        return

    n_mod      = len(models)
    n_ds       = len(datasets)
    n_meth     = len(methods)
    n_metrics  = len(METRICS_PLOT)
    n_cols     = n_ds * n_meth        # cols: datasets × methods
    n_rows     = n_metrics * n_mod    # rows: metrics × models

    # build full matrix — each metric block is n_mod rows
    mat_full   = np.full((n_rows, n_cols), np.nan)
    cmap_rows  = []   # cmap per metric block
    vmin_rows  = []
    vmax_rows  = []

    for m_idx, (metric, mlabel, lower_better) in enumerate(METRICS_PLOT):
        block = np.full((n_mod, n_cols), np.nan)
        for d_idx, ds in enumerate(datasets):
            for t_idx, method in enumerate(methods):
                col = d_idx * n_meth + t_idx
                for mi, model in enumerate(models):
                    row = sub[(sub["dataset"]==ds) & (sub["model"]==model) &
                              (sub["method"]==method)]
                    if not row.empty:
                        block[mi, col] = float(row[metric].iloc[0])
        mat_full[m_idx * n_mod:(m_idx + 1) * n_mod, :] = block
        vmin_rows.append(np.nanmin(block))
        vmax_rows.append(np.nanmax(block))
        cmap_rows.append("RdYlGn_r" if lower_better else "RdYlGn")

    fig, ax = plt.subplots(figsize=(n_cols * 1.15 + 2.0, n_rows * 0.52 + 2.0))

    # draw each metric block with its own colorscale
    for m_idx, (metric, mlabel, lower_better) in enumerate(METRICS_PLOT):
        r0  = m_idx * n_mod
        r1  = r0 + n_mod
        blk = mat_full[r0:r1, :]
        vmin, vmax = vmin_rows[m_idx], vmax_rows[m_idx]
        cmap = cmap_rows[m_idx]
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        for mi in range(n_mod):
            for ci in range(n_cols):
                v = blk[mi, ci]
                if not np.isnan(v):
                    color  = mapper.to_rgba(v)
                    rect   = plt.Rectangle([ci - 0.5, r0 + mi - 0.5], 1, 1,
                                           color=color, zorder=1)
                    ax.add_patch(rect)
                    norm_v = (v - vmin) / (vmax - vmin + 1e-9)
                    txt_c  = "white" if (norm_v < 0.25 or norm_v > 0.75) else "black"
                    ax.text(ci, r0 + mi, f"{v:.3f}", ha="center", va="center",
                            fontsize=8.0, color=txt_c, fontweight="bold", zorder=2)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # invert y

    # thick horizontal separators between metric blocks
    for m_idx in range(1, n_metrics):
        ax.axhline(m_idx * n_mod - 0.5, color="white", linewidth=3.5, zorder=3)

    # thin horizontal separators within metric blocks (between models)
    for ri in range(n_rows):
        if ri % n_mod != 0:
            ax.axhline(ri - 0.5, color="white", linewidth=0.5, alpha=0.4, zorder=3)

    # vertical separators between dataset groups
    for d_idx in range(1, n_ds):
        ax.axvline(d_idx * n_meth - 0.5, color="white", linewidth=2.5, zorder=3)

    # x-axis: method labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods] * n_ds,
                       fontsize=8, rotation=30, ha="right")

    # dataset labels above columns
    for d_idx, ds in enumerate(datasets):
        centre = d_idx * n_meth + (n_meth - 1) / 2
        ax.text(centre, -0.9, DATASET_LABELS.get(ds, ds),
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # y-axis: model labels + metric labels
    ytick_pos    = list(range(n_rows))
    ytick_labels = [MODEL_LABELS[m] for m in models] * n_metrics
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=8)

    # metric block labels on the right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([m_idx * n_mod + (n_mod - 1) / 2 for m_idx in range(n_metrics)])
    ax2.set_yticklabels([mlabel for _, mlabel, _ in METRICS_PLOT], fontsize=9, fontweight="bold")
    ax2.tick_params(length=0)

    ax.set_title(f"Method comparison — Pool-{pool}  (green = better)",
                 fontsize=11, pad=20)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(fig_dir / f"fig_calibration_compare_{pool}.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → results/figures/calibration/fig_calibration_compare_{pool}.pdf")


def save_calibration_slope(df: pd.DataFrame, pool: str = "247"):
    """Slope graph: x = methods, y = metric, one line per model.
    Shows how each model changes Uniform → ZPE-raw → ZPE-norm.
    Layout: rows = metrics, cols = datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig_dir = Path("results/figures/calibration")
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASET_LABELS = {"food101_full": "Food-101", "food11": "Food-11",
                      "agriculture": "Agriculture", "beans": "Beans"}
    MODEL_COLORS   = {"siglip-base": "#1f77b4", "siglip-so400m": "#ff7f0e",
                      "siglip-large": "#2ca02c", "clip-vit-b32": "#9467bd",
                      "clip-vit-b16": "#8c564b", "clip-vit-large": "#e377c2"}
    MODEL_LABELS   = {"siglip-base": "SigLIP-B", "siglip-so400m": "SigLIP-SO400M",
                      "siglip-large": "SigLIP-L", "clip-vit-b32": "CLIP-B/32",
                      "clip-vit-b16": "CLIP-B/16", "clip-vit-large": "CLIP-L"}
    METRICS_PLOT   = [("ece", "ECE ↓"), ("brier", "Brier ↓"),
                      ("nll", "NLL ↓"), ("accuracy", "Accuracy ↑")]
    methods        = ["uniform", "zpe_raw", "zpe_norm"]
    METHOD_LABELS  = ["Uniform", "ZPE-raw", "ZPE-norm"]

    all_pools = ["247", "426", "food", "agri"]
    POOL_LABELS = {"247": "Pool-247", "426": "Pool-426",
                   "food": "Pool-Food", "agri": "Pool-Agri"}

    sub = df[df["method"].isin(methods)].copy()
    if sub.empty:
        return

    datasets      = [d for d in ALL_DATASETS if d in sub["dataset"].unique()]
    models        = [m for m in ALL_MODELS   if m in sub["model"].unique()]
    pools_present = [p for p in all_pools    if p in sub["pool"].unique()]
    if not datasets or not models:
        return

    x = np.arange(len(methods))

    # Row groups: same layout as figA/figC — no blank cells
    ROW_GROUPS = [
        ("Pool-247",         [("247",  "food101_full"), ("247",  "food11"),
                              ("247",  "agriculture"),  ("247",  "beans")]),
        ("Pool-426",         [("426",  "food101_full"), ("426",  "food11"),
                              ("426",  "agriculture"),  ("426",  "beans")]),
        ("Domain pool",      [("food", "food101_full"), ("food", "food11"),
                              ("agri", "agriculture"),  ("agri", "beans")]),
    ]
    N_ROWS = len(ROW_GROUPS)
    N_COLS = 4

    # one figure per metric, shared y-scale across all subplots
    for metric, mlabel in METRICS_PLOT:
        # compute data range first
        all_vals = [float(sub[(sub["pool"]==pool) & (sub["dataset"]==ds) &
                               (sub["model"]==model) & (sub["method"]==method)][metric].iloc[0])
                    for _, cells in ROW_GROUPS
                    for pool, ds in cells
                    for model in models
                    for method in methods
                    if not sub[(sub["pool"]==pool) & (sub["dataset"]==ds) &
                                (sub["model"]==model) & (sub["method"]==method)].empty]
        if not all_vals:
            continue
        pad  = (max(all_vals) - min(all_vals)) * 0.08
        ymin = min(all_vals) - pad
        ymax = max(all_vals) + pad

        fig, axes = plt.subplots(N_ROWS, N_COLS,
                                 figsize=(N_COLS * 2.6, N_ROWS * 2.2),
                                 squeeze=False, sharey=True)

        for r_idx, (row_label, cells) in enumerate(ROW_GROUPS):
            for c_idx, (pool, ds) in enumerate(cells):
                ax       = axes[r_idx][c_idx]
                sub_cell = sub[(sub["pool"]==pool) & (sub["dataset"]==ds)]
                if sub_cell.empty:
                    ax.set_visible(False); continue

                for model in models:
                    vals = [float(sub_cell[(sub_cell["model"]==model) &
                                          (sub_cell["method"]==method)][metric].iloc[0])
                            if not sub_cell[(sub_cell["model"]==model) &
                                            (sub_cell["method"]==method)].empty
                            else np.nan
                            for method in methods]
                    ax.plot(x, vals, color=MODEL_COLORS[model],
                            marker="o", markersize=4, linewidth=1.4)

                ax.set_ylim(ymin, ymax)
                ax.set_xticks(x)
                ax.set_xticklabels(METHOD_LABELS, fontsize=7.5, rotation=20, ha="right")
                ax.grid(axis="y", alpha=0.3)
                ax.tick_params(labelsize=7)
                if r_idx == 0:
                    ax.set_title(DATASET_LABELS.get(ds, ds), fontsize=9)
                if c_idx == 0:
                    ax.set_ylabel(row_label, fontsize=7.5)

        legend_h = [Line2D([0],[0], color=MODEL_COLORS[m], linewidth=1.5,
                            marker="o", markersize=4, label=MODEL_LABELS[m])
                    for m in models]
        fig.legend(handles=legend_h, loc="lower center", ncol=3,
                   fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Uniform → ZPE-raw → ZPE-norm  —  {mlabel}", fontsize=11)
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        safe = metric.replace("+", "p")
        for ext in ("pdf", "png"):
            plt.savefig(fig_dir / f"fig_calibration_slope_all_{safe}.{ext}",
                        dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → results/figures/calibration/fig_calibration_slope_all_{safe}.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Paper figures
# ─────────────────────────────────────────────────────────────────────────────

def save_paper_figures(df: pd.DataFrame):
    """Publication-ready figure: fig_paper_results.pdf

    Layout: 4 rows (metrics) × 4 cols (datasets).
    Each panel: slope chart, x = methods (Uniform, ZPE, ZPE-norm).
    Color = model, line style = pool. Y shared per row.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D

    fig_dir = Path("results/figures/paper")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,
    })

    DS_LABELS = {
        "food101_full": "Food-101", "food11": "Food-11",
        "agriculture":  "Agriculture", "beans": "Beans",
    }
    METRICS = [
        ("accuracy", "Accuracy ↑", False),
        ("ece",      "ECE ↓",      True),
        ("nll",      "NLL ↓",      True),
        ("brier",    "Brier ↓",    True),
    ]
    MODEL_COLOR  = {"siglip-so400m": "#d95f02", "clip-vit-large": "#1b7ab3"}
    MODEL_MARKER = {"siglip-so400m": "o",        "clip-vit-large": "s"}
    MODEL_LABEL  = {"siglip-so400m": "SigLIP SO400M", "clip-vit-large": "CLIP ViT-L/14"}
    POOL_LS      = {"domain": "-",  "247": "--", "426": ":"}
    POOL_LW      = {"domain": 2.0,  "247": 1.4,  "426": 1.4}
    POOL_LABEL   = {"domain": "Domain pool", "247": "Pool-247", "426": "Pool-426"}

    METHODS     = ["uniform", "zpe_raw", "zpe_norm"]
    METH_LABELS = ["Uniform", "ZPE", "ZPE-norm"]
    X           = np.arange(len(METHODS))

    sub      = df[df["method"].isin(METHODS)].copy()
    models   = [m for m in ALL_MODELS   if m in sub["model"].unique()]
    datasets = [d for d in ALL_DATASETS if d in sub["dataset"].unique()]

    def val(model, ds, pool_key, method, metric):
        r = sub[(sub["model"] == model) & (sub["dataset"] == ds) &
                (sub["pool"]  == pool_key) & (sub["method"] == method)]
        return float(r[metric].iloc[0]) if not r.empty else np.nan

    def resolve_pool(pn, ds):
        return DOMAIN_POOL[ds] if pn == "domain" else pn

    n_rows, n_cols = len(METRICS), len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.9, n_rows * 2.1),
                             squeeze=False, sharey="row")
    fig.subplots_adjust(hspace=0.15, wspace=0.22)

    for r_idx, (metric, mlabel, _) in enumerate(METRICS):
        all_vals = [val(m, ds, resolve_pool(pn, ds), meth, metric)
                    for m in models for ds in datasets
                    for pn in ("247", "426", "domain") for meth in METHODS]
        all_vals = [v for v in all_vals if not np.isnan(v)]
        pad = (max(all_vals) - min(all_vals)) * 0.12
        ylo, yhi = min(all_vals) - pad, max(all_vals) + pad
        fmt = "%.3f" if metric in ("ece", "nll", "brier") else "%.2f"

        for c_idx, ds in enumerate(datasets):
            ax = axes[r_idx][c_idx]
            for model in models:
                for pn in ("247", "426", "domain"):
                    ys = [val(model, ds, resolve_pool(pn, ds), m, metric) for m in METHODS]
                    if all(np.isnan(v) for v in ys):
                        continue
                    ax.plot(X, ys,
                            color=MODEL_COLOR[model],
                            linestyle=POOL_LS[pn],
                            linewidth=POOL_LW[pn],
                            marker=MODEL_MARKER[model],
                            markersize=4, alpha=0.9)

            ax.set_xticks(X)
            ax.set_xticklabels(METH_LABELS, fontsize=9, rotation=0, ha="center")
            ax.set_ylim(ylo, yhi)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
            ax.yaxis.set_tick_params(labelsize=8)
            ax.tick_params(axis="x", labelsize=9)
            if r_idx == 0:
                ax.set_title(DS_LABELS.get(ds, ds), fontsize=11,
                             fontweight="bold", pad=5)
            if c_idx == 0:
                ax.set_ylabel(mlabel, fontsize=10)

    # ── Legend ─────────────────────────────────────────────────────────────────
    model_h = [Line2D([0], [0], color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                      markersize=5, linewidth=2.0, label=MODEL_LABEL[m])
               for m in models]
    pool_h  = [Line2D([0], [0], color="gray", linestyle=POOL_LS[pn],
                      linewidth=POOL_LW[pn], label=POOL_LABEL[pn])
               for pn in ("247", "426", "domain")]
    fig.legend(handles=model_h + pool_h,
               loc="lower center", ncol=len(model_h) + len(pool_h),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(fig_dir / "fig_paper_results.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_paper_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → results/figures/paper/fig_paper_results.pdf")

    plt.rcParams.update(plt.rcParamsDefault)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",   nargs="+", default=ALL_MODELS,  choices=ALL_MODELS)
    p.add_argument("--dataset",  nargs="+", default=ALL_DATASETS, dest="datasets",
                   choices=ALL_DATASETS)
    p.add_argument("--pool",     nargs="+", default=None,
                   choices=["247", "426", "food", "agri", "247+food", "247+agri"])
    p.add_argument("--metric",   default="accuracy", choices=METRICS,
                   help="Metric for the console table")
    p.add_argument("--latex",    action="store_true", help="Print LaTeX table")
    p.add_argument("--latex-pool", default="247",
                   help="Pool to use for the LaTeX table (default: 247)")
    p.add_argument("--csv",      default=None, help="Save full DataFrame to CSV")
    p.add_argument("--figures",       action="store_true", help="Save exploratory calibration figures")
    p.add_argument("--figures-pool",  default="247",
                   help="Pool for calibration figures (default: 247)")
    p.add_argument("--paper-figures", action="store_true", help="Save publication-ready figures")
    return p.parse_args()


def main():
    args = parse_args()
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    print("Loading results…")
    df = build_dataframe(args.models, args.datasets, args.pool)
    if df.empty:
        return

    print(f"Loaded {len(df)} rows  "
          f"({df['model'].nunique()} models × "
          f"{df['dataset'].nunique()} datasets × "
          f"{df['pool'].nunique()} pools × "
          f"{df['method'].nunique()} methods)")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved → {args.csv}")

    if args.figures:
        print(f"\nGenerating calibration figures…")
        save_calibration_figures(df, style="lines")
        save_calibration_figures(df, style="bars")
        save_calibration_figures(df, style="bars_err")
        save_calibration_heatmaps(df)
        for pool in ["247", "426", "food", "agri"]:
            save_calibration_comparison(df, pool=pool)
        save_calibration_slope(df)

    if args.paper_figures:
        print(f"\nGenerating paper figures…")
        save_paper_figures(df)

    if args.latex:
        print(f"\n% LaTeX table — pool={args.latex_pool}\n")
        print_latex_table(df, pool=args.latex_pool)
    else:
        print_console_table(df, metric=args.metric)


if __name__ == "__main__":
    main()
