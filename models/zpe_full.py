"""
ZPE full evaluation — all models × all datasets × all pools.

Saves two files per completed (model, dataset, pool) batch:
  results/comprehensive/<model>__<dataset>__<pool>.jsonl
      One JSON line per mode:
        mode="single"   – one record per prompt (aggregate metrics only)
        mode="ensemble" – one record per method × τ (metrics + per-sample summary)

  results/comprehensive/<model>__<dataset>__<pool>.npz
      Numpy arrays for calibration / reliability analysis:
        true_labels        : (N,)      int32
        class_names        : (C,)      object
        prompts            : (P,)      object
        scores_raw         : (P,)      float32   ZPE Algorithm-1 scores
        scores_norm        : (P,)      float32   ZPE Algorithm-2 scores
        single_pred        : (P, N)    int16     argmax per prompt
        single_conf        : (P, N)    float32   max prob per prompt
        single_true_prob   : (P, N)    float32   prob of true class per prompt
        ens_uniform_probs  : (N, C)    float32
        ens_zpe_raw_probs  : (N, C)    float32   at best τ
        ens_zpe_norm_probs : (N, C)    float32   at best τ

A manifest (results/comprehensive/manifest.json) tracks completed batches so
the script can be interrupted and resumed safely.

Metrics computed
----------------
  accuracy, f1_weighted, f1_macro
  nll    : negative log-likelihood  −mean log p(y_true)
  brier  : multiclass Brier score   mean_i Σ_c (p_ic − 1[c=y_i])²
  ece    : expected calibration error (15 equal-width confidence bins)

Usage
-----
    python models/zpe_full.py                              # all combos
    python models/zpe_full.py --models siglip-so400m       # one model
    python models/zpe_full.py --force                      # re-run everything
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.zpe import MODEL_REGISTRY, load_model, build_text_embeddings, compute_logits
from data.loaders import load_dataset_by_name
from prompts.pool import POOL_247, POOL_426
from prompts.domain_pools import POOL_FOOD, POOL_AGRI

# Combined pools: generic + domain-specific (deduped, order preserved)
POOL_247_FOOD = list(dict.fromkeys(POOL_247 + POOL_FOOD))
POOL_247_AGRI = list(dict.fromkeys(POOL_247 + POOL_AGRI))

# ─────────────────────────────────────────────────────────────────────────────
# Experiment grid
# ─────────────────────────────────────────────────────────────────────────────

ALL_MODELS = list(MODEL_REGISTRY.keys())

# (pool_name, pool, datasets it applies to)
ALL_POOLS = [
    ("247",      POOL_247,      ["beans", "food101", "food101_full", "food11", "agriculture"]),
    ("426",      POOL_426,      ["beans", "food101", "food101_full", "food11", "agriculture"]),
    ("food",     POOL_FOOD,     ["food101", "food101_full", "food11"]),
    ("agri",     POOL_AGRI,     ["beans", "agriculture"]),
    ("247+food", POOL_247_FOOD, ["food101", "food101_full", "food11"]),
    ("247+agri", POOL_247_AGRI, ["beans", "agriculture"]),
]

TAU_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0]

OUT_DIR = Path("results/comprehensive")

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _nll(probs: np.ndarray, true_labels: np.ndarray) -> float:
    """Negative log-likelihood, clipped for numerical stability."""
    p_true = probs[np.arange(len(true_labels)), true_labels]
    return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())


def _brier(probs: np.ndarray, true_labels: np.ndarray) -> float:
    """Multiclass Brier score: mean_i Σ_c (p_ic − 1[c=y_i])²"""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(true_labels)), true_labels] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def _ece(probs: np.ndarray, true_labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == true_labels).astype(float)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece / len(true_labels))


def _accuracy(probs: np.ndarray, true_labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == true_labels).mean())


def _f1(probs: np.ndarray, true_labels: np.ndarray):
    from sklearn.metrics import f1_score
    pred = probs.argmax(axis=1)
    return (
        float(f1_score(true_labels, pred, average="weighted", zero_division=0)),
        float(f1_score(true_labels, pred, average="macro", zero_division=0)),
    )


def all_metrics(probs: np.ndarray, true_labels: np.ndarray) -> dict:
    f1w, f1m = _f1(probs, true_labels)
    return {
        "accuracy":    _accuracy(probs, true_labels),
        "f1_weighted": f1w,
        "f1_macro":    f1m,
        "nll":         _nll(probs, true_labels),
        "brier":       _brier(probs, true_labels),
        "ece":         _ece(probs, true_labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Softmax-weighted ensemble
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_probs(
    logits: torch.Tensor,   # (P, N, C)
    scores: torch.Tensor,   # (P,)
    tau: float,
) -> np.ndarray:            # (N, C)
    weights = F.softmax(scores / tau, dim=0)   # (P,)
    raw = (logits * weights[:, None, None]).sum(0).numpy()  # (N, C)
    # Convert logit-weighted sum to probabilities via softmax
    p = np.exp(raw - raw.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def uniform_probs(logits: torch.Tensor) -> np.ndarray:  # (N, C)
    raw = logits.mean(dim=0).numpy()
    p = np.exp(raw - raw.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Single-prompt probabilities (from cosine-sim logits + softmax)
# ─────────────────────────────────────────────────────────────────────────────

def single_prompt_probs(logits_p: torch.Tensor) -> np.ndarray:
    """logits_p : (N, C)  → (N, C) float32 probabilities via softmax."""
    raw = logits_p.numpy()
    p = np.exp(raw - raw.max(axis=1, keepdims=True))
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ZPE prompt scores
# ─────────────────────────────────────────────────────────────────────────────

def score_raw(logits: torch.Tensor) -> torch.Tensor:
    return logits.max(dim=2).values.mean(dim=1)


def score_norm(logits: torch.Tensor) -> torch.Tensor:
    return (logits - logits.mean(dim=1, keepdim=True)).max(dim=2).values.mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest helpers
# ─────────────────────────────────────────────────────────────────────────────

MANIFEST_PATH = OUT_DIR / "manifest.json"


def load_manifest() -> set[str]:
    if MANIFEST_PATH.exists():
        return set(json.loads(MANIFEST_PATH.read_text()))
    return set()


def save_manifest(done: set[str]):
    MANIFEST_PATH.write_text(json.dumps(sorted(done), indent=2))


def combo_key(model: str, dataset: str, pool: str) -> str:
    return f"{model}|{dataset}|{pool}"


def combo_stem(model: str, dataset: str, pool: str) -> str:
    return f"{model}__{dataset}__{pool}"


# ─────────────────────────────────────────────────────────────────────────────
# Main batch function
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    model_key: str,
    dataset_name: str,
    pool_name: str,
    pool: list[str],
    encoder,
    logit_scale: float,
    img_batch_size: int,
    txt_batch_size: int,
):
    from tqdm import tqdm

    print(f"\n{'═'*70}")
    print(f"  {model_key} | {dataset_name} | pool={pool_name}")
    print(f"{'═'*70}")

    samples, class_names = load_dataset_by_name(dataset_name)
    N = len(samples)
    C = len(class_names)
    P = len(pool)
    print(f"  {N:,} samples | {C} classes | {P} prompts")

    images = [s["image"] for s in samples]
    true_labels = np.array([s["label"] for s in samples], dtype=np.int32)

    # Text embeddings (P, C, D)
    txt_embs = build_text_embeddings(pool, class_names, encoder, txt_batch_size)

    # Image embeddings (N, D)
    print(f"  Encoding {N:,} images…")
    img_embs = encoder.encode_images(images, img_batch_size)

    # Full logit tensor (P, N, C) — kept on CPU to avoid OOM
    print("  Computing logits…")
    logits = compute_logits(img_embs, txt_embs, logit_scale)  # (P, N, C)

    # ZPE scores
    s_raw  = score_raw(logits)   # (P,)
    s_norm = score_norm(logits)  # (P,)

    stem = combo_stem(model_key, dataset_name, pool_name)
    jsonl_path = OUT_DIR / f"{stem}.jsonl"
    npz_path   = OUT_DIR / f"{stem}.npz"

    # ── JSONL records ────────────────────────────────────────────────────────

    meta = {
        "model": model_key,
        "dataset": dataset_name,
        "pool": pool_name,
        "n_prompts": P,
        "n_samples": N,
        "n_classes": C,
        "logit_scale": logit_scale,
    }

    records: list[dict] = []

    # 1. Single-prompt records (per prompt, compact metrics)
    print(f"  Computing single-prompt metrics ({P} prompts)…")
    single_pred      = np.zeros((P, N), dtype=np.int16)
    single_conf      = np.zeros((P, N), dtype=np.float32)
    single_true_prob = np.zeros((P, N), dtype=np.float32)
    single_probs     = np.zeros((P, N, C), dtype=np.float16)   # for PID / D(x)

    for p_idx in tqdm(range(P), desc="  single", leave=False):
        probs_p = single_prompt_probs(logits[p_idx])   # (N, C)
        pred_p = probs_p.argmax(axis=1)
        conf_p = probs_p.max(axis=1)
        tp_p   = probs_p[np.arange(N), true_labels]

        single_pred[p_idx]      = pred_p.astype(np.int16)
        single_conf[p_idx]      = conf_p.astype(np.float32)
        single_true_prob[p_idx] = tp_p.astype(np.float32)
        single_probs[p_idx]     = probs_p.astype(np.float16)

        m = all_metrics(probs_p, true_labels)
        records.append({
            **meta,
            "mode": "single",
            "prompt_idx": p_idx,
            "prompt": pool[p_idx],
            **m,
        })

    # 2. Ensemble records
    print("  Computing ensemble metrics…")

    # Uniform
    p_uni = uniform_probs(logits)
    m = all_metrics(p_uni, true_labels)
    records.append({**meta, "mode": "ensemble", "method": "uniform", "temperature": None, **m})
    print(f"    uniform        acc={m['accuracy']:.4f}  ece={m['ece']:.4f}  nll={m['nll']:.4f}")

    # ZPE raw + norm × all τ
    best_raw  = {"accuracy": -1.0}
    best_norm = {"accuracy": -1.0}
    p_raw_best = p_norm_best = None

    for tau in TAU_GRID:
        for score_key, scores, best_ref in [
            ("zpe_raw",  s_raw,  best_raw),
            ("zpe_norm", s_norm, best_norm),
        ]:
            probs_ens = ensemble_probs(logits, scores, tau)
            m = all_metrics(probs_ens, true_labels)
            records.append({
                **meta,
                "mode": "ensemble",
                "method": score_key,
                "temperature": tau,
                **m,
            })
            if m["accuracy"] > best_ref["accuracy"]:
                best_ref.update({**m, "temperature": tau})
                if score_key == "zpe_raw":
                    p_raw_best = probs_ens.copy()
                else:
                    p_norm_best = probs_ens.copy()

    # Best-τ summary records
    for score_key, best_ref in [("zpe_raw", best_raw), ("zpe_norm", best_norm)]:
        records.append({
            **meta,
            "mode": "ensemble",
            "method": f"{score_key}_best_tau",
            "temperature": best_ref["temperature"],
            **{k: v for k, v in best_ref.items() if k != "temperature"},
        })
        print(f"    {score_key+'_best_tau':<18} acc={best_ref['accuracy']:.4f}  "
              f"ece={best_ref['ece']:.4f}  nll={best_ref['nll']:.4f}  "
              f"τ*={best_ref['temperature']}")

    # ── Write JSONL ──────────────────────────────────────────────────────────
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # ── Write NPZ ────────────────────────────────────────────────────────────
    np.savez_compressed(
        npz_path,
        true_labels       = true_labels,
        class_names       = np.array(class_names, dtype=object),
        prompts           = np.array(pool, dtype=object),
        scores_raw        = s_raw.numpy().astype(np.float32),
        scores_norm       = s_norm.numpy().astype(np.float32),
        single_pred       = single_pred,
        single_conf       = single_conf,
        single_true_prob  = single_true_prob,
        single_probs      = single_probs,
        ens_uniform_probs = p_uni,
        ens_zpe_raw_probs = p_raw_best  if p_raw_best  is not None else p_uni,
        ens_zpe_norm_probs= p_norm_best if p_norm_best is not None else p_uni,
    )

    print(f"  Saved → {jsonl_path.name}  ({jsonl_path.stat().st_size//1024:,} KB)")
    print(f"  Saved → {npz_path.name}  ({npz_path.stat().st_size//1024:,} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

ALL_DATASETS_FLAT = sorted({d for _, _, ds in ALL_POOLS for d in ds})


def parse_args():
    p = argparse.ArgumentParser()
    all_pool_names = [name for name, _, _ in ALL_POOLS]
    p.add_argument("--models",   nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--datasets", nargs="+", default=None,       choices=ALL_DATASETS_FLAT,
                   help="Restrict to specific datasets (default: all)")
    p.add_argument("--pools",    nargs="+", default=None,       choices=all_pool_names,
                   help="Restrict to specific pools (default: all)")
    p.add_argument("--force",    action="store_true", help="Re-run completed batches")
    p.add_argument("--img-batch-size", type=int, default=64,  dest="img_batch_size")
    p.add_argument("--txt-batch-size", type=int, default=256, dest="txt_batch_size")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = set() if args.force else load_manifest()
    print(f"Already completed: {len(done)} batches")

    dataset_filter = set(args.datasets) if args.datasets else None
    pool_filter    = set(args.pools)    if args.pools    else None

    # Build full job list
    jobs: list[tuple[str, str, str, list[str]]] = []
    for model_key in args.models:
        for pool_name, pool, datasets in ALL_POOLS:
            if pool_filter and pool_name not in pool_filter:
                continue
            for dataset_name in datasets:
                if dataset_filter and dataset_name not in dataset_filter:
                    continue
                key = combo_key(model_key, dataset_name, pool_name)
                if key not in done:
                    jobs.append((model_key, dataset_name, pool_name, pool))

    print(f"Remaining: {len(jobs)} batches")
    if not jobs:
        print("Nothing to do.")
        return

    # Group by model to load each model once
    from itertools import groupby
    jobs.sort(key=lambda x: x[0])
    for model_key, model_jobs in groupby(jobs, key=lambda x: x[0]):
        model_jobs = list(model_jobs)
        encoder, logit_scale = load_model(model_key, args.device)

        for (mk, dataset_name, pool_name, pool) in model_jobs:
            key = combo_key(mk, dataset_name, pool_name)
            t0 = time.time()
            try:
                run_batch(
                    model_key=mk,
                    dataset_name=dataset_name,
                    pool_name=pool_name,
                    pool=pool,
                    encoder=encoder,
                    logit_scale=logit_scale,
                    img_batch_size=args.img_batch_size,
                    txt_batch_size=args.txt_batch_size,
                )
                done.add(key)
                save_manifest(done)
                print(f"  [{len(done)} done | {time.time()-t0:.0f}s]")
            except Exception as exc:
                print(f"  [ERROR] {key}: {exc}")
                import traceback; traceback.print_exc()

        del encoder
        torch.cuda.empty_cache()

    # Final summary across all completed JSONL files
    print(f"\n{'═'*80}")
    print("SUMMARY — best method per combo (by accuracy)")
    print(f"{'═'*80}")
    print(f"{'Model':<20} {'Dataset':<16} {'Pool':<6} {'Method':<22} "
          f"{'Acc':>7} {'NLL':>7} {'Brier':>7} {'ECE':>7}")
    print(f"{'─'*80}")

    for model_key in args.models:
        for pool_name, _, datasets in ALL_POOLS:
            for dataset_name in datasets:
                key = combo_key(model_key, dataset_name, pool_name)
                if key not in done:
                    continue
                path = OUT_DIR / f"{combo_stem(model_key, dataset_name, pool_name)}.jsonl"
                if not path.exists():
                    continue
                ens_records = [
                    json.loads(l) for l in path.read_text().splitlines()
                    if json.loads(l)["mode"] == "ensemble"
                    and "best_tau" in json.loads(l).get("method", "")
                    or json.loads(l).get("method") == "uniform"
                ]
                if not ens_records:
                    continue
                best = max(ens_records, key=lambda r: r["accuracy"])
                tau_str = f"τ={best['temperature']}" if best.get("temperature") else ""
                print(
                    f"{model_key:<20} {dataset_name:<16} {pool_name:<6} "
                    f"{best['method']:<22} "
                    f"{best['accuracy']:>7.4f} {best['nll']:>7.4f} "
                    f"{best['brier']:>7.4f} {best['ece']:>7.4f}  {tau_str}"
                )


if __name__ == "__main__":
    main()
