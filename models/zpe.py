"""
Zero-shot Prompt Ensembling (ZPE) — unified runner for CLIP and SigLIP.

Replicates Allingham et al., ICML 2023:
  "A Simple Zero-shot Prompt Weighting Technique to Improve Prompt
   Ensembling in Text-Image Models"

Three ensemble methods per (dataset, pool) pair:
  uniform    – equal weight (simple mean baseline)
  zpe_raw    – Algorithm 1: softmax(mean max-logit) weighting
  zpe_norm   – Algorithm 2: mean-centred logits before scoring

Usage
-----
    # SigLIP, all datasets, both pools
    python models/zpe.py --model siglip-so400m --datasets all --pool both

    # CLIP ViT-L/14, same
    python models/zpe.py --model clip-vit-large --datasets all --pool both
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
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loaders import load_dataset_by_name
from prompts.pool import POOL_247, POOL_426

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    # ── SigLIP ──────────────────────────────────────────────────────────────
    "siglip-base": {
        "family": "siglip",
        "hf_id": "google/siglip-base-patch16-224",
    },
    "siglip-large": {
        "family": "siglip",
        "hf_id": "google/siglip-large-patch16-256",
    },
    "siglip-so400m": {
        "family": "siglip",
        "hf_id": "google/siglip-so400m-patch14-384",
    },
    # ── CLIP ────────────────────────────────────────────────────────────────
    "clip-vit-b32": {
        "family": "clip",
        "hf_id": "openai/clip-vit-base-patch32",
    },
    "clip-vit-b16": {
        "family": "clip",
        "hf_id": "openai/clip-vit-base-patch16",
    },
    "clip-vit-large": {
        "family": "clip",
        "hf_id": "openai/clip-vit-large-patch14",
    },
}

ALL_DATASETS = ["beans", "food101", "food101_full", "agriculture", "food11"]

POOLS = {"247": POOL_247, "426": POOL_426}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_key: str, device: str):
    """Returns (encoder, logit_scale) where encoder is a ModelEncoder instance."""
    cfg = MODEL_REGISTRY[model_key]
    hf_id = cfg["hf_id"]
    family = cfg["family"]
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {model_key} ({hf_id})…")

    if family == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        processor = SiglipProcessor.from_pretrained(hf_id)
        model = SiglipModel.from_pretrained(hf_id, dtype=dtype).to(device).eval()
        logit_scale = model.logit_scale.exp().item()
        encoder = _SiglipEncoder(model, processor, device)

    elif family == "clip":
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained(hf_id)
        model = CLIPModel.from_pretrained(hf_id, dtype=dtype).to(device).eval()
        logit_scale = model.logit_scale.exp().item()
        encoder = _CLIPEncoder(model, processor, device)

    else:
        raise ValueError(f"Unknown family '{family}'")

    print(f"  Loaded on {device} ({dtype}) | logit_scale={logit_scale:.3f}")
    return encoder, logit_scale


# ─────────────────────────────────────────────────────────────────────────────
# Encoder wrappers (unified interface)
# ─────────────────────────────────────────────────────────────────────────────

class _SiglipEncoder:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self.device)
            embs = self.model.get_text_features(**inputs).float()
            all_embs.append(F.normalize(embs, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def encode_images(self, images: list, batch_size: int = 64) -> torch.Tensor:
        all_embs = []
        for i in tqdm(range(0, len(images), batch_size), desc="  images", leave=False):
            inputs = self.processor(
                images=images[i : i + batch_size], return_tensors="pt"
            ).to(self.device)
            embs = self.model.get_image_features(**inputs).float()
            all_embs.append(F.normalize(embs, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)


class _CLIPEncoder:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            embs = self.model.get_text_features(**inputs).float()
            all_embs.append(F.normalize(embs, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def encode_images(self, images: list, batch_size: int = 64) -> torch.Tensor:
        all_embs = []
        for i in tqdm(range(0, len(images), batch_size), desc="  images", leave=False):
            inputs = self.processor(
                images=images[i : i + batch_size],
                return_tensors="pt",
            ).to(self.device)
            embs = self.model.get_image_features(**inputs).float()
            all_embs.append(F.normalize(embs, dim=-1).cpu())
        return torch.cat(all_embs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-class text embeddings  →  (P, C, D)
# ─────────────────────────────────────────────────────────────────────────────

def build_text_embeddings(
    pool: list[str],
    class_names: list[str],
    encoder,
    txt_batch_size: int = 256,
) -> torch.Tensor:
    P, C = len(pool), len(class_names)
    flat = [tmpl.format(cls) for tmpl in pool for cls in class_names]
    print(f"  Encoding {P} × {C} = {P * C:,} text embeddings…")
    flat_embs = encoder.encode_texts(flat, batch_size=txt_batch_size)
    return flat_embs.view(P, C, -1)  # (P, C, D)


# ─────────────────────────────────────────────────────────────────────────────
# Logits  (P, N, C)
# ─────────────────────────────────────────────────────────────────────────────

def compute_logits(
    img_embs: torch.Tensor,   # (N, D)
    txt_embs: torch.Tensor,   # (P, C, D)
    logit_scale: float,
) -> torch.Tensor:
    return torch.einsum("nd,pcd->pnc", img_embs, txt_embs) * logit_scale


# ─────────────────────────────────────────────────────────────────────────────
# ZPE scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_raw(logits: torch.Tensor) -> torch.Tensor:
    """Algorithm 1: s_p = mean_i max_c logits[p,i,c]. Shape: (P,)."""
    return logits.max(dim=2).values.mean(dim=1)


def score_norm(logits: torch.Tensor) -> torch.Tensor:
    """Algorithm 2: subtract per-prompt test-set mean before max. Shape: (P,)."""
    return (logits - logits.mean(dim=1, keepdim=True)).max(dim=2).values.mean(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    logits: torch.Tensor,   # (P, N, C)
    scores: torch.Tensor,   # (P,)
    temperature: float,
) -> np.ndarray:
    weights = F.softmax(scores / temperature, dim=0)
    return (logits * weights[:, None, None]).sum(0).argmax(dim=1).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# τ search grid (from paper: {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5})
# extended with higher values for SigLIP's large logit_scale
# ─────────────────────────────────────────────────────────────────────────────

TAU_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, 5.0, 10.0, 20.0, 50.0]


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    dataset_name: str,
    pool: list[str],
    pool_name: str,
    model_key: str,
    encoder,
    logit_scale: float,
    img_batch_size: int = 64,
    txt_batch_size: int = 256,
    dataset_kwargs: dict | None = None,
) -> dict:
    print(f"\n{'─'*65}")
    print(f"  {model_key} | {dataset_name} | pool={pool_name}")
    print(f"{'─'*65}")

    samples, class_names = load_dataset_by_name(dataset_name, **(dataset_kwargs or {}))
    print(f"  {len(samples):,} samples | {len(class_names)} classes")

    images = [s["image"] for s in samples]
    true_labels = np.array([s["label"] for s in samples])

    txt_embs = build_text_embeddings(pool, class_names, encoder, txt_batch_size)

    print(f"  Encoding {len(images):,} images…")
    img_embs = encoder.encode_images(images, img_batch_size)

    print("  Computing logits…")
    logits = compute_logits(img_embs, txt_embs, logit_scale)

    s_raw = score_raw(logits)
    s_norm = score_norm(logits)

    results: dict[str, dict] = {}

    # Uniform
    preds = predict(logits, torch.ones(len(pool)), temperature=1.0)
    results["uniform"] = metrics(preds, true_labels)
    print(f"  {'uniform':<18} acc={results['uniform']['accuracy']:.4f}  "
          f"f1_w={results['uniform']['f1_weighted']:.4f}")

    # ZPE fixed τ=1
    for key, scores in [("zpe_raw", s_raw), ("zpe_norm", s_norm)]:
        preds = predict(logits, scores, temperature=1.0)
        results[key] = metrics(preds, true_labels)
        results[key]["temperature"] = 1.0
        print(f"  {key:<18} acc={results[key]['accuracy']:.4f}  "
              f"f1_w={results[key]['f1_weighted']:.4f}  τ=1.0")

    # ZPE best τ sweep
    for key, scores in [("zpe_raw_best_tau", s_raw), ("zpe_norm_best_tau", s_norm)]:
        best: dict = {"accuracy": -1.0}
        for tau in TAU_GRID:
            preds = predict(logits, scores, tau)
            m = metrics(preds, true_labels)
            if m["accuracy"] > best["accuracy"]:
                best = {**m, "temperature": tau}
        results[key] = best
        print(f"  {key:<18} acc={best['accuracy']:.4f}  "
              f"f1_w={best['f1_weighted']:.4f}  τ*={best['temperature']}")

    return {
        "model": model_key,
        "dataset": dataset_name,
        "pool": pool_name,
        "n_prompts": len(pool),
        "n_samples": len(samples),
        "n_classes": len(class_names),
        "class_names": class_names,
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ZPE with CLIP / SigLIP")
    p.add_argument(
        "--models",
        nargs="+",
        default=["siglip-base"],
        choices=list(MODEL_REGISTRY),
        help="Models to evaluate (default: siglip-base)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["beans"],
        help=f"Datasets or 'all'. Available: {ALL_DATASETS}",
    )
    p.add_argument(
        "--pool",
        choices=["247", "426", "both"],
        default="247",
    )
    p.add_argument("--img-batch-size", type=int, default=64, dest="img_batch_size")
    p.add_argument("--txt-batch-size", type=int, default=256, dest="txt_batch_size")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--out-dir", default="results", dest="out_dir")
    return p.parse_args()


def main():
    args = parse_args()

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets
    pools = (
        [("247", POOL_247), ("426", POOL_426)]
        if args.pool == "both"
        else [(args.pool, POOLS[args.pool])]
    )

    all_results = []
    for model_key in args.models:
        encoder, logit_scale = load_model(model_key, args.device)
        for dataset_name in datasets:
            for pool_name, pool in pools:
                try:
                    res = run_experiment(
                        dataset_name=dataset_name,
                        pool=pool,
                        pool_name=pool_name,
                        model_key=model_key,
                        encoder=encoder,
                        logit_scale=logit_scale,
                        img_batch_size=args.img_batch_size,
                        txt_batch_size=args.txt_batch_size,
                    )
                    all_results.append(res)
                except Exception as exc:
                    print(f"  [SKIP] {dataset_name}: {exc}")

        # Free GPU memory before loading next model
        del encoder
        torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    models_tag = "_".join(args.models)
    out_path = out_dir / f"zpe_{models_tag}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print summary table
    print(f"\n{'─'*80}")
    print(f"{'Model':<20} {'Dataset':<16} {'Pool':<5} "
          f"{'uniform':>8} {'zpe_norm':>9} {'zpe_norm*':>10} {'τ*':>5}")
    print(f"{'─'*80}")
    for r in all_results:
        res = r["results"]
        print(
            f"{r['model']:<20} {r['dataset']:<16} {r['pool']:<5} "
            f"{res['uniform']['accuracy']:>8.4f} "
            f"{res['zpe_norm']['accuracy']:>9.4f} "
            f"{res['zpe_norm_best_tau']['accuracy']:>10.4f} "
            f"{res['zpe_norm_best_tau']['temperature']:>5.1f}"
        )


if __name__ == "__main__":
    main()
