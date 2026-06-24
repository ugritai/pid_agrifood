"""
Dataset loaders for agrifood zero-shot experiments.

Each loader returns:
    samples    : list of {"image": PIL.Image, "label": int, "label_text": str}
    class_names: list[str] ordered by label index
"""

import os
import random
from collections import defaultdict

import numpy as np
from PIL import Image

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace datasets
# ─────────────────────────────────────────────────────────────────────────────

def load_beans():
    """
    3-class bean leaf disease dataset (test split, 128 images).

    The original HuggingFace class names ('angular_leaf_spot', 'bean_rust',
    'healthy') are too short and ambiguous for zero-shot VLMs — 'healthy' in
    particular has no visual anchor without domain context, causing collapse to
    'bean_rust'. We use descriptive names that include the subject ('bean leaf')
    and disease context, which substantially improves zero-shot discrimination.
    """
    from datasets import load_dataset as hf_load
    ds = hf_load("beans", split="test")

    # Descriptive names: map original → domain-contextualised
    DESCRIPTIVE = {
        "angular_leaf_spot": "bean leaf with angular leaf spot disease",
        "bean_rust":         "bean leaf with rust disease",
        "healthy":           "healthy bean leaf",
    }
    raw_names   = ds.features["labels"].names
    label_names = [DESCRIPTIVE.get(n, n.replace("_", " ")) for n in raw_names]

    samples = [
        {
            "image":      ex["image"].convert("RGB"),
            "label":      ex["labels"],
            "label_text": label_names[ex["labels"]],
        }
        for ex in ds
    ]
    return samples, label_names


def load_food101(top_k: int = 10, samples_per_class: int = 100):
    """
    Food-101 validation split, stratified to top_k most-represented classes.
    Uses the human-readable class name (underscores replaced with spaces).
    """
    from datasets import load_dataset as hf_load
    ds = hf_load("food101", split="validation")
    label_names = [n.replace("_", " ") for n in ds.features["label"].names]

    by_class: dict[int, list] = defaultdict(list)
    for ex in ds:
        by_class[ex["label"]].append(ex)

    # Top-k by count
    selected = sorted(by_class, key=lambda k: -len(by_class[k]))[:top_k]

    rng = random.Random(SEED)
    samples = []
    for new_idx, old_idx in enumerate(selected):
        pool = by_class[old_idx]
        chosen = rng.sample(pool, min(samples_per_class, len(pool)))
        for ex in chosen:
            samples.append(
                {
                    "image": ex["image"].convert("RGB"),
                    "label": new_idx,
                    "label_text": label_names[old_idx],
                }
            )

    class_names = [label_names[old_idx] for old_idx in selected]
    rng.shuffle(samples)
    return samples, class_names


def load_food101_full(samples_per_class: int = 50):
    """
    Full Food-101 validation split (101 classes), stratified.
    Use this instead of load_food101 when you want all classes.
    """
    from datasets import load_dataset as hf_load
    ds = hf_load("food101", split="validation")
    label_names = [n.replace("_", " ") for n in ds.features["label"].names]

    by_class: dict[int, list] = defaultdict(list)
    for ex in ds:
        by_class[ex["label"]].append(ex)

    rng = random.Random(SEED)
    samples = []
    for cls_idx in sorted(by_class):
        pool = by_class[cls_idx]
        n = samples_per_class if samples_per_class else len(pool)
        chosen = rng.sample(pool, min(n, len(pool)))
        for ex in chosen:
            samples.append(
                {
                    "image": ex["image"].convert("RGB"),
                    "label": cls_idx,
                    "label_text": label_names[cls_idx],
                }
            )
    return samples, label_names


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle datasets (folder-based, image/label_folder/file structure)
# ─────────────────────────────────────────────────────────────────────────────

def _load_folder_dataset(
    root: str,
    samples_per_class: int | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
):
    """Generic loader for datasets with one sub-folder per class."""
    classes = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    rng = random.Random(SEED)
    samples = []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(root, cls_name)
        files = [
            f
            for f in os.listdir(cls_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        if samples_per_class:
            files = rng.sample(files, min(samples_per_class, len(files)))
        for fname in files:
            try:
                img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
                samples.append(
                    {"image": img, "label": cls_idx, "label_text": cls_name}
                )
            except Exception:
                pass
    return samples, classes


def load_food11(samples_per_class: int | None = None):
    """Food-11 from Kaggle (trolukovich/food11-image-dataset)."""
    import kagglehub
    path = kagglehub.dataset_download("trolukovich/food11-image-dataset")
    # Dataset has train/validation/evaluation sub-splits; use evaluation
    eval_dir = os.path.join(path, "evaluation")
    if not os.path.isdir(eval_dir):
        eval_dir = path
    return _load_folder_dataset(eval_dir, samples_per_class)


def load_agriculture(samples_per_class: int | None = None):
    """Agricultural Crops Image Classification from Kaggle (30 crop classes)."""
    import kagglehub
    path = kagglehub.dataset_download(
        "mdwaquarazam/agricultural-crops-image-classification"
    )
    # Dataset root is: <path>/Agricultural-crops/<class>/
    subdir = os.path.join(path, "Agricultural-crops")
    root = subdir if os.path.isdir(subdir) else path
    # Clean up underscores/dashes in class names at load time
    samples, classes = _load_folder_dataset(root, samples_per_class)
    classes = [c.replace("-", " ").replace("_", " ") for c in classes]
    for s in samples:
        s["label_text"] = s["label_text"].replace("-", " ").replace("_", " ")
    return samples, classes


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, callable] = {
    "beans": load_beans,
    "food101": load_food101,
    "food101_full": load_food101_full,
    "food11": load_food11,
    "agriculture": load_agriculture,
}


def load_dataset_by_name(name: str, **kwargs):
    """Load a dataset by name. Returns (samples, class_names)."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](**kwargs)
