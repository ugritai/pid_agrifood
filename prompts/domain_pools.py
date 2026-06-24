"""
Domain-specific prompt pools for agrifood zero-shot classification.

Two pools:
  POOL_FOOD  – optimised for food/dish recognition (food101, food101_full, food11)
  POOL_AGRI  – optimised for agricultural crop and plant recognition (agriculture, beans)

Design principles:
  - Templates cover multiple visual angles: appearance, context, preparation, texture
  - Include domain vocabulary that CLIP/SigLIP text encoders recognise
  - Avoid templates that favour spurious concepts (e.g. "a photo of a {}" already
    exists in pool-247; these add complementary signal)
  - Each template must contain exactly one {} placeholder for the class name
"""

# ─────────────────────────────────────────────────────────────────────────────
# Food pool
# ─────────────────────────────────────────────────────────────────────────────

POOL_FOOD: list[str] = [
    # Appearance / presentation
    "a photo of {}, a type of food.",
    "a close-up photo of a dish of {}.",
    "a plate of {}.",
    "a bowl of {}.",
    "a serving of {}.",
    "a photo of a prepared {}.",
    "a photo of freshly cooked {}.",
    "a photo of homemade {}.",
    "a restaurant plate of {}.",
    "a photo of {} on a white plate.",
    "a photo of {} on a wooden table.",
    "a top-down photo of a dish of {}.",
    "an overhead view of {}.",
    "a macro photo of {}.",
    # Culinary context
    "a delicious serving of {}.",
    "a gourmet plate of {}.",
    "street food: {}.",
    "traditional {} dish.",
    "a classic recipe of {}.",
    "a photo of {} as a main course.",
    "a photo of {} as a side dish.",
    "a photo of {} as a dessert.",
    "a photo of {} as a snack.",
    "a photo of {} as a breakfast item.",
    # Texture / ingredient focus
    "a close-up of the texture of {}.",
    "a detailed photo showing the ingredients of {}.",
    "a cross-section of {}.",
    "a photo of sliced {}.",
    "a photo of {} showing its inside.",
    # Preparation states
    "raw {} ingredients.",
    "freshly prepared {}.",
    "a photo of baked {}.",
    "a photo of fried {}.",
    "a photo of grilled {}.",
    "a photo of steamed {}.",
    "a photo of roasted {}.",
    "a photo of boiled {}.",
    # Food category anchoring
    "this is {} food.",
    "an image of the food known as {}.",
    "a food photo of {}.",
    "this food is called {}.",
    "a culinary photo of {}.",
    "a photo of {} cuisine.",
    "a photo of a {} meal.",
    "a photo of {} as eaten in a restaurant.",
    # Negative-space / contrast prompts
    "a bad photo of the food {}.",
    "a blurry photo of {}.",
    "a dark photo of the dish {}.",
    "a bright photo of a plate of {}.",
    "a high-resolution photo of {}.",
    "a low-quality photo of {}.",
]

# ─────────────────────────────────────────────────────────────────────────────
# Agricultural / crop pool
# ─────────────────────────────────────────────────────────────────────────────

POOL_AGRI: list[str] = [
    # Plant / crop identity
    "a photo of a {} plant.",
    "a photo of {} crops.",
    "a photo of a {} crop.",
    "an agricultural photo of {}.",
    "a photo of a {} field.",
    "a photo of {} growing in a field.",
    "a photo of a {} farm.",
    "a photo of {} produce.",
    "a photo of a {} harvest.",
    "a photo of {} vegetation.",
    "a photo of the {} plant.",
    "a photo of {} plants in a row.",
    # Botanical focus
    "a close-up photo of a {} leaf.",
    "a photo of the leaves of {}.",
    "a close-up of {} foliage.",
    "a photo of a {} stem.",
    "a photo of {} roots.",
    "a photo of {} seeds.",
    "a photo of a {} flower.",
    "a photo of {} fruit.",
    "a photo of {} grain.",
    "a macro photo of a {} leaf.",
    # Disease / health (relevant for beans)
    "a photo of a {} plant showing disease symptoms.",
    "a photo of a diseased {} leaf.",
    "a photo of a healthy {} plant.",
    "a photo of {} with visible infection.",
    "a close-up photo of {} leaf spots.",
    "a photo of {} showing rust disease.",
    # Growth stage
    "a photo of a {} seedling.",
    "a photo of a young {} plant.",
    "a photo of a mature {} plant.",
    "a photo of a {} plant at harvest stage.",
    "a photo of a {} sprout.",
    # Context / setting
    "an aerial view of a {} field.",
    "a satellite image of {} crops.",
    "a photo of {} in a greenhouse.",
    "a photo of {} in a tropical climate.",
    "a photo of {} in a dry climate.",
    "a photo of {} in an irrigated field.",
    "a photo taken on a farm showing {}.",
    # Produce / post-harvest
    "a photo of harvested {}.",
    "a photo of fresh {} produce.",
    "a photo of dried {}.",
    "a photo of {} kernels.",
    "a photo of a pile of {} seeds.",
    "a photo of raw {} commodities.",
    # Anchoring
    "this is a {} crop.",
    "this plant is {}.",
    "this is the {} plant.",
    "an image used to identify {} in agriculture.",
    "a botanical image of {}.",
    "a photo of {} as classified in agronomy.",
]


if __name__ == "__main__":
    print(f"POOL_FOOD : {len(POOL_FOOD)} prompts")
    print(f"POOL_AGRI : {len(POOL_AGRI)} prompts")
    print("\nSample FOOD:")
    for p in POOL_FOOD[:5]:
        print(f"  {p!r}")
    print("\nSample AGRI:")
    for p in POOL_AGRI[:5]:
        print(f"  {p!r}")
