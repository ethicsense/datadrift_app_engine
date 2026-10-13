"""Frozen protocol for the attribute-conditioned ranking experiment.

Date: 2026-10-12

Prompt lists, palette RGB, split ratios, α/SVD grid, and success limits are
fixed before any metric is computed. CIELAB values are derived from these RGB
anchors at runtime and are used only for evaluation.
"""

from __future__ import annotations

DATE = "2026-10-12"
SEED = 42
TRAIN_FRAC = 0.60
DEV_FRAC = 0.20
NEIGHBOR_K = 10
BOOTSTRAP = 2000
SVD_RANKS = (8, 16)
ALPHAS = (0.25, 0.50, 0.75, 1.00)
TAU_DE = 18.0
NEAR_DE = 12.0
FAR_DE = 35.0
TRIPLET_PAIRS = 16
MIN_CLASS_N = 12
NON_TARGET_DROP_LIMIT = 0.02
MODEL_ID = "Marqo/marqo-fashionSigLIP"
SOURCE_COHORT = "artifacts/garment_segmentation/attribute_cluster_eval-20261007"
SOURCE_EMBEDDING = (
    "artifacts/garment_segmentation/soft_background_methods-20261008-n2k/"
    "embeddings/background_methods/pixelate_p060.npy"
)
SOURCE_PREPROCESS = "배경 픽셀화 24px · pad 6%"
PROJECT_ID = "color_control-20261012"
SAM3_RUNS = (
    "sam3-mps-450-expanded",
    "sam3-mps-supp211",
    "sam3-mps-expand293-recovered",
    "sam3-mps-topup49",
    "sam3-mps-topup36",
    "sam3-mps-topup28",
    "sam3-mps-add1k-recovered",
)

# Evaluation-only sRGB anchors. Not used to build text prompts or subspaces.
PALETTE_RGB = {
    "white": (245, 245, 242),
    "ivory": (242, 232, 208),
    "black": (18, 18, 18),
    "charcoal": (48, 48, 50),
    "gray": (140, 140, 142),
    "light_gray": (196, 196, 196),
    "navy": (18, 32, 68),
    "blue": (36, 78, 158),
    "sky_blue": (150, 196, 224),
    "red": (188, 32, 42),
    "burgundy": (112, 24, 40),
    "pink": (226, 152, 170),
    "orange": (216, 108, 38),
    "yellow": (228, 196, 48),
    "green": (36, 124, 68),
    "khaki": (168, 156, 96),
    "beige": (214, 192, 160),
    "brown": (118, 78, 48),
    "purple": (108, 48, 138),
}

COLOR_PROMPTS = {
    "black": ["a black top", "a jet black shirt", "a solid black garment"],
    "charcoal": ["a charcoal top", "a dark gray shirt"],
    "gray": ["a gray top", "a heather gray shirt", "a medium gray garment"],
    "white": ["a white top", "a bright white shirt"],
    "ivory": ["an ivory top", "a cream shirt", "an off-white garment"],
    "navy": ["a navy top", "a dark navy shirt"],
    "blue": ["a blue top", "a medium blue shirt"],
    "sky_blue": ["a sky blue top", "a light blue shirt"],
    "red": ["a red top", "a bright red shirt"],
    "burgundy": ["a burgundy top", "a wine red shirt", "a maroon garment"],
    "pink": ["a pink top", "a light pink shirt"],
    "orange": ["an orange top", "a bright orange shirt"],
    "yellow": ["a yellow top", "a mustard shirt"],
    "green": ["a green top", "a forest green shirt"],
    "khaki": ["a khaki top", "an olive shirt"],
    "beige": ["a beige top", "a sand colored shirt"],
    "brown": ["a brown top", "a chocolate brown shirt"],
    "purple": ["a purple top", "a violet shirt"],
}

PATTERN_PROMPTS = {
    "solid": ["a solid plain top", "a top with no pattern", "a plain solid shirt"],
    "stripe": ["a striped top", "a shirt with horizontal stripes", "a pinstripe shirt"],
    "check": ["a checked top", "a plaid shirt", "a gingham shirt"],
    "graphic_logo": ["a graphic print top", "a shirt with a logo", "a printed slogan shirt"],
    "other": ["a floral top", "a dotted shirt", "a camouflage top"],
}

SLEEVE_PROMPTS = {
    "sleeveless": ["a sleeveless top", "a tank top with no sleeves"],
    "short": ["a short sleeve top", "a short sleeved shirt"],
    "long": ["a long sleeve top", "a long sleeved shirt"],
}

NECK_PROMPTS = {
    "round": ["a crew neck top", "a round neck shirt"],
    "vneck": ["a v-neck top", "a v neck shirt"],
    "collar": ["a collared shirt", "a shirt with a pointed collar"],
    "hood": ["a hooded top", "a hoodie"],
    "turtleneck": ["a turtleneck top", "a shirt with a high neck"],
}

FIT_PROMPTS = {
    "slim": ["a slim fit top", "a tight fitting shirt"],
    "regular": ["a regular fit top", "a standard fit shirt"],
    "oversized": ["an oversized top", "a loose boxy shirt"],
}

COARSE_ATTRS = ("pattern", "sleeve", "neck", "color", "fit")
COLOR_NON_TARGETS = ("pattern", "sleeve", "neck")
PATTERN_NON_TARGETS = ("color", "sleeve", "neck")
PROMPT_GROUPS = {
    "color": COLOR_PROMPTS,
    "pattern": PATTERN_PROMPTS,
    "sleeve": SLEEVE_PROMPTS,
    "neck": NECK_PROMPTS,
    "fit": FIT_PROMPTS,
}
# Fit labels stay out of the 2%p budget unless fit itself is the target.
DECISION_ATTRS = ("pattern", "sleeve", "neck", "color")


def project_id(target: str) -> str:
    return f"{target}_control-20261012"


def removal_prompt_names(target: str) -> list[str]:
    names: list[str] = []
    for attr, groups in PROMPT_GROUPS.items():
        if attr == target:
            continue
        names.extend(f"{attr}:{name}" for name in groups)
    return names


def decision_non_targets(target: str) -> tuple[str, ...]:
    if target == "fit":
        return DECISION_ATTRS
    return tuple(attr for attr in DECISION_ATTRS if attr != target)


def as_dict() -> dict:
    return {
        "date": DATE,
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "dev_frac": DEV_FRAC,
        "neighbor_k": NEIGHBOR_K,
        "bootstrap": BOOTSTRAP,
        "svd_ranks": list(SVD_RANKS),
        "alphas": list(ALPHAS),
        "tau_de": TAU_DE,
        "near_de": NEAR_DE,
        "far_de": FAR_DE,
        "triplet_pairs": TRIPLET_PAIRS,
        "min_class_n": MIN_CLASS_N,
        "non_target_drop_limit": NON_TARGET_DROP_LIMIT,
        "model_id": MODEL_ID,
        "source_cohort": SOURCE_COHORT,
        "source_embedding": SOURCE_EMBEDDING,
        "source_preprocess": SOURCE_PREPROCESS,
        "project_id": PROJECT_ID,
        "palette_rgb": PALETTE_RGB,
        "color_prompts": COLOR_PROMPTS,
        "pattern_prompts": PATTERN_PROMPTS,
        "sleeve_prompts": SLEEVE_PROMPTS,
        "neck_prompts": NECK_PROMPTS,
        "fit_prompts": FIT_PROMPTS,
        "success": {
            "coarse_purity_ci_above_zero": True,
            "graded_ndcg_and_weighted_purity_improve": True,
            "mean_delta_e_decreases": True,
            "non_target_mean_drop_at_most": NON_TARGET_DROP_LIMIT,
        },
        "notes": [
            "Palette RGB and garment Lab are evaluation-only.",
            "Text methods do not read image labels.",
            "Data-centroid directions use the train split only.",
            "Alpha and SVD rank are chosen on dev; test is reported after selection.",
            "Base image vectors are the pixelate pad 6% embeddings, not raw A.",
        ],
    }
