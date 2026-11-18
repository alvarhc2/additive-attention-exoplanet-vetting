"""Published schedules, view lengths, and parameter counts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    """One stretch of Adam updates at a fixed learning rate."""

    epochs: int
    learning_rate: float
    batch_size: int


# The training scripts pass epsilon=1e-8. The paper text quotes the
# TensorFlow default, 1e-7. These runs follow the scripts.
ADAM_EPSILON = 1e-8
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999

LENGTHS = {
    "kepler": {"global": 2001, "local": 201, "secondary": None},
    "tess": {"global": 201, "local": 61, "secondary": 61},
}

# Mean of ten initializations. Baseline models read one concatenated sequence.
SCHEDULES: dict[tuple[str, str], tuple[Stage, ...]] = {
    ("kepler", "baseline"): (Stage(150, 1e-4, 16),),
    ("kepler", "baseline-attention"): (Stage(150, 1e-4, 16),),
    ("kepler", "optimized"): (Stage(25, 1e-4, 64), Stage(25, 1e-5, 64)),
    ("kepler", "optimized-attention"): (Stage(25, 1e-4, 64), Stage(25, 1e-5, 64)),
    ("kepler", "astronet"): (Stage(50, 1e-5, 64),),
    ("tess", "baseline"): (Stage(150, 1e-4, 16),),
    ("tess", "baseline-attention"): (Stage(150, 1e-4, 16),),
    ("tess", "optimized"): (Stage(20, 1e-4, 64),),
    ("tess", "optimized-attention"): (Stage(20, 1e-4, 64),),
    ("tess", "astronet"): (Stage(51, 1e-5, 64),),
}

MODELS = (
    "baseline",
    "baseline-attention",
    "optimized",
    "optimized-attention",
    "astronet",
)

DATASETS = ("kepler", "tess")

# Counts reported for the published view lengths. Baseline is 14 parameters
# on either mission. Attention adds one bias per time step of its input.
PAPER_PARAMETERS = {
    ("kepler", "baseline"): 14,
    ("kepler", "baseline-attention"): 2219,
    ("kepler", "optimized"): 2_751_853,
    ("kepler", "optimized-attention"): 2_754_173,
    ("kepler", "astronet"): 8_793_313,
    ("tess", "baseline"): 14,
    ("tess", "baseline-attention"): 341,
    ("tess", "optimized"): 3_338_679,
    ("tess", "optimized-attention"): 3_339_154,
    ("tess", "astronet"): 2_216_593,
}

# Positive labels inside the frozen 80/10/10 Kepler split (seed 0).
KEPLER_SPLIT_POSITIVES = {"train": 2906, "val": 328, "test": 366}
KEPLER_SPLIT_SIZES = {"train": 12589, "val": 1573, "test": 1575}
KEPLER_POSITIVES = 3600
KEPLER_ROWS = 15737

TESS_SPLIT_POSITIVES = {"train": 349, "val": 57, "test": 48}
TESS_SPLIT_SIZES = {"train": 11573, "val": 1465, "test": 1442}


def paper_schedule(dataset: str, model: str) -> tuple[Stage, ...]:
    try:
        return SCHEDULES[(dataset, model)]
    except KeyError as error:
        raise KeyError(f"No published schedule for {dataset} / {model}.") from error
