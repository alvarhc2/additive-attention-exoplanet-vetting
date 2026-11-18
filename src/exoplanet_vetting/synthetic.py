"""Short curves with a transit dip, for a smoke run.

A fit on this cohort checks that the code path runs. It is not a stand-in
for the Kepler or TESS views, and its metrics are not the paper's metrics.
"""

from __future__ import annotations

import numpy as np

from exoplanet_vetting.data import Cohort, split_indices


def synthetic_cohort(dataset: str = "kepler", n: int = 48, seed: int = 0) -> Cohort:
    if dataset not in {"kepler", "tess"}:
        raise ValueError(f"Unknown dataset {dataset!r}.")
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=np.int8)
    labels[: max(1, n // 4)] = 1
    rng.shuffle(labels)
    global_bins = 64
    local_bins = 32

    def curve(bins: int, dip: float) -> np.ndarray:
        values = rng.normal(0.0, 0.02, size=(n, bins)).astype(np.float32)
        center = bins // 2
        half = max(1, bins // 16)
        values[labels == 1, center - half : center + half] -= np.float32(dip)
        return values

    secondary = None
    depth = None
    if dataset == "tess":
        secondary = curve(local_bins, dip=0.0)
        # Eclipsing binaries, coded as the negative class, carry a second dip.
        center = local_bins // 2
        secondary[labels == 0, center - 1 : center + 2] -= np.float32(0.25)
        depth = rng.normal(0.0, 1.0, size=n).astype(np.float32)
        depth[labels == 0] += np.float32(0.5)
    train_idx, val_idx, test_idx = split_indices(n, seed=seed)
    return Cohort(
        name=dataset,
        global_view=curve(global_bins, dip=0.35),
        local_view=curve(local_bins, dip=0.55),
        secondary_view=secondary,
        depth_change=depth,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
