from pathlib import Path

import numpy as np
import pytest

from exoplanet_vetting.config import (
    KEPLER_POSITIVES,
    KEPLER_ROWS,
    KEPLER_SPLIT_POSITIVES,
    KEPLER_SPLIT_SIZES,
    TESS_SPLIT_POSITIVES,
    TESS_SPLIT_SIZES,
)
from exoplanet_vetting.data import load_dataset, model_inputs, split_indices


def test_split_sizes_follow_the_paper_fractions():
    train, val, test = split_indices(KEPLER_ROWS, seed=0)
    assert (len(train), len(val), len(test)) == (
        KEPLER_SPLIT_SIZES["train"],
        KEPLER_SPLIT_SIZES["val"],
        KEPLER_SPLIT_SIZES["test"],
    )
    assert len(set(train) | set(val) | set(test)) == KEPLER_ROWS


def test_published_views_match_the_paper_counts():
    root = Path("data")
    if not (root / "kepler").exists() or not (root / "tess" / "views.npz").exists():
        pytest.skip("Published views are not in data/.")
    kepler = load_dataset("kepler", root)
    assert len(kepler.labels) == KEPLER_ROWS
    assert int(kepler.labels.sum()) == KEPLER_POSITIVES
    assert kepler.global_view.shape[1] == 2001
    assert kepler.local_view.shape[1] == 201
    for split, size in KEPLER_SPLIT_SIZES.items():
        assert len(kepler.indices(split)) == size
        assert int(kepler.labels[kepler.indices(split)].sum()) == KEPLER_SPLIT_POSITIVES[split]
    tess = load_dataset("tess", root)
    assert tess.global_view.shape[1] == 201
    assert tess.local_view.shape[1] == 61
    assert tess.secondary_view.shape[1] == 61
    for split, size in TESS_SPLIT_SIZES.items():
        assert len(tess.indices(split)) == size
        assert int(tess.labels[tess.indices(split)].sum()) == TESS_SPLIT_POSITIVES[split]
    views, labels = model_inputs(kepler, kepler.train_idx[:4], "baseline")
    assert views.shape[1:] == (2202, 1)
    assert len(labels) == 4
    columns, _labels = model_inputs(tess, tess.test_idx[:3], "optimized-attention")
    assert [array.shape[1] for array in columns[:3]] == [201, 61, 61]
    assert columns[3].shape == (3, 1)
