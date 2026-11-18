"""Binned views and the frozen train, validation, and test indices.

Kepler is stored as shards because GitHub rejects a file over 100 MB.
The loader concatenates ``part-*.npz`` in name order. TESS is one file.
Indices are stored in the file. They are not recomputed at training time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Cohort:
    name: str
    global_view: np.ndarray
    local_view: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    secondary_view: np.ndarray | None = None
    depth_change: np.ndarray | None = None
    period_days: np.ndarray | None = None
    duration_hours: np.ndarray | None = None
    transit_depth: np.ndarray | None = None
    tce_id: np.ndarray | None = None

    def lengths(self) -> dict[str, int | None]:
        secondary = None if self.secondary_view is None else int(self.secondary_view.shape[1])
        return {
            "global": int(self.global_view.shape[1]),
            "local": int(self.local_view.shape[1]),
            "secondary": secondary,
        }

    def indices(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train_idx
        if split == "val":
            return self.val_idx
        if split == "test":
            return self.test_idx
        raise ValueError(f"Unknown split {split!r}.")


def split_indices(n: int, seed: int = 0, algorithm: str = "randbelow") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 indices.

    ``randbelow`` is Fisher-Yates with ``Random._randbelow``, which is what
    ``random.shuffle`` did on Python 3.10. ``multiply`` is the older
    ``int(random() * (i + 1))`` shuffle. Both are implemented here so the
    result does not depend on the Python version that loads the file.
    """

    rng = random.Random(seed)
    order = list(range(n))
    for i in reversed(range(1, n)):
        if algorithm == "randbelow":
            j = rng._randbelow(i + 1)
        elif algorithm == "multiply":
            j = int(rng.random() * (i + 1))
        else:
            raise ValueError(f"Unknown split algorithm {algorithm!r}.")
        order[i], order[j] = order[j], order[i]
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = np.asarray(order[:n_train], dtype=np.int64)
    val = np.asarray(order[n_train : n_train + n_val], dtype=np.int64)
    test = np.asarray(order[n_train + n_val :], dtype=np.int64)
    return train, val, test


def _channel(view: np.ndarray) -> np.ndarray:
    array = np.asarray(view, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"A view must have shape (examples, bins), not {array.shape}.")
    return array[:, :, None]


def model_inputs(cohort: Cohort, indices: np.ndarray, model: str):
    """Arrays in the order the published networks expect, and the labels."""

    labels = cohort.labels[indices].astype(np.float32)
    global_view = _channel(cohort.global_view[indices])
    local_view = _channel(cohort.local_view[indices])
    if model in {"baseline", "baseline-attention"}:
        parts = [global_view, local_view]
        if cohort.secondary_view is not None:
            parts.append(_channel(cohort.secondary_view[indices]))
        if cohort.depth_change is not None:
            depth = cohort.depth_change[indices].astype(np.float32).reshape(-1, 1, 1)
            parts.append(depth)
        return np.concatenate(parts, axis=1), labels
    views = [global_view, local_view]
    if cohort.name == "tess":
        if cohort.secondary_view is None or cohort.depth_change is None:
            raise ValueError("TESS column models need the secondary view and the depth change.")
        views.append(_channel(cohort.secondary_view[indices]))
        views.append(cohort.depth_change[indices].astype(np.float32).reshape(-1, 1))
    return views, labels


def _concat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts, axis=0)


def load_kepler(directory: str | Path) -> Cohort:
    directory = Path(directory)
    shards = sorted(directory.glob("part-*.npz"))
    if not shards:
        raise FileNotFoundError(f"No Kepler shards in {directory}.")
    globals_, locals_, labels = [], [], []
    periods, durations, depths, ids = [], [], [], []
    train_idx = val_idx = test_idx = None
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            globals_.append(np.array(data["global_view"]))
            locals_.append(np.array(data["local_view"]))
            labels.append(np.array(data["label"]))
            if "period_days" in data:
                periods.append(np.array(data["period_days"]))
                durations.append(np.array(data["duration_hours"]))
                depths.append(np.array(data["transit_depth"]))
                ids.append(np.array(data["tce_id"]))
            if "train_idx" in data:
                train_idx = np.array(data["train_idx"])
                val_idx = np.array(data["val_idx"])
                test_idx = np.array(data["test_idx"])
    if train_idx is None:
        raise FileNotFoundError(f"{directory} has views but no frozen split.")
    return Cohort(
        name="kepler",
        global_view=_concat(globals_),
        local_view=_concat(locals_),
        labels=_concat(labels).astype(np.int8),
        train_idx=np.asarray(train_idx),
        val_idx=np.asarray(val_idx),
        test_idx=np.asarray(test_idx),
        period_days=_concat(periods) if periods else None,
        duration_hours=_concat(durations) if durations else None,
        transit_depth=_concat(depths) if depths else None,
        tce_id=_concat(ids) if ids else None,
    )


def load_tess(path: str | Path) -> Cohort:
    path = Path(path)
    if path.is_dir():
        path = path / "views.npz"
    with np.load(path, allow_pickle=False) as data:
        split = np.array(data["split"])
        return Cohort(
            name="tess",
            global_view=np.array(data["global_view"]),
            local_view=np.array(data["local_view"]),
            secondary_view=np.array(data["secondary_view"]),
            depth_change=np.array(data["depth_change"]),
            labels=np.array(data["label"]).astype(np.int8),
            train_idx=np.flatnonzero(split == 0),
            val_idx=np.flatnonzero(split == 1),
            test_idx=np.flatnonzero(split == 2),
            period_days=np.array(data["period_days"]) if "period_days" in data else None,
            transit_depth=np.array(data["transit_depth"]) if "transit_depth" in data else None,
        )


def load_dataset(name: str, data_dir: str | Path = "data") -> Cohort:
    root = Path(data_dir)
    if name == "kepler":
        return load_kepler(root / "kepler")
    if name == "tess":
        return load_tess(root / "tess")
    raise ValueError(f"Unknown dataset {name!r}.")
