"""Turn the study pickles into the float32 files shipped in data/.

Run from the repository root with the scientific environment installed.
The Kepler split is chosen by matching the published positive counts, using
a version-stable shuffle rather than whatever ``random.shuffle`` does on
the current interpreter.
"""

from __future__ import annotations

import argparse
import csv
import io
import pickle
from pathlib import Path

import numpy as np

from exoplanet_vetting.config import (
    KEPLER_ROWS,
    KEPLER_SPLIT_POSITIVES,
    TESS_SPLIT_POSITIVES,
    TESS_SPLIT_SIZES,
)
from exoplanet_vetting.data import split_indices

_MAX_BYTES = 90 * 1024 * 1024


def _numeric(item) -> np.ndarray:
    # TESS examples were pickled as TensorFlow tensors.
    if hasattr(item, "numpy"):
        item = item.numpy()
    return np.asarray(item, dtype=np.float32).reshape(-1)


def _stack(items) -> np.ndarray:
    first = _numeric(items[0])
    if first.ndim != 1:
        raise SystemExit(f"Expected a 1-D view, got shape {first.shape}.")
    out = np.empty((len(items), first.shape[0]), dtype=np.float32)
    out[0] = first
    for index, item in enumerate(items[1:], start=1):
        out[index] = _numeric(item)
    return out


def _labels(items) -> np.ndarray:
    return np.asarray([int(_numeric(item)[0]) for item in items], dtype=np.int8)


def _kepler_catalog(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    header = text.find("loc_rowid,")
    if header < 0:
        raise SystemExit(f"No catalog header in {path}.")
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in csv.DictReader(io.StringIO(text[header:])):
        label = row["av_training_set"]
        if label == "UNK":
            continue
        if label not in {"PC", "AFP", "NTP"}:
            raise SystemExit(f"Unexpected Autovetter label {label!r}.")
        kepid = int(row["kepid"])
        grouped.setdefault(kepid, []).append(row)
    records = []
    for kepid in sorted(grouped):
        for row in grouped[kepid]:
            depth = row["tce_depth"]
            records.append(
                {
                    "tce_id": f"{kepid}-{int(row['tce_plnt_num'])}",
                    "label": 1 if row["av_training_set"] == "PC" else 0,
                    "period_days": float(row["tce_period"]),
                    "duration_hours": float(row["tce_duration"]),
                    "transit_depth": float(depth) if depth else np.nan,
                }
            )
    return records


def _matching_split(labels: np.ndarray):
    target = (
        KEPLER_SPLIT_POSITIVES["train"],
        KEPLER_SPLIT_POSITIVES["val"],
        KEPLER_SPLIT_POSITIVES["test"],
    )
    for algorithm in ("randbelow", "multiply"):
        train, val, test = split_indices(len(labels), seed=0, algorithm=algorithm)
        found = (int(labels[train].sum()), int(labels[val].sum()), int(labels[test].sum()))
        print(f"Kepler shuffle {algorithm}: positives {found}")
        if found == target:
            return algorithm, train, val, test
    raise SystemExit(
        "Neither shuffle reproduced the published Kepler positive counts "
        f"{target}. The views were not written."
    )


def _write_kepler(output: Path, views: dict[str, np.ndarray], split) -> None:
    algorithm, train, val, test = split
    n = len(views["label"])
    # Two shards stay under GitHub's 100 MB file limit even before compression.
    edges = [0, n // 2, n]
    output.mkdir(parents=True, exist_ok=True)
    for part, (start, stop) in enumerate(zip(edges, edges[1:])):
        payload = {
            "global_view": views["global_view"][start:stop],
            "local_view": views["local_view"][start:stop],
            "label": views["label"][start:stop],
            "period_days": views["period_days"][start:stop],
            "duration_hours": views["duration_hours"][start:stop],
            "transit_depth": views["transit_depth"][start:stop],
            "tce_id": views["tce_id"][start:stop],
        }
        if part == 0:
            payload["train_idx"] = train
            payload["val_idx"] = val
            payload["test_idx"] = test
        path = output / f"part-{part:02d}.npz"
        np.savez_compressed(path, **payload)
        size = path.stat().st_size
        print(f"Wrote {path} ({size / 1e6:.1f} MB)")
        if size > _MAX_BYTES:
            raise SystemExit(f"{path} is over 90 MB. Split it further before publishing.")
    note = output / "SPLIT.txt"
    note.write_text(
        "Frozen 80/10/10 indices from random seed 0.\n"
        f"Algorithm: {algorithm}.\n"
        "randbelow is Fisher-Yates with Random._randbelow, matching random.shuffle on Python 3.10.\n"
        "multiply is the older int(random() * (i + 1)) shuffle.\n"
        "The exporter kept the algorithm whose positive counts match the paper:\n"
        "train 2906, validation 328, test 366.\n",
        encoding="utf-8",
    )


def export_kepler(pickle_path: Path, catalog_path: Path, output: Path) -> None:
    print(f"Loading {pickle_path}")
    with pickle_path.open("rb") as handle:
        global_view, local_view, label = pickle.load(handle)
    global_view = _stack(global_view)
    local_view = _stack(local_view)
    label = _labels(label)
    if len(label) != KEPLER_ROWS:
        raise SystemExit(f"Expected {KEPLER_ROWS} Kepler TCEs, found {len(label)}.")
    if int(label.sum()) != 3600:
        raise SystemExit(f"Expected 3600 planet candidates, found {int(label.sum())}.")
    if global_view.shape[1] != 2001 or local_view.shape[1] != 201:
        raise SystemExit(f"Unexpected view lengths {global_view.shape[1]} and {local_view.shape[1]}.")
    catalog = _kepler_catalog(catalog_path)
    if len(catalog) != len(label):
        raise SystemExit(f"Catalog has {len(catalog)} labeled TCEs, pickle has {len(label)}.")
    catalog_labels = np.asarray([row["label"] for row in catalog], dtype=np.int8)
    if not np.array_equal(catalog_labels, label):
        mismatch = int(np.sum(catalog_labels != label))
        raise SystemExit(
            f"Catalog order does not match the pickle ({mismatch} labels differ). "
            "Views were not written, so a wrong TCE id cannot be attached."
        )
    views = {
        "global_view": global_view,
        "local_view": local_view,
        "label": label,
        "period_days": np.asarray([row["period_days"] for row in catalog], dtype=np.float32),
        "duration_hours": np.asarray([row["duration_hours"] for row in catalog], dtype=np.float32),
        "transit_depth": np.asarray([row["transit_depth"] for row in catalog], dtype=np.float32),
        "tce_id": np.asarray([row["tce_id"] for row in catalog], dtype="U32"),
    }
    _write_kepler(output, views, _matching_split(label))


def _tess_split(path: Path, code: int):
    print(f"Loading {path}")
    with path.open("rb") as handle:
        examples = pickle.load(handle)
    if not examples or not isinstance(examples[0], dict):
        raise SystemExit(f"{path} is not a list of example dicts.")
    global_view = _stack([ex["time_series_features"]["global_view"] for ex in examples])
    local_view = _stack([ex["time_series_features"]["local_view"] for ex in examples])
    secondary = _stack([ex["time_series_features"]["secondary_view"] for ex in examples])
    depth = _stack([ex["aux_features"]["depth_change"] for ex in examples])
    if depth.shape[1] != 1:
        raise SystemExit(f"Depth change has width {depth.shape[1]}, expected 1.")
    labels = _labels([ex["labels"] for ex in examples])
    return global_view, local_view, secondary, depth[:, 0], labels, np.full(len(labels), code, dtype=np.int8)


def export_tess(train: Path, val: Path, test: Path, output: Path) -> None:
    blocks = [
        _tess_split(train, 0),
        _tess_split(val, 1),
        _tess_split(test, 2),
    ]
    names = ("train", "val", "test")
    for name, block in zip(names, blocks):
        size = len(block[4])
        positives = int(block[4].sum())
        print(f"TESS {name}: {size} rows, {positives} positive")
        if size != TESS_SPLIT_SIZES[name] or positives != TESS_SPLIT_POSITIVES[name]:
            raise SystemExit(
                f"TESS {name} does not match the paper "
                f"({TESS_SPLIT_SIZES[name]} rows, {TESS_SPLIT_POSITIVES[name]} positive)."
            )
        if block[0].shape[1] != 201 or block[1].shape[1] != 61 or block[2].shape[1] != 61:
            raise SystemExit(f"Unexpected TESS lengths in {name}.")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "views.npz"
    np.savez_compressed(
        path,
        global_view=np.concatenate([block[0] for block in blocks]),
        local_view=np.concatenate([block[1] for block in blocks]),
        secondary_view=np.concatenate([block[2] for block in blocks]),
        depth_change=np.concatenate([block[3] for block in blocks]).astype(np.float32),
        label=np.concatenate([block[4] for block in blocks]),
        split=np.concatenate([block[5] for block in blocks]),
    )
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kepler-pickle", type=Path, required=True)
    parser.add_argument("--kepler-catalog", type=Path, required=True)
    parser.add_argument("--tess-train", type=Path, required=True)
    parser.add_argument("--tess-val", type=Path, required=True)
    parser.add_argument("--tess-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    export_kepler(args.kepler_pickle, args.kepler_catalog, args.output / "kepler")
    export_tess(args.tess_train, args.tess_val, args.tess_test, args.output / "tess")


if __name__ == "__main__":
    main()
