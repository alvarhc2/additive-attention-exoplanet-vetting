"""Command line for the published vetting experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from exoplanet_vetting.config import DATASETS, MODELS, Stage, paper_schedule
from exoplanet_vetting.console import Status, enable_color, line, title
from exoplanet_vetting.data import load_dataset, model_inputs
from exoplanet_vetting.metrics import mean_std
from exoplanet_vetting.synthetic import synthetic_cohort


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m exoplanet_vetting",
        description="Vet transiting exoplanet candidates with the published networks.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", type=str, help="Directory with kepler/ and tess/ view files")
    source.add_argument("--synthetic", action="store_true", help="Train on a tiny generated cohort")
    parser.add_argument("--paper", action="store_true", help="Published model, schedule, and ensemble of 10")
    parser.add_argument("--dataset", default="kepler", choices=DATASETS, help="Which mission to train")
    parser.add_argument("--model", default=None, choices=MODELS, help="Network. Default: optimized-attention with --paper, otherwise baseline")
    parser.add_argument("--ensemble", type=int, default=None, help="Initializations to average. Default: 10 with --paper, otherwise 1")
    parser.add_argument("--epochs", type=int, default=None, help="Replace the published schedule with one stage of this length")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the batch size of every stage")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override the learning rate of every stage")
    parser.add_argument("--seed", type=int, default=0, help="First ensemble seed. Members use seed, seed+1, ...")
    parser.add_argument("--output", default="outputs", help="Directory for metrics.txt")
    return parser


def _bind_runtime() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass


def _stages(args: argparse.Namespace, dataset: str, model: str) -> tuple[Stage, ...]:
    published = paper_schedule(dataset, model)
    if args.epochs is None and args.learning_rate is None and args.batch_size is None:
        return published
    if args.epochs is None:
        return tuple(
            Stage(
                stage.epochs,
                args.learning_rate if args.learning_rate is not None else stage.learning_rate,
                args.batch_size if args.batch_size is not None else stage.batch_size,
            )
            for stage in published
        )
    learning_rate = args.learning_rate if args.learning_rate is not None else published[0].learning_rate
    batch_size = args.batch_size if args.batch_size is not None else published[0].batch_size
    return (Stage(args.epochs, learning_rate, batch_size),)


def _format_stage(stage: Stage) -> str:
    epochs = "epoch" if stage.epochs == 1 else "epochs"
    return f"{stage.epochs} {epochs}, learning rate {stage.learning_rate:g}, batch {stage.batch_size}"


def _format_pair(mean: float, deviation: float) -> str:
    if np.isnan(mean):
        return "undefined"
    return f"{mean:.4f} ± {deviation:.4f}"


def _write_metrics(path: Path, dataset: str, model: str, result, summary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"dataset {dataset}",
        f"model {model}",
        f"parameters {result.parameters}",
        f"members {len(result.member_scores)}",
    ]
    for name, (mean, deviation) in summary.items():
        rows.append(f"{name} {_format_pair(mean, deviation)}")
    pooled = result.ensemble
    rows.append(f"mean_probability_accuracy {pooled.accuracy:.4f}")
    rows.append(f"mean_probability_auc {pooled.auc:.4f}")
    rows.append(f"tp {pooled.tp} fp {pooled.fp} tn {pooled.tn} fn {pooled.fn}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    enable_color()
    parser = _parser()
    args = parser.parse_args(argv)
    if args.paper and args.synthetic:
        parser.error("--paper trains on the published views. Drop --synthetic.")
    if args.ensemble is not None and args.ensemble < 1:
        parser.error("--ensemble must be at least 1.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    model = args.model or ("optimized-attention" if args.paper else "baseline")
    ensemble = args.ensemble if args.ensemble is not None else (10 if args.paper else 1)
    published = paper_schedule(args.dataset, model)
    if args.paper or args.epochs is not None or args.learning_rate is not None or args.batch_size is not None:
        stages = _stages(args, args.dataset, model)
    else:
        stages = (Stage(1, 1e-4, 16),)
    smoke = stages == (Stage(1, 1e-4, 16),) and ensemble == 1 and not args.paper

    if args.synthetic or (not args.paper and args.data is None):
        cohort = synthetic_cohort(args.dataset, seed=args.seed)
        source = "synthetic"
    else:
        data_dir = args.data or "data"
        try:
            cohort = load_dataset(args.dataset, data_dir)
        except FileNotFoundError as error:
            parser.error(str(error))
        source = str(data_dir)

    _bind_runtime()
    from exoplanet_vetting.training import train_ensemble

    title("Budget")
    if smoke:
        line("Smoke run. This is not the published budget.")
    line(f"{args.dataset}, {model}, {ensemble} member{'s' if ensemble != 1 else ''}")
    for index, stage in enumerate(stages, start=1):
        prefix = f"Stage {index}: " if len(stages) > 1 else ""
        line(prefix + _format_stage(stage))
    if args.epochs is not None and len(published) > 1:
        line("One learning rate replaces the two-stage published schedule.")
    line(f"Seeds {args.seed} to {args.seed + ensemble - 1}")

    title("Data")
    counts = {split: int(cohort.labels[cohort.indices(split)].sum()) for split in ("train", "val", "test")}
    sizes = {split: int(len(cohort.indices(split))) for split in ("train", "val", "test")}
    line(f"Source {source}")
    line(
        "Train {train} ({train_pos} positive), val {val} ({val_pos} positive), test {test} ({test_pos} positive)".format(
            train=sizes["train"],
            train_pos=counts["train"],
            val=sizes["val"],
            val_pos=counts["val"],
            test=sizes["test"],
            test_pos=counts["test"],
        )
    )
    lengths = cohort.lengths()
    bins = f"global {lengths['global']}, local {lengths['local']}"
    if lengths["secondary"] is not None:
        bins += f", secondary {lengths['secondary']}, depth change"
    line(bins)

    x_train, y_train = model_inputs(cohort, cohort.train_idx, model)
    x_val, y_val = model_inputs(cohort, cohort.val_idx, model)
    x_test, y_test = model_inputs(cohort, cohort.test_idx, model)
    status = Status()
    result = train_ensemble(
        args.dataset,
        model,
        {key: value for key, value in lengths.items() if value is not None},
        stages,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        ensemble,
        args.seed,
        status,
    )
    summary = mean_std(result.member_scores)
    title("Test")
    line(f"{result.parameters:,} parameters")
    for name in ("accuracy", "sensitivity", "specificity", "precision", "auc"):
        line(f"{name:<12} {_format_pair(*summary[name])}")
    pooled = result.ensemble
    line(f"Mean probability  accuracy {pooled.accuracy:.4f}  AUC {pooled.auc:.4f}")
    line(f"tp {pooled.tp}  fp {pooled.fp}  tn {pooled.tn}  fn {pooled.fn}")
    path = Path(args.output) / "metrics.txt"
    _write_metrics(path, args.dataset, model, result, summary)
    title("Files")
    line(str(path))


if __name__ == "__main__":
    main()
