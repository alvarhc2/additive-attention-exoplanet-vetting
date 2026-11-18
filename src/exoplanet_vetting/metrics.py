"""Thresholded classification scores used in the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scores:
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    auc: float
    tp: int
    fp: int
    tn: int
    fn: int


def _divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Area under the ROC curve. Undefined when a class is missing."""

    y = np.asarray(labels).astype(int).reshape(-1)
    score = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = np.concatenate(([0.0], tps / n_pos))
    fpr = np.concatenate(([0.0], fps / n_neg))
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def scores(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Scores:
    y = np.asarray(labels).astype(int).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predicted = (probability >= threshold).astype(int)
    tp = int(np.sum((predicted == 1) & (y == 1)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    precision = _divide(tp, tp + fp)
    sensitivity = _divide(tp, tp + fn)
    return Scores(
        accuracy=_divide(tp + tn, tp + tn + fp + fn),
        sensitivity=sensitivity,
        specificity=_divide(tn, tn + fp),
        precision=precision,
        f1=_divide(2 * precision * sensitivity, precision + sensitivity),
        auc=roc_auc(y, probability),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def mean_std(members: list[Scores]) -> dict[str, tuple[float, float]]:
    """Mean and sample standard deviation of each metric across members.

    A single member has standard deviation 0. The paper table is this mean
    and this deviation, not a bootstrap.
    """

    names = ("accuracy", "sensitivity", "specificity", "precision", "f1", "auc")
    summary = {}
    for name in names:
        values = np.array([getattr(member, name) for member in members], dtype=np.float64)
        if len(values) == 1:
            deviation = 0.0
        else:
            deviation = float(np.std(values, ddof=1))
        summary[name] = (float(np.mean(values)), deviation)
    return summary
