"""Fit one published schedule and average an ensemble of initializations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from exoplanet_vetting.config import Stage
from exoplanet_vetting.console import Status
from exoplanet_vetting.metrics import Scores, scores
from exoplanet_vetting.models import build_model


@dataclass(frozen=True)
class TrainResult:
    parameters: int
    member_scores: list[Scores]
    ensemble: Scores


class _Epochs(tf.keras.callbacks.Callback):
    def __init__(self, status: Status, short: str, offset: int, total: int):
        super().__init__()
        self._status = status
        self._short = short
        self._offset = offset
        self._total = total

    def on_epoch_end(self, epoch, logs=None):
        self._status.update("Training", self._short, self._offset + epoch + 1, self._total)


def _set_learning_rate(model: tf.keras.Model, learning_rate: float) -> None:
    variable = model.optimizer.learning_rate
    if hasattr(variable, "assign"):
        variable.assign(np.float32(learning_rate))
        return
    model.optimizer.learning_rate = float(learning_rate)


def _fit(model, x_train, y_train, x_val, y_val, stages: tuple[Stage, ...], status: Status, short: str) -> None:
    total = sum(stage.epochs for stage in stages)
    offset = 0
    for index, stage in enumerate(stages):
        if index:
            _set_learning_rate(model, stage.learning_rate)
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=stage.epochs,
            batch_size=stage.batch_size,
            verbose=0,
            callbacks=[_Epochs(status, short, offset, total)],
        )
        offset += stage.epochs


def train_ensemble(
    dataset: str,
    model_name: str,
    lengths: dict[str, int],
    stages: tuple[Stage, ...],
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    ensemble: int,
    seed: int,
    status: Status | None = None,
) -> TrainResult:
    """Train ``ensemble`` networks from seeds ``seed`` ... and score the test set.

    The table in the paper is the mean and standard deviation of the ten
    members. Attention plots in the paper use the mean probability instead,
    which is returned as ``ensemble``.
    """

    status = status or Status()
    member_scores: list[Scores] = []
    probabilities = []
    parameters = 0
    batch = stages[-1].batch_size
    try:
        for member in range(ensemble):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed + member)
            model = build_model(dataset, model_name, stages[0].learning_rate, lengths)
            parameters = int(model.count_params())
            short = f"Member {member + 1}/{ensemble}"
            _fit(model, x_train, y_train, x_val, y_val, stages, status, short)
            predicted = model.predict(x_test, batch_size=batch, verbose=0)
            predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
            probabilities.append(predicted)
            member_scores.append(scores(y_test, predicted))
            del model
    finally:
        status.close()
    mean_probability = np.mean(np.stack(probabilities, axis=0), axis=0)
    return TrainResult(parameters, member_scores, scores(y_test, mean_probability))
