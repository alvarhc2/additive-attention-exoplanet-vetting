"""The five networks compared in the paper.

Columns are independent until the feed-forward stack. ``optimized-attention``
is the published model: two valid convolutions, an LSTM stack, and additive
attention on every view. ``optimized`` is that network with the last LSTM
returning its final state instead of the attention vector. ``baseline`` is a
single LSTM unit on the concatenated views. ``astronet`` is the Keras
reimplementation of Shallue & Vanderburg (Kepler) and Yu et al. (TESS).
"""

from __future__ import annotations

import tensorflow as tf

from exoplanet_vetting.attention import AdditiveAttention
from exoplanet_vetting.config import ADAM_BETA_1, ADAM_BETA_2, ADAM_EPSILON

_GLOBAL_UNITS = (500, 200, 100)
_LOCAL_UNITS = (200, 100, 50)


def _compile(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=ADAM_BETA_1,
            beta_2=ADAM_BETA_2,
            epsilon=ADAM_EPSILON,
        ),
    )
    return model


def _conv_lstm_column(inputs, units: tuple[int, ...], attention: bool, prefix: str):
    hidden = tf.keras.layers.Conv1D(128, 10, activation="relu", name=f"{prefix}_conv1")(inputs)
    hidden = tf.keras.layers.Conv1D(128, 10, activation="relu", name=f"{prefix}_conv2")(hidden)
    for index, width in enumerate(units):
        last = index == len(units) - 1
        hidden = tf.keras.layers.LSTM(
            width,
            activation="tanh",
            return_sequences=attention or not last,
            name=f"{prefix}_lstm{index + 1}",
        )(hidden)
    if attention:
        context, _weights = AdditiveAttention(name=f"{prefix}_attention")(hidden)
        return context
    return hidden


def _feed_forward(merged):
    hidden = merged
    for index in range(3):
        hidden = tf.keras.layers.Dense(150, activation="relu", name=f"fc{index + 1}")(hidden)
    return tf.keras.layers.Dense(1, activation="sigmoid", name="out")(hidden)


def _optimized(dataset: str, lengths: dict[str, int], attention: bool, learning_rate: float):
    global_in = tf.keras.Input((lengths["global"], 1), name="global_view")
    local_in = tf.keras.Input((lengths["local"], 1), name="local_view")
    parts = [
        _conv_lstm_column(global_in, _GLOBAL_UNITS, attention, "global"),
        _conv_lstm_column(local_in, _LOCAL_UNITS, attention, "local"),
    ]
    inputs = [global_in, local_in]
    if dataset == "tess":
        secondary_in = tf.keras.Input((lengths["secondary"], 1), name="secondary_view")
        depth_in = tf.keras.Input((1,), name="depth_change")
        parts.append(_conv_lstm_column(secondary_in, _LOCAL_UNITS, attention, "secondary"))
        parts.append(depth_in)
        inputs.extend([secondary_in, depth_in])
    merged = tf.keras.layers.Concatenate(name="views")(parts)
    model = tf.keras.Model(inputs, _feed_forward(merged), name=f"{dataset}_{'attention' if attention else 'lstm'}")
    return _compile(model, learning_rate)


def _baseline(lengths: dict[str, int], attention: bool, learning_rate: float, dataset: str):
    steps = lengths["global"] + lengths["local"]
    if dataset == "tess":
        # Global, local, secondary, and the depth scalar as one extra step.
        steps += lengths["secondary"] + 1
    inputs = tf.keras.Input((steps, 1), name="views")
    sequence = tf.keras.layers.LSTM(
        1,
        activation="tanh",
        return_sequences=attention,
        name="lstm",
    )(inputs)
    if attention:
        sequence, _weights = AdditiveAttention(name="attention")(sequence)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="out")(sequence)
    model = tf.keras.Model(inputs, outputs, name=f"{dataset}_baseline")
    return _compile(model, learning_rate)


def _conv_pool(inputs, filters: tuple[int, ...], pools: tuple[int, ...], padding: str, prefix: str):
    hidden = inputs
    block = 0
    for index, width in enumerate(filters):
        hidden = tf.keras.layers.Conv1D(
            width,
            5,
            padding=padding,
            activation="relu",
            name=f"{prefix}_conv{index + 1}",
        )(hidden)
        if index % 2 == 1:
            hidden = tf.keras.layers.MaxPooling1D(
                pool_size=pools[block],
                strides=2,
                name=f"{prefix}_pool{block + 1}",
            )(hidden)
            block += 1
    return tf.keras.layers.Flatten(name=f"{prefix}_flat")(hidden)


def _astronet(dataset: str, lengths: dict[str, int], learning_rate: float):
    # Kepler uses valid convolutions. TESS uses same padding, a secondary
    # column, and the depth scalar. Both end in four dense layers of 512.
    padding = "valid" if dataset == "kepler" else "same"
    global_filters = (16, 16, 32, 32, 64, 64, 128, 128, 256, 256)
    local_filters = (16, 16, 32, 32)
    global_in = tf.keras.Input((lengths["global"], 1), name="global_view")
    local_in = tf.keras.Input((lengths["local"], 1), name="local_view")
    parts = [
        _conv_pool(global_in, global_filters, (5, 5, 5, 5, 5), padding, "global"),
        _conv_pool(local_in, local_filters, (7, 7), padding, "local"),
    ]
    inputs = [global_in, local_in]
    if dataset == "tess":
        secondary_in = tf.keras.Input((lengths["secondary"], 1), name="secondary_view")
        depth_in = tf.keras.Input((1,), name="depth_change")
        parts.append(_conv_pool(secondary_in, local_filters, (7, 7), padding, "secondary"))
        parts.append(depth_in)
        inputs.extend([secondary_in, depth_in])
    hidden = tf.keras.layers.Concatenate(name="views")(parts)
    for index in range(4):
        hidden = tf.keras.layers.Dense(512, activation="relu", name=f"fc{index + 1}")(hidden)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="out")(hidden)
    model = tf.keras.Model(inputs, outputs, name=f"{dataset}_astronet")
    return _compile(model, learning_rate)


def build_model(dataset: str, model: str, learning_rate: float, lengths: dict[str, int]) -> tf.keras.Model:
    """Build one of the published networks for ``lengths`` in bins."""

    if dataset not in {"kepler", "tess"}:
        raise ValueError(f"Unknown dataset {dataset!r}.")
    if model == "baseline":
        return _baseline(lengths, attention=False, learning_rate=learning_rate, dataset=dataset)
    if model == "baseline-attention":
        return _baseline(lengths, attention=True, learning_rate=learning_rate, dataset=dataset)
    if model == "optimized":
        return _optimized(dataset, lengths, attention=False, learning_rate=learning_rate)
    if model == "optimized-attention":
        return _optimized(dataset, lengths, attention=True, learning_rate=learning_rate)
    if model == "astronet":
        return _astronet(dataset, lengths, learning_rate)
    raise ValueError(f"Unknown model {model!r}.")
