"""Simplified additive attention from the published model.

The score of time step j is ``v tanh(b_j + W h_j)`` with a single attention
unit. Classification has no decoder state, so the query term of Bahdanau
attention is a learned vector ``b`` of one value per time step. ``W`` is the
paper's ``U_a`` and ``v`` is ``v_a``. This is the layer called
``BahdanauAttentionMod`` in the experiment code.
"""

from __future__ import annotations

import tensorflow as tf


class AdditiveAttention(tf.keras.layers.Layer):
    """Pool an LSTM sequence into one vector and return the time weights."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_layer = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        features = int(input_shape[-1])
        steps = int(input_shape[1])
        if steps < 1:
            raise ValueError("Additive attention needs a known sequence length.")
        self.w = self.add_weight(
            name="attention_weight",
            shape=(features, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(steps, 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, values):
        energy = tf.nn.tanh(self.b + tf.matmul(values, self.w))
        score = self.score_layer(energy)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * values, axis=1)
        return context, weights
