import tensorflow as tf

from exoplanet_vetting.attention import AdditiveAttention
from exoplanet_vetting.config import LENGTHS, PAPER_PARAMETERS
from exoplanet_vetting.models import build_model


def test_attention_weights_sum_to_one():
    layer = AdditiveAttention()
    values = tf.random.normal((4, 20, 8))
    context, weights = layer(values)
    assert context.shape == (4, 8)
    assert weights.shape == (4, 20, 1)
    totals = tf.reduce_sum(weights, axis=1)
    np_totals = totals.numpy()
    assert np_totals.shape == (4, 1)
    assert abs(float(np_totals.mean()) - 1.0) < 1e-5


def _count(dataset: str, model: str) -> int:
    tf.keras.backend.clear_session()
    lengths = {key: value for key, value in LENGTHS[dataset].items() if value is not None}
    network = build_model(dataset, model, learning_rate=1e-4, lengths=lengths)
    count = int(network.count_params())
    tf.keras.backend.clear_session()
    return count


def test_published_parameter_counts():
    for key, expected in PAPER_PARAMETERS.items():
        dataset, model = key
        assert _count(dataset, model) == expected, key
