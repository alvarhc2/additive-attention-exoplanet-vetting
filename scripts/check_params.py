import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

from exoplanet_vetting.config import LENGTHS, PAPER_PARAMETERS
from exoplanet_vetting.models import build_model

for (dataset, model), expected in PAPER_PARAMETERS.items():
    tf.keras.backend.clear_session()
    lengths = {key: value for key, value in LENGTHS[dataset].items() if value is not None}
    count = int(build_model(dataset, model, 1e-4, lengths).count_params())
    mark = "OK" if count == expected else f"DIFF {count - expected}"
    print(f"{dataset:7} {model:22} {count:10} expected {expected:10} {mark}")
