"""
.. module:: test_booster
   :synopsis: Unit tests for booster module
"""

import numpy as np

from nutsflow import Collect
from nutsml.network import Network, PredictNut
from nutsml import Boost, BuildBatch


def predict_all_positive(batch):
    return [np.array([0.0, 1.0]) for _ in batch]


def predict_all_negative(batch):
    return [np.array([1.0, 0.0]) for _ in batch]


def predict_all_perfect(batch):
    pos = np.array([0.0, 1.0])
    neg = np.array([1.0, 0.0])
    return [neg if o < 2 else pos for o in batch]


def predict_all_wrong(batch):
    pos = np.array([0.0, 1.0])
    neg = np.array([1.0, 0.0])
    return [pos if o < 2 else neg for o in batch]


class FakeNetwork(Network):
    def __init__(self, func):
        self.func = func

    def predict(self, flatten=True):
        return PredictNut(self.func, flatten)


def test_Boost():
    negatives = [(0, 0), (1, 0)]
    positives = [(2, 1), (3, 1), (4, 1)]
    samples = negatives + positives

    build_batch = (BuildBatch(3)
                   .by(0, 'number', 'uint8')
                   .by(1, 'one_hot', 'uint8', 2))

    network = FakeNetwork(predict_all_positive)
    boost = Boost(build_batch, network)
    boosted = samples >> boost >> Collect()
    assert boosted == negatives, 'Expect negatives boosted'

    network = FakeNetwork(predict_all_negative)
    boost = Boost(build_batch, network)
    boosted = samples >> boost >> Collect()
    assert boosted == positives, 'Expect positives boosted'

    network = FakeNetwork(predict_all_perfect)
    boost = Boost(build_batch, network)
    boosted = samples >> boost >> Collect()
    assert boosted == [], 'Expect no samples left for boosting'

    network = FakeNetwork(predict_all_wrong)
    boost = Boost(build_batch, network)
    boosted = samples >> boost >> Collect()
    assert boosted == samples, 'Expect all samples boosted'
