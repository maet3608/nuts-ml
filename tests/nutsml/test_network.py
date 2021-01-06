"""
.. module:: test_network
   :synopsis: Unit tests for network module
"""

import pytest

import numpy as np

from nutsflow import Collect, GetCols, Print
from nutsml.network import Network, TrainValNut, PredictNut, EvalNut


class FakeModel(object):
    def __init__(self):
        self.saved_weights = None
        self.loaded_weights = None

    def train(self, X, y):
        return sum(X) + 1, sum(y) + 1

    def validate(self, X, y):
        return sum(X) + 2, sum(y) + 2

    def predict(self, X):
        return X

    def save_weights(self, weightspath):
        self.saved_weights = weightspath

    def load_weights(self, weightspath):
        self.loaded_weights = weightspath

    def summary(self):
        return "network summary"


class FakeNetwork(Network):
    def __init__(self, model, weightspath):
        Network.__init__(self, weightspath)
        self.model = model

    def train(self):
        return TrainValNut(self.model.train)

    def validate(self):
        return TrainValNut(self.model.validate)

    def predict(self, flatten=True):
        return PredictNut(self.model.predict, flatten)

    def evaluate(self, metrics, predcol=None):
        def compute(metric, targets, preds):
            return metric(targets, preds)

        return EvalNut(self, metrics, compute, predcol)

    def save_weights(self, weightspath=None):
        weightspath = super(FakeNetwork, self)._weightspath(weightspath)
        self.model.save_weights(weightspath)

    def load_weights(self, weightspath=None):
        weightspath = super(FakeNetwork, self)._weightspath(weightspath)
        self.model.load_weights(weightspath)

    def print_layers(self):
        self.model.summary()


def test_TrainValNut():
    batches = [(1, 2), (3, 4)]
    nut = TrainValNut(lambda X, y: X + y)
    assert batches >> nut >> Collect() == [3, 7]


def test_PredictNut():
    batches = [(1, 2), (3, 4)]
    nut = PredictNut(lambda X: X, flatten=True)
    assert batches >> nut >> Collect() == [1, 2, 3, 4]


def test_EvalNut():
    model = FakeModel()
    network = FakeNetwork(model, 'dummy_filepath')

    acc = lambda X, y: np.sum(X == y)
    compute = lambda m, t, p: m(t, p)
    nut = EvalNut(network, [acc], compute)

    batches = [((1, 2), (1, 2)), ((5, 6), (5, 6))]
    assert batches >> nut == 4
    batches = [((1, 2), (1, 2)), ((5, 0), (5, 6))]
    assert batches >> nut == 3

    batches = [(((0, 1), (0, 2)), (0, 2)), (((5, 5), (6, 6)), (6, 6))]
    nut = EvalNut(network, [acc], compute, predcol=1)
    assert batches >> nut == 4
    nut = EvalNut(network, [acc], compute, predcol=0)
    assert batches >> nut == 1


def test_Network_constructor():
    weightspath = 'dummy_filepath'
    network = Network(weightspath)
    assert network.weightspath == weightspath


def test_Network_weightspath():
    weightspath = 'dummy_filepath'
    network = Network(weightspath)
    assert network._weightspath(None) == weightspath
    assert network._weightspath('new_path') == 'new_path'


def test_Network_save_load_weights():
    model = FakeModel()
    weightspath = 'dummy_filepath'
    network = FakeNetwork(model, weightspath)
    assert not model.saved_weights
    assert not model.loaded_weights

    network.save_weights()
    assert model.saved_weights == weightspath

    network.load_weights()
    assert model.loaded_weights == weightspath


def test_Network_save_best():
    model = FakeModel()
    weightspath = 'dummy_filepath'
    network = FakeNetwork(model, weightspath)

    network.save_best(2.0)
    assert network.best_score == 2.0
    assert model.saved_weights == weightspath

    network.save_best(1.0, isloss=True)
    assert network.best_score == 1.0

    network.save_best(3.0, isloss=False)
    assert network.best_score == 3.0


def test_Network_exceptions():
    network = Network('dummy_filepath')

    with pytest.raises(NotImplementedError) as ex:
        network.train()
    assert str(ex.value) == 'Implement train()!'

    with pytest.raises(NotImplementedError) as ex:
        network.validate()
    assert str(ex.value) == 'Implement validate()!'

    with pytest.raises(NotImplementedError) as ex:
        network.predict()
    assert str(ex.value) == 'Implement predict()!'

    with pytest.raises(NotImplementedError) as ex:
        network.evaluate([])
    assert str(ex.value) == 'Implement evaluate()!'

    with pytest.raises(NotImplementedError) as ex:
        network.save_weights()
    assert str(ex.value) == 'Implement save_weights()!'

    with pytest.raises(NotImplementedError) as ex:
        network.load_weights()
    assert str(ex.value) == 'Implement load_weights()!'

    with pytest.raises(NotImplementedError) as ex:
        network.print_layers()
    assert str(ex.value) == 'Implement print_layers()!'


def test_Network():
    model = FakeModel()

    weightspath = 'dummy_filepath'
    network = FakeNetwork(model, weightspath)
    assert network.weightspath == weightspath

    batches = [((1, 2), (3, 4)), ((5, 6), (7, 8))]
    train_err = batches >> network.train() >> Collect()
    assert train_err == [(4, 8), (12, 16)]

    val_err = batches >> network.validate() >> Collect()
    assert val_err == [(5, 9), (13, 17)]

    prediction = batches >> GetCols(0) >> network.predict() >> Collect()
    assert prediction == [(1, 2), (5, 6)]

    prediction = batches >> GetCols(0) >> network.predict(False) >> Collect()
    assert prediction == [((1, 2),), ((5, 6),)]

    batches = [((1, 2), (1, 2)), ((5, 6), (5, 6))]
    acc = lambda X, y: np.sum(X == y)
    assert batches >> network.evaluate([acc]) == 4

    batches = [(((0, 1), (0, 2)), (0, 2)), (((5, 5), (6, 6)), (6, 6))]
    assert batches >> network.evaluate([acc], predcol=1) == 4
    assert batches >> network.evaluate([acc], predcol=0) == 1
