"""
.. module:: test_checkpoint
   :synopsis: Unit tests for checkpoint module
"""
import pytest
import os
import shutil
import time
import nutsml.checkpoint as nc

from nutsml.network import Network
from nutsml.config import Config
from os.path import join

BASEPATH = 'tests/data/checkpoints'


@pytest.fixture(scope="function")
def checkpointdirs(request):
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def rmdir(path):
        if os.path.exists(path):
            os.rmdir(path)

    checkpoint1 = join(BASEPATH, 'checkpoint1')
    checkpoint2 = join(BASEPATH, 'checkpoint2')
    mkdir(checkpoint1)
    time.sleep(0.1)  # ensure diff in creation time of checkpoints
    mkdir(checkpoint2)

    def fin():
        if os.path.exists(BASEPATH):
            shutil.rmtree(BASEPATH)

    request.addfinalizer(fin)
    return checkpoint1, checkpoint2


class FakeNetwork(Network):
    def __init__(self):
        self.weights = 'weights'

    def save_weights(self, weightspath=None):
        with open(weightspath, 'w') as f:
            f.write(self.weights)

    def load_weights(self, weightspath=None):
        with open(weightspath, 'r') as f:
            self.weights = f.read()


def create_net(lr=0.1):
    network = FakeNetwork()
    optimizer = Config(lr=lr)
    return network, optimizer


def parameters(network, optimizer):
    return dict(lr=optimizer.lr)

def create_net0():
    network = FakeNetwork()
    return network


def parameters0(network):
    return dict()


@pytest.fixture(scope="function")
def create_checkpoint(request):
    def fin():
        if os.path.exists(BASEPATH):
            shutil.rmtree(BASEPATH)

    request.addfinalizer(fin)
    return nc.Checkpoint(create_net, parameters, BASEPATH)


def test_constructor_single():
    checkpoint = nc.Checkpoint(create_net0, parameters0, BASEPATH)
    network = checkpoint.load()
    assert network.weights == 'weights'

    network.weights = 'new_weights'
    checkpoint.save('checkpoint0')

    network = checkpoint.load()
    assert network.weights == 'new_weights'

    shutil.rmtree(BASEPATH)


def test_dirs_empty(create_checkpoint):
    checkpoint = create_checkpoint
    assert checkpoint.dirs() == []


def test_dirs(checkpointdirs, create_checkpoint):
    checkpoint1, checkpoint2 = checkpointdirs
    checkpoint = create_checkpoint
    assert sorted(checkpoint.dirs()) == [checkpoint1, checkpoint2]


def test_latest_empty(create_checkpoint):
    checkpoint = create_checkpoint
    assert checkpoint.latest() is None


def test_latest(checkpointdirs, create_checkpoint):
    checkpoint1, checkpoint2 = checkpointdirs
    checkpoint = create_checkpoint
    assert checkpoint.latest() == checkpoint2


def test_datapaths_empty(create_checkpoint):
    checkpoint = create_checkpoint
    assert checkpoint.datapaths() == (None, None, None)


def test_datapaths(checkpointdirs, create_checkpoint):
    checkpoint1, checkpoint2 = checkpointdirs
    checkpoint = create_checkpoint

    wgt, par, cfg = (join(checkpoint2, 'weights'),
                     join(checkpoint2, 'params.json'),
                     join(checkpoint2, 'config.json'))
    assert checkpoint.datapaths() == (wgt, par, cfg)

    wgt, par, cfg = (join(checkpoint1, 'weights'),
                     join(checkpoint1, 'params.json'),
                     join(checkpoint1, 'config.json'))
    assert checkpoint.datapaths('checkpoint1') == (wgt, par, cfg)


def test_load(create_checkpoint):
    checkpoint = create_checkpoint
    network, optimizer = checkpoint.load()
    assert optimizer.lr == 0.1
    assert isinstance(network, FakeNetwork)
    assert network.weights == 'weights'


def test_save(create_checkpoint):
    checkpoint = create_checkpoint
    network, optimizer = checkpoint.load()

    optimizer.lr = 0.2
    network.weights = 'new_weights'
    checkpoint.save('checkpoint0')

    network, optimizer = checkpoint.load()
    assert optimizer.lr == 0.2
    assert network.weights == 'new_weights'


def test_savebest(create_checkpoint):
    checkpoint = create_checkpoint
    network, optimizer = checkpoint.load()

    for loss in [5, 3, 1, 2, 7]:
        network.weights = 'weights:loss=' + str(loss)
        checkpoint.save_best(loss, isloss=True)

    network, optimizer = checkpoint.load()
    assert network.weights == 'weights:loss=1'

    network.weights = 'weights:loss=3'
    checkpoint.save_best(3, isloss=True)
    network, optimizer = checkpoint.load()
    assert network.weights == 'weights:loss=1'

    network.weights = 'weights:loss=0'
    checkpoint.save_best(0, isloss=True)
    network, optimizer = checkpoint.load()
    assert network.weights == 'weights:loss=0'
