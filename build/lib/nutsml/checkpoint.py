"""
.. module:: checkpoint
   :synopsis: Conveniency class to create checkpoints for network training.
"""

import os
import time

from os.path import join, exists, isdir, getmtime
from .config import Config

"""
.. module:: checkpoint
   :synopsis: Creation of checkpoints with network weights and parameters.
"""


class Checkpoint(object):
    """
    A factory for checkpoints to periodically save network weights and other
    hyper/configuration parameters.

    | Example usage:
    |
    | def create_network(lr=0.01, momentum=0.9):
    |   model = Sequential()
    |   ...
    |   optimizer = opt.SGD(lr=lr, momentum=momentum)
    |   model.compile(optimizer=optimizer, metrics=['accuracy'])
    |   return KerasNetwork(model), optimizer
    |
    | def parameters(network, optimizer):
    |   return dict(lr = optimizer.lr, momentum = optimizer.momentum)
    |
    | def train_network():
    |   checkpoint = Checkpoint(create_network, parameters)
    |   network, optimizer = checkpoint.load()
    |
    |   for epoch in xrange(EPOCHS):
    |     train_err = train_network()
    |     val_err = validate_network()
    |
    |     if epoch % 10 == 0:  # Reduce learning rate every 10 epochs
    |       optimizer.lr /= 2
    |
    |     checkpoint.save_best(val_err)
    |

    Checkpoints can also be saved under different names, e.g.

    |  checkpoint.save_best(val_err, 'checkpoint'+str(epoch))

    And specific checkpoints can be loaded:

    | network, config = checkpoint.load('checkpoint103')

    If no checkpoint is specified the most recent one is loaded.
    """

    def __init__(self, create_net, parameters, checkpointspath='checkpoints'):
        """
        Create checkpoint factory.

        >>> def create_network(lr=0.1):
        ...     return 'MyNetwork', lr

        >>> def parameters(network, lr):
        ...     return dict(lr = lr)

        >>> checkpoint = Checkpoint(create_network, parameters)
        >>> network, lr = checkpoint.load()
        >>> network, lr
        ('MyNetwork', 0.1)

        :param function create_net: Function that takes keyword parameters
           and returns a nuts-ml Network and and any other values or objects
           needed to describe the state to be checkpointed.
           Note: parameters(*create_net()) must work!
        :param function parameters: Function that takes output of create_net()
            and returns dictionary with parameters (same as the one that are
            used in create_net(...))
        :param string checkpointspath: Path to folder that will contain
          checkpoint folders.
        """
        if not exists(checkpointspath):
            os.makedirs(checkpointspath)
        self.basepath = checkpointspath
        self.create_net = create_net
        self.parameters = parameters
        self.state = None  # network and other objets
        self.network = None  # only the network
        self.config = None  # bestscore and other checkpoint params

    def dirs(self):
        """
        Return full paths to all checkpoint folders.

        :return: Paths to all folders under the basedir.
        :rtype: list
        """
        paths = (join(self.basepath, d) for d in os.listdir(self.basepath))
        return [p for p in paths if isdir(p)]

    def latest(self):
        """
        Find most recently modified/created checkpoint folder.

        :return: Full path to checkpoint folder if it exists otherwise None.
        :rtype: str | None
        """
        dirs = sorted(self.dirs(), key=getmtime, reverse=True)
        return dirs[0] if dirs else None

    def datapaths(self, checkpointname=None):
        """
        Return paths to network weights, parameters and config files.

        If no checkpoints exist under basedir (None, None, None) is returned.

        :param str|None checkpointname: Name of checkpoint. If name is None
           the most recent checkpoint is used.
        :return: (weightspath, paramspath, configpath) or (None, None, None)
        :rtype: tuple
        """
        name = checkpointname
        if name is None:
            path = self.latest()
            if path is None:
                return None, None, None
        else:
            path = join(self.basepath, name)
            if not exists(path):
                os.makedirs(path)
        return (join(path, 'weights'), join(path, 'params.json'),
                join(path, 'config.json'))

    def save(self, checkpointname='checkpoint'):
        """
        Save network weights and parameters under the given name.

        :param str checkpointname: Name of checkpoint folder. Path will be
           self.basepath/checkpointname
        :return: path to checkpoint folder
        :rtype: str
        """
        weightspath, paramspath, configpath = self.datapaths(checkpointname)
        self.config.timestamp = time.time()
        self.network.save_weights(weightspath)
        state = self.state if hasattr(self.state, '__iter__') else [self.state]
        Config(self.parameters(*state)).save(paramspath)
        Config(self.config).save(configpath)
        return join(self.basepath, checkpointname)

    def save_best(self, score, checkpointname='checkpoint', isloss=False):
        """
        Save best network weights and parameters under the given name.

        :param float|int score: Some score indicating quality of network.
        :param str checkpointname: Name of checkpoint folder.
        :param bool isloss: True, score is a loss and lower is better otherwise
           higher is better.
        :return: path to checkpoint folder
        :rtype: str
        """
        bestscore = self.config.bestscore
        if (bestscore is None
            or (isloss and score < bestscore)
            or (not isloss and score > bestscore)):
            self.config.bestscore = score
            self.config.isloss = isloss
            self.save(checkpointname)
        return join(self.basepath, checkpointname)

    def load(self, checkpointname=None):
        """
        Create network, load weights and parameters.

        :param str|none checkpointname: Name of checkpoint to load. If None
           the most recent checkpoint is used. If no checkpoint exists yet
           the network will be created but no weights loaded and the
           default configuration will be returned.
        :return: whatever self.create_net returns
        :rtype: object
        """
        weightspath, paramspath, configpath = self.datapaths(checkpointname)
        params = Config().load(paramspath) if paramspath else None
        state = self.create_net(**params) if params else self.create_net()
        self.network = state[0] if hasattr(state, '__iter__') else state
        self.state = state
        if weightspath:
            self.network.load_weights(weightspath)
        defaultconfig = Config(bestscore=None, timestamp=None)
        self.config = Config().load(configpath) if configpath else defaultconfig
        return state
