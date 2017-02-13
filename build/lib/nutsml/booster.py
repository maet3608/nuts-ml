"""
.. module:: booster
   :synopsis: Boosting of wrongly predicted samples
"""

import numpy as np

from random import random
from nutsflow import nut_processor, Tee, Collect


@nut_processor
def Boost(iterable, batcher, network, targetcol=-1):
    """
    iterable >> Boost(batcher, network, targetcol=-1)

    Boost samples with high softmax probability for incorrect class
    Expects one-hot encoded targets and softmax predictions for output.

    network = Network()
    build_batch = BuildBatch(BATCHSIZE, colspec)
    boost = Boost(build_batch, network)
    samples >> boost >> network.train() >> Consume()

    :param iterable iterable: Iterable with samples.
    :param nutsml.BuildBatch batcher: Batcher used for network training.
    :param nutsml.Network network: Network used for prediction
    :param int targetcol: Column in sample that contains target values.
    :return: Iterator with samples to boost
    :rtype: iterator
    """

    def do_boost(probs, target):
        assert len(target) > 1, 'Expect one-hot encoded target: ' + str(target)
        assert len(target) == len(probs), 'Expect softmax probs: ' + str(probs)
        return random() > probs[np.argmax(target)]

    samples1, samples2 = iterable >> Tee(2)
    for batch in samples1 >> batcher:
        p_batch, target = batch[:targetcol], batch[targetcol]
        pred = [p_batch] >> network.predict() >> Collect()
        for p, t, s in zip(pred, target, samples2):
            if do_boost(p, t):
                yield s
