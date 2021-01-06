"""
.. module:: booster
   :synopsis: Boosting of wrongly predicted samples
"""

import random as rnd
import numpy as np

from nutsflow.common import StableRandom
from nutsflow import nut_processor, Tee, Collect, Flatten, Print


@nut_processor
def Boost(iterable, batcher, network, rand=None):
    """
    iterable >> Boost(batcher, network, rand=None)

    Boost samples with high softmax probability for incorrect class.
    Expects one-hot encoded targets and softmax predictions for output.

    NOTE: prefetching of batches must be disabled when using boosting!

    | network = Network()
    | build_batch = BuildBatch(BATCHSIZE, prefetch=0).input(...).output(...)
    | boost = Boost(build_batch, network)
    | samples >> ... ?>> boost >> build_batch >> network.train() >> Consume()

    :param iterable iterable: Iterable with samples.
    :param nutsml.BuildBatch batcher: Batcher used for network training.
    :param nutsml.Network network: Network used for prediction
    :param Random|None rand: Random number generator used for down-sampling.
       If None, random.Random() is used.
    :return: Generator over samples to boost
    :rtype: generator
    """

    def do_boost(probs, target):
        assert len(target) > 1, 'Expect one-hot encoded target: ' + str(target)
        assert len(target) == len(probs), 'Expect softmax probs: ' + str(probs)
        return rand.random() > probs[np.argmax(target)]

    assert batcher.prefetch == 0, 'Disable prefetch when boosting'
    rand = rnd.Random() if rand is None else rand
    samples1, samples2 = iterable >> Tee(2)
    for batch in samples1 >> batcher:
        inputs, targets = batch
        tars = targets[0]
        preds = iter(inputs) >> network.predict() >> Collect()
        for p,t,s in zip(preds, tars, samples2):
            if do_boost(p, t):
                yield s
