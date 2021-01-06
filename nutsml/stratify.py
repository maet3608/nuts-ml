"""
.. module:: stratify
   :synopsis: Stratification of sample sets
"""
from __future__ import absolute_import

import random as rnd

from nutsflow import nut_processor, nut_sink, Sort
from nutsml.datautil import upsample, random_downsample


@nut_processor
def Stratify(iterable, labelcol, labeldist, rand=None):
    """
    iterable >> Stratify(labelcol, labeldist, rand=None)

    Stratifies samples by randomly down-sampling according to the given
    label distribution. In detail: samples belonging to the class with the
    smallest number of samples are returned with probability one. Samples
    from other classes are randomly down-sampled to match the number of
    samples in the smallest class.

    Note that in contrast to SplitRandom, which generates the same random
    split per default, Stratify generates different stratifications.
    Furthermore, while the downsampling is random the order of samples
    remains the same!

    While labeldist needs to be provided or computed upfront the actual
    stratification occurs online and only one sample per time is stored
    in memory.

    >>> from nutsflow import Collect, CountValues
    >>> from nutsflow.common import StableRandom
    >>> fix = StableRandom(1)  # Stable random numbers for doctest

    >>> samples = [('pos', 1), ('pos', 1), ('neg', 0)]
    >>> labeldist = samples >> CountValues(1)
    >>> samples >> Stratify(1, labeldist, rand=fix) >> Sort()
    [('neg', 0), ('pos', 1)]

    :param iterable over tuples iterable: Iterable of tuples where column
       labelcol contains a sample label that is used for stratification
    :param int labelcol: Column of tuple/samples that contains the label,
    :param dict labeldist: Dictionary with numbers of different labels,
       e.g. {'good':12, 'bad':27, 'ugly':3}
    :param Random|None rand: Random number generator used for down-sampling.
       If None, random.Random() is used.
    :return: Stratified samples
    :rtype: Generator over tuples
    """
    rand = rnd.Random() if rand is None else rand
    min_n = float(min(labeldist.values()))
    probs = {l: min_n / n for l, n in labeldist.items()}
    for sample in iterable:
        label = sample[labelcol]
        if rand.random() < probs[label]:
            yield sample


@nut_sink
def CollectStratified(iterable, labelcol, mode='downrnd', container=list,
                      rand=None):
    """
    iterable >> CollectStratified(labelcol, mode='downrnd',  container=list,
                                  rand=rnd.Random())

    Collects samples in a container and stratifies them by either randomly
    down-sampling classes or up-sampling classes by duplicating samples.

    >>> from nutsflow import Collect
    >>> samples = [('pos', 1), ('pos', 1), ('neg', 0)]
    >>> samples >> CollectStratified(1) >> Sort()
    [('neg', 0), ('pos', 1)]

    :param iterable over tuples iterable: Iterable of tuples where column
       labelcol contains a sample label that is used for stratification
    :param int labelcol: Column of tuple/samples that contains the label
    :param string mode:
       'downrnd' : randomly down-sample
       'up' : up-sample
    :param container container: Some container, e.g. list, set, dict
           that can be filled from an iterable
    :param Random|None rand: Random number generator used for sampling.
       If None, random.Random() is used.
    :return: Stratified samples
    :rtype: List of tuples
    """
    rand = rnd.Random() if rand is None else rand
    samples = list(iterable)
    if mode == 'up':
        stratified = upsample(samples, labelcol, rand)
    elif mode == 'downrnd':
        stratified = random_downsample(samples, labelcol, rand)
    else:
        raise ValueError('Unknown mode: ' + mode)
    return container(stratified)
