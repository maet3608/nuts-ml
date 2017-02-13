"""
.. module:: common
   :synopsis: Common nuts
"""

import numpy as np
import random as rnd

from nutsflow import nut_function, nut_sink
from nutsml.datautil import group_by


@nut_function
def CheckNaN(data):
    """
    Raise exception if data contains NaN.

    Useful to stop training if network doesn't converge and loss gets NaN,
    e.g. samples >> network.train() >> CheckNan() >> log >> Consume()

    >>> [1, 2, 3] >> CheckNaN() >> Collect()
    [1, 2, 3]

    >>> import numpy as np
    >>> [1, np.NaN, 3] >> CheckNaN() >> Collect()
    Traceback (most recent call last):
    ...
    RuntimeError: NaN encountered

    :param data: Items or iterables.
    :return: Return input data if they don't contain NaNs
    :rtype: any
    :raise: RuntimeError if data contains NaN.
    """
    if np.any(np.isnan(data)):
        raise RuntimeError('NaN encountered: ' + str(data))
    return data


@nut_sink
def PartitionByCol(iterable, col, values):
    """
    Partition samples in interable depending on column value.

    >>> samples = [(1,1), (2,0), (2,4), (1,3), (3,0)]
    >>> ones, twos = samples >> PartitionByCol(0, [1, 2])
    >>> ones
    [(1, 1), (1, 3)]
    >>> twos
    [(2, 0), (2, 4)]

    Note that values does not need to contain all possible values. It is
    sufficient to provide the values for the partitions wanted.

    :param iterable iterable: Iterable over samples
    :param int col: Index of column to extract
    :param list values: List of column values to create partitions for.
    :return: tuple of partitions
    :rtype: tuple
    """
    groups = group_by(iterable, lambda sample: sample[col])
    return tuple(groups.get(v, []) for v in values)


@nut_sink
def SplitRandom(iterable, ratio=0.7, constraint=lambda x: x,
                rand=rnd.Random(0)):
    """
    Randomly split iterable into partitions.

    >>> train, val = xrange(10) >> SplitRandom(ratio=0.7)
    >>> train, val
    [9, 4, 0, 5, 2, 7, 1], [3, 6, 8]

    >>> train, val, test = xrange(10) >> SplitRandom(ratio=(0.6, 0.3, 0.1))
    >>> train, val, test
    [7, 6, 5, 9, 3, 8], [2, 1, 4], [0]

    >>> data = zip('aabbccddee', xrange(10))
    >>> same_letter = lambda (c,i): c
    >>> train, val = data >> SplitRandom(ratio=0.6, constraint=same_letter)
    >>> train
    [('b', 3), ('c', 4), ('b', 2), ('c', 5), ('a', 0), ('a', 1)]
    >>> val
    [('e', 8), ('e', 9), ('d', 7), ('d', 6)]

    :param iterable iterable: Iterable over anything. Will be consumed!
    :param float|tuple ratio: Ratio of two partition e.g. a ratio of 0.7
            means 70%, 30% split.
            Alternatively a list or ratios can be provided, e.g.
            ratio=(0.6, 0.3, 0.1). Note that ratios must sum up to one.
    :param function constraint: Function that returns key the elements of
        the iterable are grouped by before partitioning. Useful to ensure
        that a partition contains related elements, e.g. left and right eye
        images are not scattered across partitions.
        Note that constrains have precedence over ratios.
    :param random.Random rand: Random number generator.
            rand=rnd.Random(0) = 0 ensures that the same split is created
            every time SplitRandom is called. This is important when continuing
            an interrupted training session!
            see random.Random for more info.
    :return: partitions of iterable with sizes according to provided ratios.
    :rtype: (list, list, ...)
    """
    samples = list(iterable)
    if hasattr(ratio, '__iter__'):
        ratios = tuple(ratio)
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError('Ratios must sum up to one: ' + str(ratios))
    else:
        ratios = (ratio, 1.0 - ratio)
    ns = [int(len(samples) * r) for r in ratios]

    groups = group_by(samples, constraint).values()
    rand.shuffle(groups)
    groups = iter(groups)
    splits = []

    def append(split):
        rand.shuffle(split)
        splits.append(split)

    for n in ns[:-1]:
        split = []
        for group in groups:
            split.extend(group)
            if len(split) >= n:
                append(split)
                break
    append([e for g in groups for e in g])  # append remaining groups
    return splits
