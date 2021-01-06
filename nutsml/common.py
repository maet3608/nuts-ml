"""
.. module:: common
   :synopsis: Common nuts
"""

import numpy as np
import random as rnd

from nutsflow import (nut_function, nut_sink, NutFunction,
                      Flatten, Collect)
from nutsflow.common import StableRandom
from nutsml.datautil import group_by, shuffle_sublists


@nut_function
def CheckNaN(data):
    """
    Raise exception if data contains NaN.

    Useful to stop training if network doesn't converge and loss function
    returns NaN. Example:
    samples >> network.train() >> CheckNan() >> log >> Consume()

    >>> from nutsflow import Collect

    >>> [1, 2, 3] >> CheckNaN() >> Collect()
    [1, 2, 3]

    >>> import numpy as np
    >>> [1, np.NaN, 3] >> CheckNaN() >> Collect()
    Traceback (most recent call last):
    ...
    RuntimeError: NaN encountered: nan

    :param data: Items or iterables.
    :return: Return input data if it doesn't contain NaN
    :rtype: any
    :raise: RuntimeError if data contains NaN.
    """
    if np.any(np.isnan(data)):
        raise RuntimeError('NaN encountered: ' + str(data))
    return data


@nut_sink
def PartitionByCol(iterable, column, values):
    """
    Partition samples in iterables depending on column value.

    >>> samples = [(1,1), (2,0), (2,4), (1,3), (3,0)]
    >>> ones, twos = samples >> PartitionByCol(0, [1, 2])
    >>> ones
    [(1, 1), (1, 3)]
    >>> twos
    [(2, 0), (2, 4)]

    Note that values does not need to contain all possible values. It is
    sufficient to provide the values for the partitions wanted.

    :param iterable iterable: Iterable over samples
    :param int column: Index of column to extract
    :param list values: List of column values to create partitions for.
    :return: tuple of partitions
    :rtype: tuple
    """
    groups = group_by(iterable, lambda sample: sample[column])
    return tuple(groups.get(v, []) for v in values)


class ConvertLabel(NutFunction):
    """
    Convert string labels to integer class ids (or one-hot) and vice versa.
    """

    def __init__(self, column, labels, onehot=False):
        """
        Convert string labels to integer class ids (or one-hot) and vice versa.

        Also converts confidence vectors, e.g. softmax output or float values
        to class labels.

        >>> from nutsflow import Collect
        >>> labels = ['class0', 'class1', 'class2']

        >>> convert = ConvertLabel(None, labels)
        >>> [1, 0] >> convert >> Collect()
        ['class1', 'class0']
        >>> ['class1', 'class0'] >> convert >> Collect()
        [1, 0]
        >>> [0.9, 0.4, 1.6] >> convert >> Collect()
        ['class1', 'class0', 'class2']
        >>> [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]] >> convert >> Collect()
        ['class1', 'class0']

        >>> convert = ConvertLabel(None, labels, onehot=True)
        >>> ['class1', 'class0'] >> convert >> Collect()
        [[0, 1, 0], [1, 0, 0]]

        >>> convert = ConvertLabel(1, labels)
        >>> [('data', 'class1'), ('data', 'class0')] >> convert >> Collect()
        [('data', 1), ('data', 0)]
        >>> [('data', 1), ('data', 2)] >> convert >> Collect()
        [('data', 'class1'), ('data', 'class2')]
        >>> [('data', 0.9)] >> convert >> Collect()
        [('data', 'class1')]
        >>> [('data', [0.1, 0.7, 0.2])] >> convert >> Collect()
        [('data', 'class1')]

        :param int column: Index of column in sample that contains label.
           If None process labels directly.
        :param list|tuple labels: List of class labels (strings).
        :param bool onehot: True: convert class labels to one-hot encoded
          vectors. False, convert to class index.
        """
        self.column = column
        self.labels = labels
        self.onehot = onehot
        self.n_labels = len(labels)
        self.id2label = {i: l for i, l in enumerate(labels)}
        self.label2id = {l: i for i, l in enumerate(labels)}

    def __call__(self, sample):
        """Return sample and replace label within sample if it is a sample"""
        hascol = self.column is not None
        x = sample[self.column] if hascol else sample

        if isinstance(x, str):
            y = self.label2id[x]
        elif isinstance(x, int):
            y = self.id2label[x]
        elif isinstance(x, float):
            y = self.id2label[round(x)]
        else:  # assume vector with confidence values
            assert len(x) == len(self.labels)
            _, argmax = max((v, i) for i, v in enumerate(x))
            y = self.id2label[argmax]

        if self.onehot and isinstance(y, int):
            y = [1 if i == y else 0 for i in range(self.n_labels)]

        if hascol:  # input has columns => return sample
            outsample = list(sample)
            outsample[self.column] = y
            return tuple(outsample)
        else:
            return y


@nut_sink
def SplitRandom(iterable, ratio=0.7, constraint=None, rand=None):
    """
    Randomly split iterable into partitions.

    For the same input data the same split is created every time and is stable
    across different Python version 2.x or 3.x. A random number generator
    can be provided to create varying splits.

    >>> train, val = range(10) >> SplitRandom(ratio=0.7)
    >>> train, val
    ([6, 3, 1, 7, 0, 2, 4], [5, 9, 8])

    >>> range(10) >> SplitRandom(ratio=0.7)  # Same split again
    [[6, 3, 1, 7, 0, 2, 4], [5, 9, 8]]

    >>> train, val, test = range(10) >> SplitRandom(ratio=(0.6, 0.3, 0.1))
    >>> train, val, test
    ([6, 1, 4, 0, 3, 2], [8, 7, 9], [5])

    >>> data = zip('aabbccddee', range(10))
    >>> same_letter = lambda t: t[0]
    >>> train, val = data >> SplitRandom(ratio=0.6, constraint=same_letter)
    >>> sorted(train)
    [('a', 0), ('a', 1), ('b', 2), ('b', 3), ('d', 6), ('d', 7)]
    >>> sorted(val)
    [('c', 4), ('c', 5), ('e', 8), ('e', 9)]

    :param iterable iterable: Iterable over anything. Will be consumed!
    :param float|tuple ratio: Ratio of two partition e.g. a ratio of 0.7
            means 70%, 30% split.
            Alternatively a list or ratios can be provided, e.g.
            ratio=(0.6, 0.3, 0.1). Note that ratios must sum up to one
            and cannot be zero.
    :param function|None constraint: Function that returns key the elements of
        the iterable are grouped by before partitioning. Useful to ensure
        that a partition contains related elements, e.g. left and right eye
        images are not scattered across partitions.
        Note that constrains have precedence over ratios.
    :param Random|None rand: Random number generator. The default None
            ensures that the same split is created every time SplitRandom
            is called. This is important when continuing an interrupted
            training session or running the same training on machines with
            different Python versions. Note that Python's random.Random(0)
            generates different number for Python 2.x and 3.x!
    :return: partitions of iterable with sizes according to provided ratios.
    :rtype: (list, list, ...)
    """
    rand = StableRandom(0) if rand is None else rand
    samples = list(iterable)

    if hasattr(ratio, '__iter__'):
        ratios = tuple(ratio)
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError('Ratios must sum up to one: ' + str(ratios))
        if min(ratios) <= 0:
            raise ValueError('Ratios cannot be zero: ' + str(ratios))
    else:
        ratios = (ratio, 1.0 - ratio)
    ns = [int(len(samples) * r) for r in ratios]

    if constraint is None:
        groups = [[s] for s in samples]
    else:
        groups = list(group_by(samples, constraint, True).values())

    rand.shuffle(groups)
    groups = iter(groups)
    splits = []
    for n in ns[:-1]:
        split = []
        for group in groups:
            split.extend(group)
            if len(split) >= n:
                splits.append(split)
                break
    splits.append([e for g in groups for e in g])  # append remaining groups
    shuffle_sublists(splits, rand)

    return splits


@nut_sink
def SplitLeaveOneOut(iterable, keyfunc=None):
    """
    Returns a leave-one-out split of the iterable.

    Note that SplitLeaveOneOut consumes the entire input stream
    and returns a generator over the leave-one-out splits.
    The splits are stable across Python version 2.x or 3.x
    and deterministic.

    >>> from nutsflow.common import console  # just for printing

    >>> samples = [1, 2, 3]
    >>> for train, test in samples >> SplitLeaveOneOut():
    ...     console(train, '  ', test)
    [2, 3]    [1]
    [1, 3]    [2]
    [1, 2]    [3]

    >>> samples = [(1, 1), (2, 0), (2, 4), (1, 3), (3, 0)]
    >>> splits = samples >> SplitLeaveOneOut(lambda x: x[0])
    >>> for train, test in splits:
    ...     console(train, '   ', test)
    [(2, 0), (2, 4), (3, 0)]     [(1, 1), (1, 3)]
    [(1, 1), (1, 3), (3, 0)]     [(2, 0), (2, 4)]
    [(1, 1), (1, 3), (2, 0), (2, 4)]     [(3, 0)]

    :param iterable iterable: Iterable over anything. Will be consumed!
    :param function/None keyfunc: Function that returns value the split
      is based on. If None, the sample itself serves as key.
    :return: generator over leave-one-out train and test splits (train, test)
    :rtype: Generator[(list, list)]
    """
    samples = list(iterable)
    if keyfunc is None:
        keyfunc = lambda x: x
    groups = list(group_by(samples, keyfunc, True).values())
    idxs = range(len(groups))
    for i in idxs:
        test = groups[i]
        others = (groups[k] for k in idxs if k != i)
        train = others >> Flatten() >> Collect()
        yield train, test
