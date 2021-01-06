"""
.. module:: test_common
   :synopsis: Unit tests for common module
"""
import pytest

import numpy as np

from six.moves import zip, range
from nutsflow import Consume, Collect, Map
from nutsflow.common import StableRandom, shapestr
from nutsml import (SplitRandom, SplitLeaveOneOut, CheckNaN, PartitionByCol,
                    ConvertLabel)


def test_CheckNaN():
    assert [1, 2] >> CheckNaN() >> Collect() == [1, 2]

    with pytest.raises(RuntimeError) as ex:
        [1, np.NaN, 3] >> CheckNaN() >> Consume()
    assert str(ex.value).startswith('NaN encountered')

    with pytest.raises(RuntimeError) as ex:
        [(1, np.NaN), (2, 4)] >> CheckNaN() >> Consume()
    assert str(ex.value).startswith('NaN encountered')


def test_PartitionByCol():
    samples = [(1, 1), (2, 0), (2, 4), (1, 3), (3, 0)]

    ones, twos = samples >> PartitionByCol(0, [1, 2])
    assert ones == [(1, 1), (1, 3)]
    assert twos == [(2, 0), (2, 4)]

    twos, ones = samples >> PartitionByCol(0, [2, 1])
    assert ones == [(1, 1), (1, 3)]
    assert twos == [(2, 0), (2, 4)]

    ones, fours = samples >> PartitionByCol(0, [1, 4])
    assert ones == [(1, 1), (1, 3)]
    assert fours == []

    ones, twos = [] >> PartitionByCol(0, [1, 2])
    assert ones == []
    assert twos == []


def test_SplitRandom_split():
    train, val = range(1000) >> SplitRandom(ratio=0.7)
    assert len(train) == 700
    assert len(val) == 300
    assert not set(train).intersection(val)


def test_SplitRandom_ratios():
    train, val, test = range(1000) >> SplitRandom(ratio=(0.6, 0.3, 0.1))
    assert len(train) == 600
    assert len(val) == 300
    assert len(test) == 100

    with pytest.raises(ValueError) as ex:
        range(100) >> SplitRandom(ratio=(0.6, 0.7))
    assert str(ex.value).startswith('Ratios must sum up to one')

    with pytest.raises(ValueError) as ex:
        range(10) >> SplitRandom(ratio=(1, 0))
    assert str(ex.value).startswith('Ratios cannot be zero')


def test_SplitRandom_stable_default():
    split1 = range(10) >> SplitRandom()
    split2 = range(10) >> SplitRandom()
    assert split1 == split2


def test_SplitRandom_seed():
    split1 = range(10) >> SplitRandom(rand=StableRandom(0))
    split2 = range(10) >> SplitRandom(rand=StableRandom(0))
    split3 = range(10) >> SplitRandom(rand=StableRandom(1))
    assert split1 == split2
    assert split1 != split3


def test_SplitRandom_constraint():
    same_letter = lambda t: t[0]
    data = zip('aabbccddee', range(10))
    train, val = data >> SplitRandom(rand=None, ratio=0.6,
                                     constraint=same_letter) >> Collect()
    train.sort()
    val.sort()
    assert train == [('a', 0), ('a', 1), ('b', 2), ('b', 3), ('d', 6), ('d', 7)]
    assert val == [('c', 4), ('c', 5), ('e', 8), ('e', 9)]


def test_SplitLeaveOneOut():
    samples = [1, 2, 3]
    splits = samples >> SplitLeaveOneOut() >> Collect()
    assert splits == [([2, 3], [1]),
                      ([1, 3], [2]),
                      ([1, 2], [3])]


def test_ConvertLabel():
    labels = ['class0', 'class1', 'class2']

    convert = ConvertLabel(None, labels)
    assert [1, 0] >> convert >> Collect() == ['class1', 'class0']
    assert ['class1', 'class0'] >> convert >> Collect() == [1, 0]
    assert [0.9, 1.6] >> convert >> Collect() == ['class1', 'class2']
    assert [[0.1, 0.7, 0.2]] >> convert >> Collect() == ['class1']

    convert = ConvertLabel(0, labels)
    assert [('class2',)] >> convert >> Collect() == [(2,)]
    assert [(1,)] >> convert >> Collect() == [('class1',)]
    assert [(0.1,)] >> convert >> Collect() == [('class0',)]
    assert [(0.9,)] >> convert >> Collect() == [('class1',)]
    assert [(1.7,)] >> convert >> Collect() == [('class2',)]
    assert [([0.1, 0.7, 0.2],)] >> convert >> Collect() == [('class1',)]
    assert [(2,), (0,)] >> convert >> Collect() == [('class2',), ('class0',)]
    assert [(1, 'data')] >> convert >> Collect() == [('class1', 'data')]

    convert = ConvertLabel(None, labels, True)
    expected = [[0, 1, 0], [1, 0, 0]]
    assert ['class1', 'class0'] >> convert >> Collect() == expected
