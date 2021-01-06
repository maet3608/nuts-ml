"""
.. module:: test_datautil
   :synopsis: Unit tests for datautil module
"""

import pytest

import numpy as np
import collections as cl
import nutsml.datautil as util

from collections import namedtuple
from nutsflow.common import StableRandom


@pytest.fixture(scope="function")
def sampleset():
    """Return list with 50 positive and 10 negative samples"""
    pos = [(0, i) for i in range(50)]
    neg = [(1, i) for i in range(10)]
    return pos + neg


def test_random_upsample(sampleset):
    samples = [('pos', 1), ('pos', 1), ('neg', 0)]
    stratified = sorted(util.upsample(samples, 1, rand=StableRandom(0)))
    assert stratified == [('neg', 0), ('neg', 0), ('pos', 1), ('pos', 1)]

    stratified1 = util.upsample(sampleset, 0, rand=StableRandom(0))
    _, labelcnts = util.group_samples(stratified1, 0)
    assert labelcnts == {0: 50, 1: 50}

    stratified2 = util.upsample(sampleset, 0, rand=StableRandom(1))
    assert stratified1 != stratified2, 'Order should be random'


def test_random_downsample(sampleset):
    samples = [('pos', 1), ('pos', 1), ('neg', 0)]
    stratified = sorted(
        util.random_downsample(samples, 1, rand=StableRandom(0)))
    assert stratified == [('neg', 0), ('pos', 1)]

    stratified1 = util.random_downsample(sampleset, 0, rand=StableRandom(0))
    _, labelcnts = util.group_samples(stratified1, 0)
    assert labelcnts == {0: 10, 1: 10}

    stratified2 = util.random_downsample(sampleset, 0, rand=StableRandom(1))
    assert stratified1 != stratified2, 'Order should be random'


def test_group_samples():
    samples = [('pos', 1), ('pos', 1), ('neg', 0)]
    groups, labelcnts = util.group_samples(samples, 1)
    assert groups == {0: [('neg', 0)], 1: [('pos', 1), ('pos', 1)]}
    assert labelcnts == cl.Counter({1: 2, 0: 1})


def test_group_by():
    is_odd = lambda e: bool(e % 2)
    numbers = [0, 1, 2, 3, 4]
    assert util.group_by(numbers, is_odd) == {False: [0, 2, 4], True: [1, 3]}
    assert util.group_by([1, 3], is_odd) == {True: [1, 3]}
    assert util.group_by([], is_odd) == dict()


def test_col_map():
    sample = (1, 2, 3)
    add_n = lambda x, n: x + n
    assert util.col_map(sample, 1, add_n, 10) == (1, 12, 3)
    assert util.col_map(sample, (0, 2), add_n, 10) == (11, 2, 13)


def test_shuffle_sublists():
    sublists = [[1, 2, 3], [4, 5, 6, 7]]
    util.shuffle_sublists(sublists, StableRandom(0))
    assert sublists == [[1, 3, 2], [4, 5, 7, 6]]
