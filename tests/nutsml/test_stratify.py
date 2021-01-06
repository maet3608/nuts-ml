"""
.. module:: test_stratify
   :synopsis: Unit tests for stratify module
"""

import pytest

from nutsflow import Collect, Sort, Get, CountValues
from nutsflow.common import StableRandom
from nutsml import Stratify, CollectStratified


def test_CollectStratified():
    rand = StableRandom(0)

    samples = [('pos', 1), ('pos', 1), ('neg', 0)]
    stratify = CollectStratified(1, mode='up', rand=rand)
    stratified = samples >> stratify >> Sort()
    assert stratified == [('neg', 0), ('neg', 0), ('pos', 1), ('pos', 1)]

    samples = [('pos', 1), ('pos', 1), ('pos', 1), ('neg1', 0), ('neg2', 0)]
    stratify = CollectStratified(1, mode='downrnd', rand=rand)
    stratified = samples >> stratify >> Sort()
    assert stratified == [('neg1', 0), ('neg2', 0), ('pos', 1), ('pos', 1)]

    with pytest.raises(ValueError) as ex:
        samples >> CollectStratified(1, mode='invalid')
    assert str(ex.value).startswith('Unknown mode')


def test_Stratify():
    samples = [('pos', 1)] * 1000 + [('neg', 0)] * 100
    dist = samples >> CountValues(1)

    stratify = Stratify(1, dist, rand=StableRandom(0))
    stratified1 = samples >> stratify >> Collect()
    stratified2 = samples >> stratify >> Collect()

    assert stratified1 != stratified2

    dist1 = stratified1 >> Get(1) >> CountValues()
    print(dist1)
    assert dist1[0] == 100
    assert 90 < dist1[1] < 110

    dist2 = stratified2 >> Get(1) >> CountValues()
    print(dist2)
    assert dist1[0] == 100
    assert 90 < dist1[1] < 110
