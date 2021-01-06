"""
.. module:: test_metrics
   :synopsis: Unit tests for metrics module

Note: In contrast to losses, which operate on batches, metricies operate on
entire collections of samples.
"""

import pytest
from pytest import approx

import nutsml.metrics as nm


def test_box_intersection():
    # no overlap
    assert nm.box_intersect((0, 0, 1, 1), (2, 2, 0, 0)) == (0, 0, 0, 0)
    # perfect overlap
    assert nm.box_intersect((1, 2, 3, 4), (1, 2, 3, 4)) == (1, 2, 3, 4)
    # completely inside
    assert nm.box_intersect((1, 2, 3, 4), (0, 1, 4, 5)) == (1, 2, 3, 4)
    # symmetric
    assert nm.box_intersect((0, 1, 4, 5), (1, 2, 3, 4)) == (1, 2, 3, 4)
    # four edge cases
    assert nm.box_intersect((1 + 1, 2, 3, 4), (1, 2, 3, 4)) == (2, 2, 2, 4)
    assert nm.box_intersect((1, 2 + 1, 3, 4), (1, 2, 3, 4)) == (1, 3, 3, 3)
    assert nm.box_intersect((1, 2, 3 - 1, 4), (1, 2, 3, 4)) == (1, 2, 2, 4)
    assert nm.box_intersect((1, 2, 3, 4 - 1), (1, 2, 3, 4)) == (1, 2, 3, 3)


def test_box_iou():
    # no overlap
    assert nm.box_iou((0, 0, 1, 1), (2, 2, 0, 0)) == 0.0
    # perfect overlap
    assert nm.box_iou((1, 2, 3, 4), (1, 2, 3, 4)) == 1.0
    # completely inside
    assert nm.box_iou((1, 2, 3, 4), (0, 1, 4, 5)) == 0.6
    # symmetric
    assert nm.box_iou((0, 1, 4, 5), (1, 2, 3, 4)) == 0.6
    # four edge cases
    assert nm.box_iou((1 + 1, 2, 3, 4), (1, 2, 3, 4)) == 0.5
    assert nm.box_iou((1, 2 + 1, 3, 4), (1, 2, 3, 4)) == 0.6
    assert nm.box_iou((1, 2, 3 - 1, 4), (1, 2, 3, 4)) == approx(0.66, 0.1)
    assert nm.box_iou((1, 2, 3, 4 - 1), (1, 2, 3, 4)) == 0.75


def test_box_matches():
    boxes = [(1, 1, 3, 3), (4, 2, 2, 3), (5, 5, 2, 1)]
    box = (2, 2, 3, 1)
    expected = [((1, 1, 3, 3), 0.2), ((4, 2, 2, 3), 0.125), ((5, 5, 2, 1), 0.0)]
    assert nm.box_matches(box, boxes) == expected


def test_box_best_match():
    boxes = [(1, 1, 3, 3), (4, 2, 2, 3), (5, 5, 2, 1)]
    box = (2, 2, 3, 1)
    assert nm.box_best_match(box, boxes) == ((1, 1, 3, 3), 0.2)
    assert nm.box_best_match(box, []) == (None, 0.0)


def test_box_pr_curve():
    approx = lambda prc: [(round(p, 2), round(r, 2), s) for p, r, s in prc]

    boxes1 = [(1, 1, 3, 3), (4, 2, 2, 3), (5, 5, 2, 1)]
    boxes2 = [(2, 1, 2, 3), (4, 3, 2, 3)]
    scores1 = [0.5, 0.2, 0.1]
    scores2 = [0.5, 0.2]

    pr_curve = list(nm.box_pr_curve(boxes2, boxes2, scores2))
    expected = [(1.0, 0.5, 0.5), (1.0, 1.0, 0.2)]
    assert pr_curve == expected

    pr_curve = list(nm.box_pr_curve(boxes1, boxes2, scores2))
    expected = [(1.0, 0.33, 0.5), (1.0, 0.67, 0.2)]
    assert approx(pr_curve) == expected

    pr_curve = list(nm.box_pr_curve(boxes2, boxes1, scores1))
    expected = [(1.0, 0.5, 0.5), (1.0, 1.0, 0.2), (0.67, 1.0, 0.1)]
    assert approx(pr_curve) == expected

    pr_curve = list(nm.box_pr_curve(boxes1, [], []))
    assert pr_curve == []

    pr_curve = list(nm.box_pr_curve([], boxes1, scores1))
    assert pr_curve == []


def test_box_avg_precision():
    boxes1 = [(1, 1, 3, 3), (4, 2, 2, 3), (5, 5, 2, 1)]
    scores1 = [0.5, 0.2, 0.1]
    boxes2 = [(2, 1, 2, 3), (4, 3, 2, 3)]
    scores2 = [0.5, 0.2]

    ap = nm.box_avg_precision(boxes2, boxes2, scores2)
    assert ap == 1.0

    ap = nm.box_avg_precision(boxes1, boxes2, scores2)
    assert ap == approx(0.63, abs=1e-2)

    ap = nm.box_avg_precision(boxes2, boxes1, scores1)
    assert ap == 1.0

    ap = nm.box_avg_precision(boxes1, [], [])
    assert ap == 0.0

    ap = nm.box_avg_precision([], boxes1, scores1)
    assert ap == 0.0


def test_box_mean_avg_precision():
    boxes1 = [(1, 1, 3, 3), (4, 2, 2, 3), (5, 5, 2, 1)]
    labels1 = ['class1', 'class2', 'class1']
    scores1 = [0.5, 0.2, 0.1]
    boxes2 = [(2, 1, 2, 3), (4, 3, 2, 3)]
    labels2 = ['class1', 'class2']
    scores2 = [0.5, 0.2]

    mAP = nm.box_mean_avg_precision(boxes1, labels1, boxes1, labels1, scores1)
    assert mAP == 1.0

    mAP = nm.box_mean_avg_precision(boxes2, labels2, boxes2, labels2, scores2)
    assert mAP == 1.0

    mAP = nm.box_mean_avg_precision(boxes1, labels1, [], [], [])
    assert mAP == 0.0

    mAP = nm.box_mean_avg_precision([], [], boxes1, labels1, scores1)
    assert mAP == 0.0

    mAP = nm.box_mean_avg_precision(boxes1, labels1, boxes2, labels2, scores2)
    assert mAP == approx(0.77, abs=1e-2)
