"""
.. module:: test_batcher
   :synopsis: Unit tests for batcher module
"""

import pytest

import numpy as np
import nutsml.batcher as nb

from nutsflow import Collect


def test_build_number_batch():
    numbers = [1, 2, 3, 1]
    batch = nb.build_number_batch(numbers, 'uint8')
    assert batch.dtype == np.uint8
    assert np.array_equal(batch, numbers)


def test_build_one_hot_batch():
    class_ids = [0, 1, 2, 1]
    batch = nb.build_one_hot_batch(class_ids, 'uint8', 3)
    one_hot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    assert batch.dtype == np.uint8
    assert np.array_equal(batch, one_hot)


def test_build_vector_batch():
    vectors = [np.array([1, 2, 3]), np.array([2, 3, 4])]
    batch = nb.build_vector_batch(vectors, 'uint8')
    expected = np.array([[1, 2, 3], [2, 3, 4]], dtype='uint8')
    assert batch.dtype == np.uint8
    assert np.array_equal(batch, expected)

    with pytest.raises(ValueError) as ex:
        nb.build_vector_batch([], 'uint8')
    assert str(ex.value).startswith('No vectors ')


def test_build_image_batch():
    images = [np.zeros((10, 5)), np.ones((10, 5))]
    batch = nb.build_image_batch(images, 'uint8', channelfirst=True)
    assert batch.dtype == np.uint8
    assert batch.shape == (2, 1, 10, 5)
    assert np.array_equal(batch[0], np.zeros((1, 10, 5)))
    assert np.array_equal(batch[1], np.ones((1, 10, 5)))

    images = [np.zeros((10, 5)), np.ones((10, 5))]
    batch = nb.build_image_batch(images, 'uint8', channelfirst=False)
    assert batch.shape == (2, 10, 5, 1)
    assert np.array_equal(batch[0], np.zeros((10, 5, 1)))
    assert np.array_equal(batch[1], np.ones((10, 5, 1)))

    images = [np.zeros((10, 5, 1)), np.ones((10, 5, 1))]
    batch = nb.build_image_batch(images, 'uint8', channelfirst=True)
    assert batch.shape == (2, 1, 10, 5)
    assert np.array_equal(batch[0], np.zeros((1, 10, 5)))
    assert np.array_equal(batch[1], np.ones((1, 10, 5)))

    images = [np.zeros((10, 5, 3)), np.ones((10, 5, 3))]
    batch = nb.build_image_batch(images, 'uint8', channelfirst=False)
    assert batch.shape == (2, 10, 5, 3)
    assert np.array_equal(batch[0], np.zeros((10, 5, 3)))
    assert np.array_equal(batch[1], np.ones((10, 5, 3)))


def test_build_image_batch_exceptions():
    with pytest.raises(ValueError) as ex:
        nb.build_image_batch([], 'uint8')
    assert str(ex.value).startswith('No images ')

    with pytest.raises(ValueError) as ex:
        images = [np.zeros((3, 10, 5)), np.ones((3, 10, 5))]
        nb.build_image_batch(images, 'uint8')
    assert str(ex.value).startswith('Channel not at last axis')

    with pytest.raises(ValueError) as ex:
        images = [np.zeros((10, 5)), np.ones((15, 5))]
        nb.build_image_batch(images, 'uint8')
    assert str(ex.value).startswith('Images vary in shape')


def test_BuildBatch():
    numbers = [4.1, 3.2, 1.1]
    vectors = [np.array([1, 2, 3]), np.array([2, 3, 4]), np.array([3, 4, 5])]
    images = [np.zeros((5, 3)), np.ones((5, 3)), np.ones((5, 3))]
    class_ids = [1, 2, 1]
    samples = zip(numbers, vectors, images, class_ids)

    build_batch = (nb.BuildBatch(2)
                   .by(0, 'number', float)
                   .by(1, 'vector', np.uint8)
                   .by(2, 'image', np.uint8, True)
                   .by(3, 'one_hot', 'uint8', 3))
    batches = samples >> build_batch >> Collect()
    assert len(batches) == 2, 'Expect two batches'

    batch = batches[0]
    assert len(batch) == 4, 'Expect four columns in batch'
    assert np.array_equal(batch[0], nb.build_number_batch(numbers[:2], float))
    assert np.array_equal(batch[1], nb.build_vector_batch(vectors[:2], 'uint8'))
    assert np.array_equal(batch[2], nb.build_image_batch(images[:2], 'uint8'))
    assert np.array_equal(batch[3],
                          nb.build_one_hot_batch(class_ids[:2], 'uint8', 3))


def test_BuildBatch_exceptions():
    class_ids = [1, 2]
    numbers = [4.1, 3.2]
    samples = zip(numbers, class_ids)

    with pytest.raises(ValueError) as ex:
        build_batch = (nb.BuildBatch(2, prefetch=0)
                       .by(0, 'number', float)
                       .by(1, 'invalid', 'uint8', 3))
        samples >> build_batch >> Collect()
    assert str(ex.value).startswith('Invalid builder')
