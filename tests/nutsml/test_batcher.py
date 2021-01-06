"""
.. module:: test_batcher
   :synopsis: Unit tests for batcher module
"""

import pytest

import numpy as np
import nutsml.batcher as nb

from pytest import approx
from nutsflow import Collect, Consume


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


def test_build_tensor_batch():
    tensors = [np.zeros((2, 3, 2)), np.ones((2, 3, 2))]
    batch = nb.build_tensor_batch(tensors, 'uint8')
    expected = np.stack(tensors)
    assert batch.dtype == np.uint8
    assert np.array_equal(batch, expected)

    batch = nb.build_tensor_batch(tensors, 'uint8', expand=0)
    expected = np.stack([np.expand_dims(t, 0) for t in tensors])
    assert np.array_equal(batch, expected)

    batch = nb.build_tensor_batch(tensors, float, axes=(1, 0, 2))
    expected = np.stack([np.transpose(t, (1, 0, 2)) for t in tensors])
    assert batch.dtype == float
    assert np.array_equal(batch, expected)

    with pytest.raises(ValueError) as ex:
        nb.build_tensor_batch([], 'uint8')
    assert str(ex.value).startswith('No tensors ')


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

    build_batch = (nb.BuildBatch(2, prefetch=0)
                   .input(0, 'number', float)
                   .input(1, 'vector', np.uint8)
                   .input(2, 'image', np.uint8, False)
                   .output(3, 'one_hot', 'uint8', 3))
    batches = samples >> build_batch >> Collect()
    assert len(batches) == 2

    batch = batches[0]
    assert len(batch) == 2, 'Expect inputs and outputs'
    ins, outs = batch
    assert len(ins) == 3, 'Expect three input columns in batch'
    assert len(outs) == 1, 'Expect one output column in batch'
    assert np.array_equal(ins[0], nb.build_number_batch(numbers[:2], float))
    assert np.array_equal(ins[1], nb.build_vector_batch(vectors[:2], 'uint8'))
    assert np.array_equal(ins[2], nb.build_image_batch(images[:2], 'uint8'))
    assert np.array_equal(outs[0],
                          nb.build_one_hot_batch(class_ids[:2], 'uint8', 3))


def test_BuildBatch_exceptions():
    class_ids = [1, 2]
    numbers = [4.1, 3.2]
    samples = zip(numbers, class_ids)

    with pytest.raises(ValueError) as ex:
        build_batch = (nb.BuildBatch(2, prefetch=0)
                       .input(0, 'number', float)
                       .output(1, 'invalid', 'uint8', 3))
        samples >> build_batch >> Collect()
    assert str(ex.value).startswith('Invalid builder')


def test_BuildBatch_prefetch():
    samples = [[1], [2]]
    build_batch = (nb.BuildBatch(2, prefetch=1)
                   .input(0, 'number', 'uint8'))
    batches = samples >> build_batch >> Collect()
    batch = batches[0][0]
    assert np.array_equal(batch, np.array([1, 2], dtype='uint8'))


def test_Mixup():
    numbers1 = [1, 2, 3]
    numbers2 = [4, 5, 6]
    samples = list(zip(numbers1, numbers2))
    build_batch = (nb.BuildBatch(3, prefetch=0)
                   .input(0, 'number', float)
                   .output(1, 'number', float))

    # no mixup, return original batch
    mixup = nb.Mixup(0.0)
    batches = samples >> build_batch >> mixup >> Collect()
    inputs, outputs = batches[0]
    assert list(inputs[0]) == numbers1
    assert list(outputs[0]) == numbers2

    # mixup with alpaha=1.0
    mixup = nb.Mixup(1.0)
    batches = samples >> build_batch >> mixup >> Collect()

    for input, output in batches:
        input, output = input[0], output[0]
        assert min(input) >= 1 and max(input) <= 3
        assert min(output) >= 4 and max(output) <= 6
        ri, ro = input[0] - samples[0][0], output[0] - samples[0][1]
        assert approx(ri, 1e-3) == ro
