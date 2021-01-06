"""
.. module:: test_writer
   :synopsis: Unit tests for writer module
"""

import pytest
import os

import numpy as np

from nutsml.imageutil import load_image
from nutsflow import Collect, Get, Consume
from nutsml import ReadImage, WriteImage


def test_ImageWriter():
    samples = [('nut_color', 1), ('nut_grayscale', 2)]
    inpath = 'tests/data/img_formats/*.bmp'
    img_samples = samples >> ReadImage(0, inpath) >> Collect()

    imagepath = 'tests/data/test_*.bmp'
    names = samples >> Get(0) >> Collect()
    img_samples >> WriteImage(0, imagepath, names) >> Consume()

    for sample, name in zip(img_samples, names):
        filepath = 'tests/data/test_{}.bmp'.format(name)
        arr = load_image(filepath)
        assert np.array_equal(arr, sample[0])
        os.remove(filepath)

    pathfunc = lambda sample, name: 'tests/data/test_{}.jpg'.format(name)
    img_samples >> WriteImage(0, pathfunc) >> Consume()
    for i, sample in enumerate(img_samples):
        filepath = 'tests/data/test_{}.jpg'.format(i)
        os.path.exists(filepath)
        os.remove(filepath)

    pathfunc = lambda sample, name: 'tests/data/test_{}.jpg'.format(name)
    img_samples >> Get(0) >> WriteImage(None, pathfunc) >> Consume()
    for i, sample in enumerate(img_samples):
        filepath = 'tests/data/test_{}.jpg'.format(i)
        os.path.exists(filepath)
        os.remove(filepath)

    namefunc=lambda sample: 'img'+str(sample[1])
    img_samples >> WriteImage(0, imagepath, namefunc) >> Consume()
    for sample, name in zip(img_samples, ['test_img1', 'test_img2']):
        filepath = 'tests/data/{}.bmp'.format(name)
        os.path.exists(filepath)
        os.remove(filepath)

    with pytest.raises(ValueError) as ex:
        img_samples >> WriteImage(0, ()) >> Consume()
    assert str(ex.value).startswith('Expect path or function')


