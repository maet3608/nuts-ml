"""
.. module:: test_plotter
   :synopsis: Unit tests for plotter module
"""
from __future__ import print_function

import os

import numpy as np
import nutsml.imageutil as ni
import nutsml.plotter as pl
import numpy.testing as nt

from nutsflow import Collect

# Set to True to create test data
CREATE_DATA = False


def assert_equal_image(imagepath, image, rtol=0.01, atol=0.01):
    if CREATE_DATA:
        ni.save_image(imagepath, image)
    expected = ni.load_image(imagepath)
    nt.assert_allclose(expected, image, rtol=rtol, atol=atol)


# TODO: This test is successful when run individually
# pytest tests\nutsml\test_plotter.py
# but fails when running as part of the test suite.
def DISABLED_test_plotlines():
    filepath = 'tests/data/temp_plotlines.png'
    xs = np.arange(0, 6.3, 1.2)
    ysin, ycos = np.sin(xs), np.cos(xs)
    data = zip(xs, ysin, ycos)

    out = data >> pl.PlotLines(1, 0, filepath=filepath) >> Collect()
    assert out == data

    expected = 'tests/data/img/plotlines.png'
    image = ni.load_image(filepath)
    os.remove(filepath)
    assert_equal_image(expected, image)
