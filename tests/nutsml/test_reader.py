"""
.. module:: test_reader
   :synopsis: Unit tests for reader module
"""

import pytest

import pandas as pd
import numpy as np
import numpy.testing as nt

from collections import namedtuple
from nutsflow import Collect, Sort
from nutsml import ReadNumpy, ReadImage, ReadLabelDirs, ReadPandas


def test_ReadLabelDirs():
    read = ReadLabelDirs('tests/data/labeldirs', '*.txt')
    samples = read >> Sort()
    assert samples == [('tests/data/labeldirs/0/test0.txt', '0'),
                       ('tests/data/labeldirs/1/test1.txt', '1'),
                       ('tests/data/labeldirs/1/test11.txt', '1')]

    read = ReadLabelDirs('tests/data/labeldirs', '*.txt', '')
    samples = read >> Sort()
    assert samples == [('tests/data/labeldirs/0/test0.txt', '0'),
                       ('tests/data/labeldirs/1/test1.txt', '1'),
                       ('tests/data/labeldirs/1/test11.txt', '1'),
                       ('tests/data/labeldirs/_2/test2.txt', '_2')]


def test_ReadNumpy():
    arr0 = np.load('tests/data/img_arrays/nut_color.gif.npy')
    arr1 = np.load('tests/data/img_arrays/nut_grayscale.gif.npy')
    samples = [('nut_color', 1), ('nut_grayscale', 2)]

    filepath = 'tests/data/img_arrays/*.gif.npy'
    np_samples = samples >> ReadNumpy(0, filepath) >> Collect()
    nt.assert_equal(np_samples[0][0], arr0)
    nt.assert_equal(np_samples[1][0], arr1)
    assert np_samples[0][1] == 1
    assert np_samples[1][1] == 2

    pathfunc = lambda s: 'tests/data/img_arrays/{0}.gif.npy'.format(*s)
    np_samples = samples >> ReadNumpy(0, pathfunc) >> Collect()
    nt.assert_equal(np_samples[0][0], arr0)
    nt.assert_equal(np_samples[1][0], arr1)

    samples = [('label', 'tests/data/img_arrays/nut_color.gif.npy')]
    np_samples = samples >> ReadImage(1) >> Collect()
    nt.assert_equal(np_samples[0][1], arr0)


def test_ReadImage():
    arr0 = np.load('tests/data/img_arrays/nut_color.gif.npy')
    arr1 = np.load('tests/data/img_arrays/nut_grayscale.gif.npy')
    samples = [('nut_color', 1), ('nut_grayscale', 2)]

    imagepath = 'tests/data/img_formats/*.gif'
    img_samples = samples >> ReadImage(0, imagepath) >> Collect()
    nt.assert_equal(img_samples[0][0], arr0)
    nt.assert_equal(img_samples[1][0], arr1)
    assert img_samples[0][1] == 1
    assert img_samples[1][1] == 2

    pathfunc = lambda sample: 'tests/data/img_formats/{0}.gif'.format(*sample)
    img_samples = samples >> ReadImage(0, pathfunc) >> Collect()
    nt.assert_equal(img_samples[0][0], arr0)
    nt.assert_equal(img_samples[1][0], arr1)

    samples = [('label', 'tests/data/img_formats/nut_color.gif')]
    img_samples = samples >> ReadImage(1, as_grey=False) >> Collect()
    assert img_samples[0][1].shape == (213, 320, 3)
    img_samples = samples >> ReadImage(1, as_grey=True) >> Collect()
    assert img_samples[0][1].shape == (213, 320)

    samples = ['tests/data/img_formats/nut_color.gif']
    img_samples = samples >> ReadImage(None, as_grey=False) >> Collect()
    assert img_samples[0][0].shape == (213, 320, 3)

    samples = ['tests/data/img_formats/nut_color.gif']
    img_samples = samples >> ReadImage(None, dtype=float) >> Collect()
    assert img_samples[0][0].dtype == float


def test_ReadPandas_isnull():
    assert not ReadPandas.isnull(1.0)
    assert not ReadPandas.isnull(0)
    assert ReadPandas.isnull(None)
    assert ReadPandas.isnull(np.NaN)


def test_ReadPandas_pkl():
    df = pd.read_csv('tests/data/pandas_table.csv')
    df.to_pickle('tests/data/pandas_table.pkl')
    assert True


def test_ReadPandas():
    for ext in ['.pkl', '.csv', '.tsv', '.xlsx']:
        filepath = 'tests/data/pandas_table' + ext

        samples = ReadPandas(filepath, dropnan=True) >> Collect()
        Row = namedtuple('Row', 'col1,col2')
        expected = [Row(col1=1.0, col2=4.0), Row(col1=3.0, col2=6.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath, dropnan=True, rowname='R') >> Collect()
        R = namedtuple('R', 'col1,col2')
        expected = [R(col1=1.0, col2=4.0), R(col1=3.0, col2=6.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath, dropnan=False) >> Collect()
        Row = namedtuple('Row', 'col1,col2')
        expected = [Row(col1=1.0, col2=4.0), Row(col1=2.0, col2=np.NaN),
                    Row(col1=3.0, col2=6.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath, replacenan=None) >> Collect()
        Row = namedtuple('Row', 'col1,col2')
        expected = [Row(col1=1.0, col2=4.0), Row(col1=2.0, col2=None),
                    Row(col1=3.0, col2=6.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath, colnames=['col2', 'col1']) >> Collect()
        Row = namedtuple('Row', 'col2,col1')
        expected = [Row(col2=4.0, col1=1.0), Row(col2=6.0, col1=3.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath, colnames=['col1']) >> Collect()
        Row = namedtuple('Row', 'col1')
        expected = [Row(col1=1.0), Row(col1=2.0), Row(col1=3.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath, colnames=['col2']) >> Collect()
        Row = namedtuple('Row', 'col2')
        expected = [Row(col2=4.0), Row(col2=6.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath,
                             colnames=['col2'], replacenan='NA') >> Collect()
        Row = namedtuple('Row', 'col2')
        expected = [Row(col2=4.0), Row(col2='NA'), Row(col2=6.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath,
                             rows='col1 > 1', replacenan=0) >> Collect()
        Row = namedtuple('Row', 'col1,col2')
        expected = [Row(col1=2.0, col2=0), Row(col1=3.0, col2=6.0)]
        nt.assert_equal(samples, expected)

        samples = ReadPandas(filepath,
                             rows='col1 < 3', colnames=['col1']) >> Collect()
        Row = namedtuple('Row', 'col1')
        expected = [Row(col1=1), Row(col1=2)]
        nt.assert_equal(samples, expected)
