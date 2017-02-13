"""
.. module:: test_reader
   :synopsis: Unit tests for reader module
"""

import pytest

import pandas as pd
import numpy as np
import numpy.testing as nt

from dplython import DplyFrame, select, X
from nutsflow import Collect
from nutsml import DplyToList, ReadImage, ReadLabelDirs, ReadPandas


def test_DplyToList():
    empty_dplyframe = DplyFrame(pd.DataFrame())
    assert empty_dplyframe >> DplyToList() >> Collect() == []

    pandasframe = pd.DataFrame(data={'c1': [1, 2, 3], 'c2': [4, 5, 6]})
    dplyframe = DplyFrame(pandasframe)
    assert dplyframe >> DplyToList() >> Collect() == [[1, 4], [2, 5], [3, 6]]

    with pytest.raises(ValueError) as ex:
        [1] >> DplyToList() >> Collect()
    assert str(ex.value) == 'Expect Dplyr dataframe!'


def test_ReadLabelDirs():
    read = ReadLabelDirs('tests/data/labeldirs', '*.txt')
    samples = read >> Collect()
    assert samples == [('tests/data/labeldirs/0/test0.txt', '0'),
                       ('tests/data/labeldirs/1/test1.txt', '1'),
                       ('tests/data/labeldirs/1/test11.txt', '1')]


def test_ReadImage():
    arr0 = np.load('tests/data/img_arrays/nut_color.jpg.npy')
    arr1 = np.load('tests/data/img_arrays/nut_grayscale.jpg.npy')
    samples = [('nut_color', 1), ('nut_grayscale', 2)]

    imagepath = 'tests/data/img_formats/*.jpg'
    img_samples = samples >> ReadImage(0, imagepath) >> Collect()
    nt.assert_equal(img_samples[0][0], arr0)
    nt.assert_equal(img_samples[1][0], arr1)
    assert img_samples[0][1] == 1
    assert img_samples[1][1] == 2

    pathfunc = lambda sample: 'tests/data/img_formats/{0}.jpg'.format(*sample)
    img_samples = samples >> ReadImage(0, pathfunc) >> Collect()
    nt.assert_equal(img_samples[0][0], arr0)
    nt.assert_equal(img_samples[1][0], arr1)

    samples = [('label', 'tests/data/img_formats/nut_color.jpg')]
    img_samples = samples >> ReadImage(1, as_grey=False) >> Collect()
    assert img_samples[0][1].shape == (213, 320, 3)
    img_samples = samples >> ReadImage(1, as_grey=True) >> Collect()
    assert img_samples[0][1].shape == (213, 320)


def test_ReadPandas_isnull():
    assert not ReadPandas.isnull(1.0)
    assert not ReadPandas.isnull(0)
    assert ReadPandas.isnull(None)
    assert ReadPandas.isnull(np.NaN)


def test_ReadPandas_dply():
    filepath = 'tests/data/pandas_table.csv'
    samples = ReadPandas(filepath).dply() >> select(X.col1) >> DplyToList()
    nt.assert_equal(samples, [[1], [2], [3]])


def test_ReadPandas_replacenan():
    data = [1, np.NaN, 2]
    nt.assert_equal(ReadPandas._replacenan(data), [1, None, 2])


def test_ReadPandas_pkl():
    # create pickle version of table from CSV table
    df = pd.read_csv('tests/data/pandas_table.csv')
    df.to_pickle('tests/data/pandas_table.pkl')

    for ext in ['.pkl', '.csv', '.tsv', '.xlsx']:
        filepath = 'tests/data/pandas_table' + ext
        samples = ReadPandas(filepath, dropnan=True) >> Collect()
        nt.assert_equal(samples, [[1, 4], [3, 6]])

        samples = ReadPandas(filepath, dropnan=False) >> Collect()
        nt.assert_equal(samples, [[1, 4], [2, np.NaN], [3, 6]])

        samples = ReadPandas(filepath, replacenan=True) >> Collect()
        nt.assert_equal(samples, [[1, 4], [2, None], [3, 6]])

        samples = ReadPandas(filepath, columns=['col1', 'col2']) >> Collect()
        nt.assert_equal(samples, [[1, 4], [3, 6]])

        samples = ReadPandas(filepath, columns=['col1']) >> Collect()
        nt.assert_equal(samples, [[1], [2], [3]])

        samples = ReadPandas(filepath, columns=['col2']) >> Collect()
        nt.assert_equal(samples, [[4], [6]])

        samples = ReadPandas(filepath,
                             columns=['col2'], replacenan=True) >> Collect()
        nt.assert_equal(samples, [[4], [None], [6]])

        samples = ReadPandas(filepath,
                             rows='col1 > 1', replacenan=True) >> Collect()
        nt.assert_equal(samples, [[2, None], [3, 6]])

        samples = ReadPandas(filepath,
                             rows='col1 < 3', columns=['col1']) >> Collect()
        nt.assert_equal(samples, [[1], [2]])
