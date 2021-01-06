"""
.. module:: test_logger
   :synopsis: Unit tests for logger module
"""

import pytest
import os
import numpy as np

from nutsflow import Collect
from nutsml import LogToFile, LogCols


@pytest.fixture(scope='function')
def filepath():
    filepath = 'tests/data/temp_logger.csv'

    def fin():
        if os.path.exists(filepath):
            os.remove(filepath)

    return filepath


@pytest.mark.filterwarnings('ignore:LogCols is deprecated. Use LogToFile!')
def test_LogCols(filepath):
    data = [[1, 2], [3, 4]]

    with LogCols(filepath) as logcols:
        assert data >> logcols >> Collect() == data
    with open(filepath) as f:
        assert f.read() == '1,2\n3,4\n'


def test_LogToFile(filepath):
    data = [[1, 2], [3, 4]]

    with LogToFile(filepath) as logtofile:
        assert data >> logtofile >> Collect() == data
    with open(filepath) as f:
        assert f.read() == '1,2\n3,4\n'

    with LogToFile(filepath, delimiter='; ') as logtofile:
        assert data >> logtofile >> Collect() == data
    with open(filepath) as f:
        assert f.read() == '1; 2\n3; 4\n'

    with LogToFile(filepath, cols=0, reset=True) as logtofile:
        assert data >> logtofile >> Collect() == data
        assert data >> logtofile >> Collect() == data
    with open(filepath) as f:
        assert f.read() == '1\n3\n1\n3\n'

    with LogToFile(filepath, cols=(1, 0), colnames=('a', 'b')) as logtofile:
        assert data >> logtofile >> Collect() == data
    with open(filepath) as f:
        assert f.read() == 'a,b\n2,1\n4,3\n'


def test_LogToFile_reset(filepath):
    data = [[1, 2], [3, 4]]

    with LogToFile(filepath, cols=0, reset=True) as logtofile:
        assert data >> logtofile >> Collect() == data
    with open(filepath) as f:
        assert f.read() == '1\n3\n'

    with LogToFile(filepath, cols=1, reset=False) as logtofile:
        assert data >> logtofile >> Collect() == data
    with open(filepath) as f:
        assert f.read() == '1\n3\n2\n4\n'


def test_LogToFile_numpy(filepath):
    data = [np.array([1, 2]), np.array([3, 4])]
    with LogToFile(filepath) as logtofile:
        assert data >> logtofile >> Collect() == data
    with open(filepath) as f:
        assert f.read() == '1,2\n3,4\n'

    data = [np.array(1), np.array(2)]
    with LogToFile(filepath) as logtofile:
        assert data >> logtofile >> Collect() == data
    with open(filepath) as f:
        assert f.read() == '1\n2\n'


def test_LogToFile_delete(filepath):
    data = [[1, 2], [3, 4]]

    logtofile = LogToFile(filepath)
    assert data >> logtofile >> Collect() == data
    assert os.path.exists(filepath)
    logtofile.delete()
    assert not os.path.exists(filepath)
