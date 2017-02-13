"""
.. module:: test_logger
   :synopsis: Unit tests for logger module
"""

import pytest
import os
import numpy as np

from nutsflow import Collect
from nutsml import LogCols


@pytest.fixture('function')
def filepath():
    filepath = 'tests/data/temp_logger.csv'

    def fin():
        if os.path.exists(filepath):
            os.remove(filepath)

    return filepath


def test_Logger(filepath):
    data = [[1, 2], [3, 4]]

    with LogCols(filepath) as log_cols:
        assert data >> log_cols >> Collect() == data
    assert open(filepath).read() == '1,2\n3,4\n'

    with LogCols(filepath, delimiter='; ') as log_cols:
        assert data >> log_cols >> Collect() == data
    assert open(filepath).read() == '1; 2\n3; 4\n'

    with LogCols(filepath, cols=0, reset=True) as log_cols:
        assert data >> log_cols >> Collect() == data
        assert data >> log_cols >> Collect() == data
    assert open(filepath).read() == '1\n3\n1\n3\n'

    with LogCols(filepath, cols=(1, 0), colnames=('a', 'b')) as log_cols:
        assert data >> log_cols >> Collect() == data
    assert open(filepath).read() == 'a,b\n2,1\n4,3\n'


def test_Logger_reset(filepath):
    data = [[1, 2], [3, 4]]

    with LogCols(filepath, cols=0, reset=True) as log_cols:
        assert data >> log_cols >> Collect() == data
    assert open(filepath).read() == '1\n3\n'

    with LogCols(filepath, cols=1, reset=False) as log_cols:
        assert data >> log_cols >> Collect() == data
    assert open(filepath).read() == '1\n3\n2\n4\n'


def test_Logger_numpy(filepath):
    data = [np.array([1, 2]), np.array([3, 4])]
    with LogCols(filepath) as log_cols:
        assert data >> log_cols >> Collect() == data
    assert open(filepath).read() == '1,2\n3,4\n'

    data = [np.array(1), np.array(2)]
    with LogCols(filepath) as log_cols:
        assert data >> log_cols >> Collect() == data
    assert open(filepath).read() == '1\n2\n'


def test_Logger_delete(filepath):
    data = [[1, 2], [3, 4]]

    log_cols = LogCols(filepath)
    assert data >> log_cols >> Collect() == data
    assert os.path.exists(filepath)
    log_cols.delete()
    assert not os.path.exists(filepath)
