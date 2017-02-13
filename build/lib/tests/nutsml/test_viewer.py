"""
.. module:: test_viewer
   :synopsis: Unit tests for viewer module
"""

import pytest
import numpy as np

from nutsflow import Consume
from nutsflow.common import Redirect
from nutsml import PrintImageInfo, PrintTypeInfo


@pytest.fixture('function')
def expected_type_info():
    return """
Row 0
  Col 0: <ndarray> shape:10x20x3, dtype:float64, min:0.0, max:0.0
  Col 1: <int> 1

Row 1
  Col 0: <str> text
  Col 1: <int> 2

Row 2
  Col 0: <int> 3
"""


def test_PrintImageInfo():
    with Redirect() as out:
        data = [(np.zeros((10, 20, 3)), 1)]
        data >> PrintImageInfo(0) >> Consume()
    expected = 'Image shape:10x20x3, dtype:float64, min:0.0, max:0.0\n'
    assert out.getvalue() == expected

    with pytest.raises(ValueError) as ex:
        ['invalid'] >> PrintImageInfo(0) >> Consume()
    assert str(ex.value).startswith('Expect image but get')


def test_PrintTypeInfo(expected_type_info):
    with Redirect() as out:
        data = [(np.zeros((10, 20, 3)), 1), ('text', 2), 3]
        data >> PrintTypeInfo() >> Consume()
    assert out.getvalue() == expected_type_info
