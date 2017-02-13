"""
.. module:: test_config
   :synopsis: Unit tests for config module
"""

import pytest

import nutsml.config as nc


def test_load_config():
    cfg = nc.load_config('tests/data/config.yaml')
    assert cfg.filepath == 'c:/Maet'
    assert cfg['imagesize'] == [100, 200]

    with pytest.raises(IOError) as ex:
        nc.load_config('does not exist')
    assert str(ex.value).startswith('Configuration file not found')
