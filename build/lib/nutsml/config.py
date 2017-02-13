"""
.. module:: config
   :synopsis: Handling of configuration files.
"""

import os
import yaml
from nutsml.datautil import AttributeDict


def load_config(filename):
    """
    Load configuration file in YAML format.

    The search order for the config file is:
    1) user home dir
    2) current dir
    3) full path

    >>> cfg = load_config('tests/data/config.yaml')
    >>> cfg.filepath
    'c:/Maet'

    >>> cfg['imagesize']
    [100, 200]

    :param filename: Name or fullpath of configuration file.
    :return: dictionary with config data. Note that config data can be
             accessed by key or attribute, e.g. cfg.filepath or cfg.['filepath']
    :rtype: AttributeDict
    """
    filepaths = []
    for dirpath in os.path.expanduser('~'), os.curdir, '':
        try:
            filepath = os.path.join(dirpath, filename)
            filepaths.append(filepath)
            with open(filepath, 'r') as f:
                return AttributeDict(yaml.load(f))
        except IOError:
            pass
    raise IOError('Configuration file not found: ' + ', '.join(filepaths))
