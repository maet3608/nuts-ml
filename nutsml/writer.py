"""
.. module:: writer
   :synopsis: Writing of sample and image data
"""
from __future__ import absolute_import
import os

import skimage.io as sio

from inspect import isfunction
from nutsml.fileutil import create_folders
from nutsflow.base import NutFunction
from nutsflow.source import Enumerate


class WriteImage(NutFunction):
    """
    Write images within samples.
    """

    def __init__(self, column, pathfunc, namefunc=None):
        """
        Write images within samples to file.

        Writes jpg, gif, png, tif and bmp format depending on file extension.
        Images in samples are expected to be numpy arrays.
        See nutsml.util.load_image for details.

        Folders on output file path are created if missing.

        >>> from nutsml import ReadImage
        >>> from nutsflow import Collect, Get, GetCols, Consume, Unzip
        >>> samples = [('nut_color', 1), ('nut_grayscale', 2)]
        >>> inpath = 'tests/data/img_formats/*.bmp'
        >>> img_samples = samples >> ReadImage(0, inpath) >> Collect()

        >>> imagepath = 'tests/data/test_*.bmp'
        >>> names = samples >> Get(0) >> Collect()
        >>> img_samples >> WriteImage(0, imagepath, names) >> Consume()

        >>> imagepath = 'tests/data/test_*.bmp'
        >>> names = samples >> Get(0) >> Collect()
        >>> images = img_samples >> Get(0)
        >>> images >> WriteImage(None, imagepath, names) >> Consume()

        >>> imagepath = 'tests/data/test_*.bmp'
        >>> namefunc = lambda sample: sample[1]
        >>> (samples >> GetCols(0,0,1) >> ReadImage(0, inpath) >>
        ... WriteImage(0, imagepath, namefunc) >> Consume())

        :param int|None column: Column in sample that contains image or
              take sample itself if column is None.
        :param str|function pathfunc: Filepath with wildcard '*',
            which is replaced by the name provided names e.g.
            'tests/data/img_formats/*.jpg' for names = ['nut_grayscale']
            will become 'tests/data/img_formats/nut_grayscale.jpg'
            or
            Function to compute path to image file from sample and name, e.g.
            pathfunc=lambda sample, name: 'tests/data/test_{}.jpg'.format(name)
        :param iterable|function|None namefunc: Iterable over names to generate
            image paths from (length need to be the same as samples),
            or
            Function to compute filenames from sample, e.g.
            namefunc=lambda samples: sample[0]
            if None, Enumerate() is used.
        """
        namefunc = Enumerate() if namefunc is None else namefunc
        self.namefunc = namefunc if isfunction(namefunc) else iter(namefunc)
        self.column = column
        self.pathfunc = pathfunc

    def __call__(self, sample):
        """Return sample and write image within sample"""
        pathfunc, namefunc = self.pathfunc, self.namefunc
        name = namefunc(sample) if isfunction(namefunc) else next(namefunc)

        if isinstance(pathfunc, str):
            filepath = pathfunc.replace('*', str(name))
        elif isfunction(pathfunc):
            filepath = pathfunc(sample, name)
        else:
            raise ValueError('Expect path or function: ' + str(pathfunc))

        create_folders(os.path.split(filepath)[0])
        img = sample if self.column is None else sample[self.column]
        sio.imsave(filepath, img)
        return sample
