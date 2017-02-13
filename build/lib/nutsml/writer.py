"""
.. module:: writer
   :synopsis: Writing of sample and image data
"""

import skimage.io as sio

from nutsflow.base import NutFunction
from nutsflow.source import Enumerate


class WriteImage(NutFunction):
    """
    Write images within samples.
    """

    def __init__(self, column, pathfunc, names=Enumerate()):
        """
        Write images within samples to file.

        Writes jpg, gif, png, tif and bmp format depending on file extension.
        Images in samples are expected to be numpy arrays.
        See nutsml.util.load_image for details.

        >>> from nutsml import ReadImage
        >>> from nutsflow import Collect, Get, Consume
        >>> samples = [('nut_color', 1), ('nut_grayscale', 2)]
        >>> inpath = 'tests/data/img_formats/*.bmp'
        >>> img_samples = samples >> ReadImage(0, inpath) >> Collect()

        >>> imagepath = 'tests/data/test_*.bmp'
        >>> names = samples >> Get(0) >> Collect()
        >>> img_samples >> WriteImage(0, imagepath, names) >> Consume()

        :param int column: Column in sample that contains image.
        :param str|function pathfunc: Filepath with wildcard '*',
            which is replaced by the name provided names e.g.
            'tests/data/img_formats/*.jpg' for names = ['nut_grayscale']
            will become 'tests/data/img_formats/nut_grayscale.jpg'
            or
            Function to compute path to image file from sample and name, e.g.
            pathfunc = lambda sample, name: 'tests/data/test_{}.jpg'.format(name)
        :param iterable names: Iterable over names to generate image paths from.
            Length need to be the same as samples.
        """
        self.column = column
        self.names = iter(names)
        self.pathfunc = pathfunc

    def __call__(self, sample):
        """Return sample and write image within sample"""
        name = next(self.names)
        if isinstance(self.pathfunc, str):
            filepath = self.pathfunc.replace('*', str(name))
        elif hasattr(self.pathfunc, '__call__'):
            filepath = self.pathfunc(sample, name)
        else:
            raise ValueError('Expect path or function: ' + str(self.pathfunc))
        sio.imsave(filepath, sample[self.column])
        return sample
