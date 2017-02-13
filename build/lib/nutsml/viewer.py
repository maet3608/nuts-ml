"""
.. module:: viewer
   :synopsis: Viewing of sample data
"""

import numpy as np
import itertools as itt
import nutsml.imageutil as iu

from datautil import shapestr
from nutsflow import NutFunction, nut_function, as_tuple, as_set
from matplotlib import pyplot as plt


@nut_function
def PrintImageInfo(data, imgcol):
    """
    iterable >> PrintImageInfo(imgcol)

    Print shape and type information of image at imgcol in data item.

    >>> data = [(np.zeros((10, 20, 3)), 1), (np.ones((15, 5, 3)), 2)]
    >>> data >> PrintImageInfo(0) >> Consume()
    Image shape:10x20x3, dtype:float64, min:0.0, max:0.0
    Image shape:15x5x3, dtype:float64, min:1.0, max:1.0

    :param tuple data: Tuple that contains an image (ndarray) at imgcol.
    :param int imgcol: Index of item in data that is image.
    :return: input data unchanged
    :rtype: same as data
    """
    img = data[imgcol]
    if not isinstance(img, np.ndarray):
        raise ValueError('Expect image but get: ' + type(img).__name__)
    text = 'Image shape:{}, dtype:{}, min:{}, max:{}'
    print text.format(shapestr(img), img.dtype, np.min(img), np.max(img))
    return data


@nut_function
def PrintTypeInfo(data, counter=itt.count()):
    """
    iterable >> PrintTypeInfo()

    Print type information for all items in data.

    >>> data = [(np.zeros((10, 20, 3)), 1), ('text', 2), 3]
    >>> data >> PrintTypeInfo() >> Consume()
    Row 0
      Col 0: <ndarray> shape:10x20x3, dtype:float64, min:0.0, max:0.0
      Col 1: <int> 1

    Row 1
      Col 0: <str> text
      Col 1: <int> 2

    Row 2
      Col 0: <int> 3

    :param any data: Any type of fas
    :param iterable counter: Infinite iterable that produces row indices.
    :return: input data unchanged
    :rtype: same as data
    """
    print '\nRow {}'.format(next(counter))
    for i, e in enumerate(as_tuple(data)):
        print '  Col {}: <{}>'.format(i, type(e).__name__),
        if isinstance(e, np.ndarray):
            text = 'shape:{}, dtype:{}, min:{}, max:{}'
            print text.format(shapestr(e), e.dtype, np.min(e), np.max(e))
        else:
            print '{}'.format(str(e))
    return data


# TODO: Fix deprecation warning
# MatplotlibDeprecationWarning: Using default event loop until function specific
# to this GUI is implemented
class ViewImage(NutFunction):  # pragma no coverage
    """
    Display images in window.
    """

    def __init__(self, imgcols, layout=(1, None), figsize=None,
                 pause=0.0001, **imargs):
        """
        iterable >> ViewImage(imgcols, layout=(1, None), figsize=None, **plotargs)

        Images must be numpy arrays in one of the following formats:
        MxN - luminance (grayscale, float array only)
        MxNx3 - RGB (float or uint8 array)
        MxNx4 - RGBA (float or uint8 array)

        See
        http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow

        Example
        >>> from nutsflow import Consume
        >>> from nutsml import ReadImage
        >>> imagepath = 'tests/data/img_formats/*.jpg'
        >>> samples = [(1, 'nut_color'), (2, 'nut_grayscale')]
        >>> read_image = ReadImage(1, imagepath)
        >>> samples >> read_image >> ViewImage(1) >> Consume() # doctest: +SKIP

        :param int|tuple imgcols: Index or tuple of indices of data columns
               containing images (ndarray)
        :param tuple layout: Rows and columns of the viewer layout., e.g.
               a layout of (2,3) means that 6 images in the data are
               arranged in 2 rows and 3 columns.
               Number of cols can be None is then derived from imgcols
        :param tuple figsize: Figure size in inch.
        :param float pause: Waiting time in seconds after each plot.
               Pressing a key skips the waiting time.
        :param kwargs imargs: Keyword arguments passed on to matplotlib's
            imshow() function. See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
        """
        imgcols = as_tuple(imgcols)
        r, c, n = layout[0], layout[1], len(imgcols)
        if c is None:
            c = n
        if n != r * c:
            raise ValueError("Number of images and layout don't match!")

        fig = plt.figure(figsize=figsize)
        fig.canvas.set_window_title('ViewImage')
        self.axes = [fig.add_subplot(r, c, i + 1) for i in xrange(n)]
        self.imgcols = imgcols
        self.pause = pause
        self.imargs = imargs

    def __call__(self, data):
        """
        View the images in data

        :param tuple data: Data with images at imgcols.
        :return: unchanged input data
        :rtype: tuple
        """
        for imgcol, ax in zip(self.imgcols, self.axes):
            ax.clear()
            ax.imshow(data[imgcol], **self.imargs)
            ax.figure.canvas.draw()
        plt.waitforbuttonpress(timeout=self.pause)  # or plt.pause(self.pause)
        return data


class ViewImageAnnotation(NutFunction):  # pragma no coverage
    """
    Display images and annotation in window.
    """

    def __init__(self, imgcol, annocols, figsize=None,
                 pause=0.0001, **annoargs):
        """
        iterable >> ViewImageAnnotation(imgcol, annocols, figsize=None, **plotargs)

        Images must be numpy arrays in one of the following formats:
        MxN - luminance (grayscale, float array only)
        MxNx3 - RGB (float or uint8 array)
        MxNx4 - RGBA (float or uint8 array)
        See
        http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow


        :param int imgcol: Index of data column that contains the image
        :param int|tuple annocols: Index or tuple of indices specifying the data
               column(s) that contain annotation (labels, or geometry)
        :param tuple figsize: Figure size in inch.
        :param float pause: Waiting time in seconds after each plot.
               Pressing a key skips the waiting time.
        :param kwargs annoargs: Keyword arguments for visual properties of
               annotation, e.g.  edgecolor='y', linewidth=1
        """

        fig = plt.figure(figsize=figsize)
        fig.canvas.set_window_title('ViewImageAnnotation')
        self.axes = fig.add_subplot(111)
        self.imgcol = imgcol
        self.annocols = as_set(annocols)
        self.pause = pause
        self.annoargs = self._getargs(annoargs)

    def _getargs(self, kwargs):
        """Return default values for annotation drawing if not provided."""
        defargs = {'edgecolor': 'y', 'facecolor': 'none', 'linewidth': 1}
        defargs.update(kwargs)
        return defargs

    def __call__(self, data):
        """
        View the image and its annotation

        :param tuple data: Data with image at imgcol and annotation at annocol.
        :return: unchanged input data
        :rtype: tuple
        """
        img = data[self.imgcol]
        ax = self.axes
        ax.clear()
        ax.imshow(img)
        labelcol = 1
        for acol in self.annocols:
            annos = data[acol]
            if hasattr(annos, '__iter__'):
                for anno in iu.annotation2pltpatch(annos, **self.annoargs):
                    ax.add_patch(anno)
            else:
                color = self.annoargs['edgecolor']
                x = img.shape[0] / 10
                y = x * labelcol
                labelcol += 1
                ax.text(x, y, str(annos), color=color, fontsize=20)
        ax.figure.canvas.draw()
        plt.waitforbuttonpress(timeout=self.pause)  # or plt.pause(self.pause)
        return data
