"""
.. module:: viewer
   :synopsis: Viewing of sample data
"""

from __future__ import print_function
from __future__ import absolute_import

import time

import numpy as np
import nutsml.imageutil as iu

from six.moves import range
from nutsflow import NutFunction, nut_function
from nutsflow.common import as_tuple, as_set, stype
from matplotlib import pyplot as plt


# TODO: Fix deprecation warning
# MatplotlibDeprecationWarning: Using default event loop until function specific
# to this GUI is implemented
class ViewImage(NutFunction):  # pragma no coverage
    """
    Display images in window.
    """

    def __init__(self, imgcols, layout=(1, None), figsize=None,
                 pause=0.0001, axis_off=False, labels_off=False, titles=None,
                 every_sec=0, every_n=0, **imargs):
        """
        iterable >> ViewImage(imgcols, layout=(1, None), figsize=None,
        **plotargs)

        |  Images should be numpy arrays in one of the following formats:
        |  MxN - luminance (grayscale, float array only)
        |  MxNx3 - RGB (float or uint8 array)
        |  MxNx4 - RGBA (float or uint8 array)

        Shapes with single-dimension axis are supported but not encouraged,
        e.g. MxNx1 will be converted to MxN.

        See
        http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow

        >>> from nutsflow import Consume
        >>> from nutsml import ReadImage

        >>> imagepath = 'tests/data/img_formats/*.jpg'
        >>> samples = [(1, 'nut_color'), (2, 'nut_grayscale')]
        >>> read_image = ReadImage(1, imagepath)
        >>> samples >> read_image >> ViewImage(1) >> Consume() # doctest: +SKIP

        >>> view_gray = ViewImage(1, cmap='gray')
        >>> samples >> read_image >> view_gray >> Consume() # doctest: +SKIP

        :param int|tuple|None imgcols: Index or tuple of indices of data columns
               containing images (ndarray). Use None if images are provided
               directly, e.g. [img1, img2, ...] >> ViewImage(None) >> Consume()
        :param tuple layout: Rows and columns of the viewer layout., e.g.
               a layout of (2,3) means that 6 images in the data are
               arranged in 2 rows and 3 columns.
               Number of cols can be None is then derived from imgcols
        :param tuple figsize: Figure size in inch.
        :param float pause: Waiting time in seconds after each plot.
               Pressing a key skips the waiting time.
        :param bool axis_off: Enable or disable display of figure axes.
        :param bool lables_off: Enable or disable display of axes labels.
        :param float every_sec: View every given second, e.g. to print
                every 2.5 sec every_sec = 2.5
        :param int every_n: View every n-th call.
        :param kwargs imargs: Keyword arguments passed on to matplotlib's
            imshow() function, e.g. cmap='gray'.  See
            http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow
        """
        imgcols = (None,) if imgcols is None else as_tuple(imgcols)
        r, c, n = layout[0], layout[1], len(imgcols)
        if c is None:
            c = n
        if n != r * c:
            raise ValueError("Number of images and layout don't match!")

        fig = plt.figure(figsize=figsize)
        fig.canvas.set_window_title('ViewImage')
        self.axes = [fig.add_subplot(r, c, i + 1) for i in range(n)]
        self.axis_off = axis_off
        self.labels_off = labels_off
        self.titles = titles
        self.imgcols = imgcols
        self.pause = pause
        self.cnt = 0
        self.time = time.time()
        self.every_sec = every_sec
        self.every_n = every_n
        self.imargs = imargs

    def __delta_sec(self):
        """Return time in seconds (float) consumed between prints so far"""
        return time.time() - self.time

    def __should_view(self):
        """Return true if data should be viewed"""
        self.cnt += 1
        return (self.cnt >= self.every_n and
                self.__delta_sec() >= self.every_sec)

    def __call__(self, data):
        """
        View the images in data

        :param tuple data: Data with images at imgcols.
        :return: unchanged input data
        :rtype: tuple
        """
        if not self.__should_view():
            return data

        self.cnt = 0
        self.time = time.time()

        for i, (imgcol, ax) in enumerate(zip(self.imgcols, self.axes)):
            ax.clear()
            if self.axis_off:
                ax.set_axis_off()
            if self.labels_off:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            if self.titles:
                ax.set_title(self.titles[i])
            img = data if imgcol is None else data[imgcol]
            img = np.squeeze(img)
            ax.imshow(img, **self.imargs)
            ax.figure.canvas.draw()
        plt.waitforbuttonpress(timeout=self.pause)  # or plt.pause(self.pause)
        return data


class ViewImageAnnotation(NutFunction):  # pragma no coverage
    """
    Display images and annotation in window.
    """

    TEXTPROP = {'edgecolor': 'k', 'backgroundcolor': (1, 1, 1, 0.5)}
    SHAPEPROP = {'edgecolor': 'y', 'facecolor': 'none', 'linewidth': 1}

    def __init__(self, imgcol, annocols, figsize=None,
                 pause=0.0001, interpolation=None, **annoargs):
        """
        iterable >> ViewImageAnnotation(imgcol, annocols, figsize=None,
                                        pause, interpolation, **annoargs)

        |  Images must be numpy arrays in one of the following formats:
        |  MxN - luminance (grayscale, float array only)
        |  MxNx3 - RGB (float or uint8 array)
        |  MxNx4 - RGBA (float or uint8 array)
        |  See
        |  http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow

        Shapes with single-dimension axis are supported but not encouraged,
        e.g. MxNx1 will be converted to MxN.

        :param int imgcol: Index of data column that contains the image
        :param int|tuple annocols: Index or tuple of indices specifying the data
               column(s) that contain annotation (labels, or geometry)
        :param tuple figsize: Figure size in inch.
        :param float pause: Waiting time in seconds after each plot.
               Pressing a key skips the waiting time.
        :param string interpolation: Interpolation for imshow, e.g.
                'nearest', 'bilinear', 'bicubic'. for details see
                http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot
                .imshow
        :param kwargs annoargs: Keyword arguments for visual properties of
               annotation, e.g.  edgecolor='y', linewidth=1
        """
        fig = plt.figure(figsize=figsize)
        fig.canvas.set_window_title('ViewImageAnnotation')
        self.axes = fig.add_subplot(111)
        self.imgcol = imgcol
        self.annocols = as_set(annocols)
        self.pause = pause
        self.interpolation = interpolation
        self.annoargs = annoargs

    def _shapeprops(self):
        """Return shape properties from kwargs or default value."""
        aa = ViewImageAnnotation.SHAPEPROP.copy()
        aa.update(self.annoargs)
        return aa

    def _textprop(self, key):
        """Return text property from kwargs or default value."""
        return self.annoargs.get(key, ViewImageAnnotation.TEXTPROP[key])

    def __call__(self, data):
        """
        View the image and its annotation

        :param tuple data: Data with image at imgcol and annotation at annocol.
        :return: unchanged input data
        :rtype: tuple
        """
        img = np.squeeze(data[self.imgcol])
        ax = self.axes
        ax.clear()
        ax.imshow(img, interpolation=self.interpolation)
        labelcol = 0.7
        for acol in self.annocols:
            annos = data[acol]
            if isinstance(annos, (list, tuple)):
                props = self._shapeprops()
                for anno in iu.annotation2pltpatch(annos, **props):
                    ax.add_patch(anno)
            else:
                fs = ax.get_window_extent().height / 22
                p = img.shape[0] / 6
                x, y = p / 2, p * labelcol
                labelcol += 1
                ax.text(x, y, str(annos),
                        color=self._textprop('edgecolor'),
                        backgroundcolor=self._textprop('backgroundcolor'),
                        size=fs, family='monospace')
        ax.figure.canvas.draw()
        plt.waitforbuttonpress(timeout=self.pause)
        return data
