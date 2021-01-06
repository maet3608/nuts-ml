"""
.. module:: plotter
   :synopsis: Plotting of data, e.g. loss over epochs
"""
import time

import itertools as itt
import matplotlib.pyplot as plt

from six.moves import range
from nutsflow import NutFunction
from nutsflow.common import as_tuple, as_list


class PlotLines(NutFunction):  # pragma no coverage
    """
    Plot line graph for selected data columns.
    """

    def __init__(self, ycols,
                 xcols=None,
                 layout=(1, None),
                 titles=None,
                 every_sec=0,
                 every_n=0,
                 filterfunc=lambda data: True,
                 figsize=None,
                 filepath=None):
        """
        iterable >> PlotLines(ycols) >> Consume()

        >>> import os
        >>> import numpy as np
        >>> from nutsflow import Consume

        >>> fp = 'tests/data/temp_plotter.png'
        >>> xs = np.arange(0, 6.3, 1.2)
        >>> ysin, ycos = np.sin(xs),  np.cos(xs)
        >>> data = zip(xs, ysin, ycos)

        >>> data >> PlotLines(1, 0, filepath=fp) >> Consume()

        >>> list(ycos) >> PlotLines(0, filepath=fp) >> Consume()

        >>> data >> PlotLines(ycols=(1,2), filepath=fp) >> Consume()

        >>> ysin.tolist() >> PlotLines(ycols=None, filepath=fp) >> Consume()

        >>> if os.path.exists(fp): os.remove(fp)

        :param int|tuple|None ycols: Index or tuple of indices of the 
            data columns that contain the y-data for the plot.
            If None data is used directly.
        :param int|tuple|function|iterable|None xcols: Index or tuple of indices
            of the data columns that contain the x-data for the plot.
            Alternatively an iterator or a function can be provided that
            generates the x-data for the plot, e.g. xcols = itertools.count()
            or xcols = lambda: epoch
            For xcols==None, itertools.count() will be used.
        :param tuple layout: Rows and columns of the plotter layout., e.g.
               a layout of (2,3) means that 6 plots in the data are
               arranged in 2 rows and 3 columns.
               Number of cols can be None is then derived from ycols
        :param float every_sec: Plot every given second, e.g. to plot
                every 2.5 sec every_sec = 2.5
        :param int every_n: Plot every n-th call.
        :param function filterfunc: Boolean function to filter plot data.
        :param tuple figsize: Figure size in inch.
        :param filepath: Path to a file to draw plot to. If provided the
               plot will not appear on the screen.
        :return: Returns input unaltered
        :rtype: any
        """
        self.ycols = [-1] if ycols is None else as_list(ycols)
        self.xcols = itt.count() if xcols is None else xcols
        self.filepath = filepath
        self.figsize = figsize
        self.titles = titles
        self.cnt = 0
        self.time = time.time()
        self.filterfunc = filterfunc
        self.every_sec = every_sec
        self.every_n = every_n
        r, c, n = layout[0], layout[1], len(self.ycols)
        if c is None:
            c = n
        self.figure = plt.figure(figsize=figsize)
        self.axes = [self.figure.add_subplot(r, c, i + 1) for i in range(n)]
        self.reset()

    def __delta_sec(self):
        """Return time in seconds (float) consumed between plots so far"""
        return time.time() - self.time

    def __should_plot(self, data):
        """Return true if data should be plotted"""
        self.cnt += 1
        return (self.filterfunc(data) and
                self.cnt >= self.every_n and
                self.__delta_sec() >= self.every_sec)

    def reset(self):
        """Reset plot data"""
        self.xdata, self.ydata = [], []
        for _ in self.ycols:
            self.xdata.append([])
            self.ydata.append([])

    def _add_data(self, data):
        """Add data point to data buffer"""
        if hasattr(data, 'ndim'):  # is it a numpy array?
            data = data.tolist() if data.ndim else [data.item()]
        else:
            data = as_list(data)

        if hasattr(self.xcols, '__iter__'):
            x = next(self.xcols)
            for i, _ in enumerate(self.ycols):
                self.xdata[i].append(x)
        elif hasattr(self.xcols, '__call__'):
            x = self.xcols()
            for i, _ in enumerate(self.ycols):
                self.xdata[i].append(x)
        else:
            for i, xcol in enumerate(as_tuple(self.xcols)):
                self.xdata[i].append(data[xcol])

        for i, ycol in enumerate(self.ycols):
            self.ydata[i].append(data if ycol < 0 else data[ycol])

    def __call__(self, data):
        """Plot data"""
        if not self.__should_plot(data):
            return data

        self.cnt = 0  # reset counter
        self.time = time.time()  # reset timer
        self._add_data(data)

        for i, ax in enumerate(self.axes):
            ax.clear()
            if self.titles:
                ax.set_title(self.titles[i])
            ax.plot(self.xdata[i], self.ydata[i], '-')
            ax.figure.canvas.draw()

        if self.filepath:
            self.figure.savefig(self.filepath, bbox_inches='tight')
        else:
            plt.pause(0.0001)  # Needed to draw
        return data
