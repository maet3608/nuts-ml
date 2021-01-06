"""
.. module:: logger
   :synopsis: Data logging
"""

import os
import numpy as np
from warnings import warn
from nutsflow import NutFunction
from nutsflow.common import as_tuple


class LogToFile(NutFunction):
    """
    Log columns of data to file.
    """

    def __init__(self, filepath, cols=None, colnames=None, reset=True,
                 delimiter=','):
        """
        Construct logger.

        >>> from __future__ import print_function
        >>> from nutsflow import Consume
        >>> filepath = 'tests/data/temp_logfile.csv'
        >>> data = [[1, 2], [3, 4]]

        >>> with LogToFile(filepath) as logtofile:
        ...     data >> logtofile >> Consume()
        >>> print(open(filepath).read())
        1,2
        3,4
        <BLANKLINE>

        >>> logtofile = LogToFile(filepath, cols=(1, 0), colnames=['a', 'b'])
        >>> data >> logtofile >> Consume()
        >>> print(open(filepath).read())
        a,b
        2,1
        4,3
        <BLANKLINE>
        >>> logtofile.close()
        >>> logtofile.delete()

        :param string filepath: Path to file to write log to.
        :param int|tuple|None cols: Indices of columns of input data to write.
                None: write all columns
                int: only write the single given column
                tuple: list of column indices
        :param tuple|None colnames: Column names to write in first line.
                If None no colnames are written.
        :param bool reset: If True the writing to the log file is reset
               if the logger is recreated. Otherwise log data is appended
               to the log file.
        :param str delimiter: Delimiter for columns in log file.
        """
        self.cols = cols
        self.reset = reset
        self.delim = delimiter
        self.filepath = filepath
        self.f = open(filepath, 'w' if self.reset else 'a')
        if colnames:
            self._writerow(colnames)

    def _writerow(self, row):
        """Write row as string to log file and flush"""
        self.f.write(self.delim.join(map(str, row)))
        self.f.write('\n')
        self.f.flush()

    def __call__(self, x):
        """
        Log x

        :param any x: Any type of data.
                      Special support for numpy arrays.
        :return: Return input unchanged
        :rtype: Same as input
        """
        if isinstance(x, np.ndarray):
            row = x.tolist() if x.ndim else [x.item()]
        else:
            row = x
        if not self.cols is None:
            row = [row[i] for i in as_tuple(self.cols)]
        self._writerow(row)
        return x

    def delete(self):
        """Delete log file"""
        self.close()
        os.remove(self.filepath)

    def close(self):
        """Implementation of context manager API"""
        self.f.close()

    def __enter__(self):
        """Implementation of context manager API"""

        return self

    def __exit__(self, *args):
        """Implementation of context manager API"""
        self.close()


class LogCols(LogToFile):
    def __init__(self, filepath, cols=None, colnames=None, reset=True,
                 delimiter=','):
        LogToFile.__init__(self, filepath, cols, colnames, reset, delimiter)
        warn('LogCols is deprecated. Use LogToFile!', DeprecationWarning)
