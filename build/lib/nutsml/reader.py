"""
.. module:: reader
   :synopsis: Reading of sample data and images
"""

import os

import pandas as pd

from glob import glob
from imageutil import load_image
from nutsflow import NutSource, nut_function, nut_source, as_set
from dplython import DplyFrame


@nut_function
def DplyToList(dplyframe):
    """
    Convert DplyDataframe to list.

    See: https://github.com/dodger487/dplython

    >>> ReadPandas().dplyr() >> DplyToList() >> Collect()

    :param DplyDataframe dplyframe: Dataframe.
    :return: List of dataframe rows
    :rtype: list of tuples
    """
    if not isinstance(dplyframe, DplyFrame):
        raise ValueError('Expect Dplyr dataframe!')
    return dplyframe.values.tolist()


@nut_source
def ReadLabelDirs(basedir, filepattern='*'):
    """
    Read file paths from label directories.

    Typically used when classification data is organized in folders,
    where the folder name represents the class label and the files in
    the folder the data samples (images, documents, ...) for that class.

    >>> read = ReadLabelDirs('tests/data/labeldirs', '*.txt')
    >>> samples = read >> Collect()
    >>> for sample in samples:
    ...     print sample
    ...
    ('tests/data/labeldirs/0/test0.txt', '0')
    ('tests/data/labeldirs/1/test1.txt', '1')
    ('tests/data/labeldirs/1/test11.txt', '1')

    :param string basedir: Path to folder that contains label directories.
    :param string filepattern: Pattern for filepaths to read from
           label directories, e.g. '*.jpg', '*.txt'
    :return: iterator over labeled file paths
    :rtype: iterator
    """
    for label in os.listdir(basedir):
        if os.path.isdir(os.path.join(basedir, label)):
            pathname = os.path.join(basedir, label, filepattern)
            for filepath in glob(pathname):
                yield (filepath.replace("\\", "/"), label)


@nut_function
def ReadImage(sample, columns, pathfunc=None, as_grey=False):
    """
    Load images for samples.

    Loads images in jpg, gif, png, tif and bmp format.
    Images are returned as numpy arrays.
    See nutsml.util.load_image for details.

    >>> samples = [('tests/data/img_formats/nut_color.gif', 'class0')]
    >>> img_samples = samples >> ReadImage(0) >> Collect()

    >>> imagepath = 'tests/data/img_formats/*.jpg'
    >>> samples = [(1, 'nut_color'), (2, 'nut_grayscale')]
    >>> img_samples = samples >> ReadImage(1, imagepath) >> Collect()

    >>> pathfunc = lambda sample: 'tests/data/img_formats/{1}.jpg'.format(*sample)
    >>> img_samples = samples >> ReadImage(1, pathfunc) >> Collect()

    :param tuple|list sample: ('nut_color', 1)
    :param int|tuple columns: Indices of columns in sample to be replaced
                              by image (based on image id in that column
    :param string|function|None pathfunc: Filepath with wildcard '*',
      which is replaced by the imageid provided in the sample, e.g.
      'tests/data/img_formats/*.jpg' for sample ('nut_grayscale', 2)
      will become 'tests/data/img_formats/nut_grayscale.jpg'
      or
      Function to compute path to image file from sample, e.g.
      lambda sample: 'tests/data/img_formats/{1}.jpg'.format(*sample)
      or
      None, in this case the image id is take as filepath.
    :param as_grey: If true, load as grayscale image.
    :return: Sample with image ids replaced by image (=ndarray)
    :rtype: tuple
    """

    def load(fileid):
        """Load image for given fileid"""
        if isinstance(pathfunc, str):
            filepath = pathfunc.replace('*', fileid)
        elif hasattr(pathfunc, '__call__'):
            filepath = pathfunc(sample)
        else:
            filepath = fileid
        return load_image(filepath, as_grey=as_grey)

    colset = as_set(columns)
    elems = enumerate(sample)
    return tuple(load(e) if i in colset else e for i, e in elems)


class ReadPandas(NutSource):
    """
    Read data as Pandas table from file system.
    """

    def __init__(self, filepath, rows=None, columns=None, dropnan=True,
                 replacenan=False, **kwargs):
        """
        Create reader for Pandas tables.

        >>> from nutsflow import Collect
        >>> ReadPandas('tests/data/pandas_table.csv') >> Collect()
        [(1.0, 4.0), (3.0, 6.0)]

        Note that samples.table contains the original Pandas dataframe and
        any Pandas operations can be performed on it.
        >>> samples = ReadPandas('tests/data/pandas_table.csv')
        >>> samples.table.head()
           col1  col2
        0     1   4.0
        1     2   NaN
        2     3   6.0

        >>> samples = ReadPandas('tests/data/pandas_table.csv')
        >>> samples.table.columns.values.tolist()
        ['col1', 'col2']

        :param str filepath: Path to a table in CSV, TSV, XLSX or
          Pandas pickle format. Depending of file extension (e.g. .csv)
          the table format is picked.
          Note tables must have a header with the column names or
          use kwarg header=None
        :param str rows: Rows to filter. Any Pandas filter expression. If
          rows = None all rows of the table are returned.
        :param list columns: List of names for the table columns to return.
          For columns = None all columns are returned.
        :param bool dropnan: If True all rows that contain NaN are dropped.
        :param bool replacenan: If True all NaNs are replaced by None.
        :param kwargs kwargs: Key word arguments passed on the the Pandas
          methods for data reading, e.g, header=None.
          See  pandas/pandas/io/parsers.py for detais

        """
        self.filepath = filepath
        self.rows = rows
        self.columns = columns
        self.dropnan = dropnan
        self.replacenan = replacenan
        self.kwargs = kwargs
        self.table = self._load_table(filepath)

    def print_head(self, n=5):  # pragma: no cover
        """
        Print head of table. Just a short cut for print self.table.head()

        >>> samples = ReadPandas('tests/data/pandas_table.csv')
        >>> samples.print_head()
           col1  col2
        0     1   4.0
        1     2   NaN
        2     3   6.0

        :param int n: Number of rows to print.
        """
        print self.table.head(n)

    def print_info(self):  # pragma: no cover
        """
        Print info about table. Just a short cut for self.table.info()

        >>> samples = ReadPandas('tests/data/pandas_table.csv')
        >>> samples.print_info()
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 3 entries, 0 to 2
        Data columns (total 2 columns):
        col1    3 non-null int64
        col2    2 non-null float64
        dtypes: float64(1), int64(1)
        memory usage: 120.0 bytes
        """
        self.table.info()

    @staticmethod
    def isnull(value):
        """
        Return true if values is NaN or None.

        >> import numpy as np
        >> ReadPandas.isnull(np.NaN)
        True

        >> ReadPandas.isnull(None)
        True

        >> ReadPandas.isnull(0)
        False

        :param value: Value to test
        :return: Return true for NaN or None values.
        :rtype: bool
        """
        return pd.isnull(value)

    @staticmethod
    def _replacenan(row):
        """
        Replace NaN values in row by None

        :param iterable row: Any iterable.
        :return: Row with None instead of NaN
        :rtype: tuple
        """
        return tuple(None if pd.isnull(v) else v for v in row)

    def dply(self):
        """
        Return dplyr frame for the read table.

        dplyr is an R inspired wrapper to process Pandas tables in a
        flow-like manner. See https://github.com/dodger487/dplython and
        https://cran.rstudio.com/web/packages/dplyr/vignettes/introduction.html
        for more details about dplyr.

        dplyr and Nuts use the same syntax (>>) for chaining functions and
        integrate nicely with each other.

        :return: dplyr dataframe instead of Pandas dataframe.
        :rtype: DplyFrame
        """
        return DplyFrame(self.table)

    def _load_table(self, filepath):
        """
        Load table from file system.

        :param str filepath: Path to table in CSV, TSV, XLSX or
                   Pandas pickle format.
        :return: Pandas table
        :rtype: pandas.core.frame.DataFrame
        """
        _, ext = os.path.splitext(filepath.lower())
        if ext == '.tsv':
            return pd.read_table(filepath, **self.kwargs)
        if ext == '.csv':
            return pd.read_csv(filepath, **self.kwargs)
        if ext == '.xlsx':
            return pd.read_excel(filepath, **self.kwargs)
        return pd.read_pickle(filepath, **self.kwargs)

    def __iter__(self):
        """
        Return iterator over rows in table.

        :return: Iterator over rows.
        :rtype: iterator
        """
        rows = self.table.query(self.rows) if self.rows else self.table
        series = rows[self.columns] if self.columns else rows
        if self.replacenan:
            return (ReadPandas._replacenan(row) for row in series.values)
        if self.dropnan:
            return (tuple(row) for row in series.dropna().values)
        return (tuple(row) for row in series.values)
