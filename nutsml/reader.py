"""
.. module:: reader
   :synopsis: Reading of sample data and images
"""
from __future__ import absolute_import

import os

import pandas as pd
import numpy as np

from glob import glob
from collections import namedtuple
from fnmatch import fnmatch
from nutsml.imageutil import load_image
from nutsml.fileutil import reader_filepath
from nutsflow import NutSource, nut_function, nut_source
from nutsflow.common import as_set


@nut_source
def ReadLabelDirs(basedir, filepattern='*', exclude='_*'):
    """
    Read file paths from label directories.

    Typically used when classification data is organized in folders,
    where the folder name represents the class label and the files in
    the folder the data samples (images, documents, ...) for that class.

    >>> from __future__ import print_function
    >>> from nutsflow import Sort

    >>> read = ReadLabelDirs('tests/data/labeldirs', '*.txt')
    >>> samples = read >> Sort()
    >>> for sample in samples:
    ...     print(sample)
    ...
    ('tests/data/labeldirs/0/test0.txt', '0')
    ('tests/data/labeldirs/1/test1.txt', '1')
    ('tests/data/labeldirs/1/test11.txt', '1')

    :param string basedir: Path to folder that contains label directories.
    :param string filepattern: Pattern for filepaths to read from
           label directories, e.g. '*.jpg', '*.txt'
    :param string exclude: Pattern for label directories to exclude.
           Default is '_*' which excludes all label folders prefixed with '_'.
    :return: iterator over labeled file paths
    :rtype: iterator
    """
    for label in os.listdir(basedir):
        if os.path.isdir(os.path.join(basedir, label)):
            if fnmatch(label, exclude):
                continue
            pathname = os.path.join(basedir, label, filepattern)
            for filepath in glob(pathname):
                yield filepath.replace('\\', '/'), label


@nut_function
def ReadNumpy(sample, columns, pathfunc=None, allow_pickle=False):
    """
    Load numpy arrays from filesystem.

    Note that the loaded numpy array replace the file name|path in the
    sample.

    >>> from nutsflow import Consume, Collect, PrintType

    >>> samples = ['tests/data/img_arrays/nut_color.jpg.npy']
    >>> samples >> ReadNumpy(None) >> PrintType() >> Consume()
    (<ndarray> 213x320x3:uint8)

    >>> samples = [('tests/data/img_arrays/nut_color.jpg.npy', 'class0')]
    >>> samples >> ReadNumpy(0) >> PrintType() >> Consume()
    (<ndarray> 213x320x3:uint8, <str> class0)

    >>> filepath = 'tests/data/img_arrays/*.jpg.npy'
    >>> samples = [(1, 'nut_color'), (2, 'nut_grayscale')]
    >>> samples >> ReadNumpy(1, filepath) >> PrintType() >> Consume()
    (<int> 1, <ndarray> 213x320x3:uint8)
    (<int> 2, <ndarray> 213x320:uint8)

    >>> pathfunc = lambda s: 'tests/data/img_arrays/{1}.jpg.npy'.format(*s)
    >>> samples >> ReadNumpy(1, pathfunc) >> PrintType() >> Consume()
    (<int> 1, <ndarray> 213x320x3:uint8)
    (<int> 2, <ndarray> 213x320:uint8)

    :param tuple|list sample: ('nut_data', 1)
    :param None|int|tuple columns: Indices of columns in sample to be replaced
                              by numpy array (based on fileid in that column)
                              If None then a flat samples is assumed and
                              a tuple with the numpy array is returned.
    :param string|function|None pathfunc: Filepath with wildcard '*',
      which is replaced by the file id/name provided in the sample, e.g.
      'tests/data/img_arrays/*.jpg.npy' for sample ('nut_grayscale', 2)
      will become 'tests/data/img_arrays/nut_grayscale.jpg.npy'
      or
      Function to compute path to numnpy file from sample, e.g.
      lambda sample: 'tests/data/img_arrays/{1}.jpg.npy'.format(*sample)
      or
      None, in this case the file id/name is taken as the filepath.
    :param bool allow_pickle : Allow loading pickled object arrays in npy files.
    :return: Sample with file ids/names replaced by numpy arrays.
    :rtype: tuple
    """

    def load(filename):
        """Load numpy array for given fileid"""
        filepath = reader_filepath(sample, filename, pathfunc)
        return np.load(filepath, allow_pickle=allow_pickle)

    if columns is None:
        return (load(sample),)  # numpy array as tuple with one element

    colset = as_set(columns)
    elems = enumerate(sample)
    return tuple(load(e) if i in colset else e for i, e in elems)


@nut_function
def ReadImage(sample, columns, pathfunc=None, as_grey=False, dtype='uint8'):
    """
    Load images from filesystem for samples.

    Loads images in jpg, gif, png, tif and bmp format.
    Images are returned as numpy arrays of shape (h, w, c) or (h, w) for
    color images or gray scale images respectively.
    See nutsml.imageutil.load_image for details.

    Note that the loaded images replace the image file name|path in the
    sample. If the images file paths are directly proved (not as a tuple
    sample) still tuples with the loaded image are returned.
    
    >>> from nutsflow import Consume, Collect
    >>> from nutsml import PrintColType

    >>> images = ['tests/data/img_formats/nut_color.gif']
    >>> images >> ReadImage(None) >> PrintColType() >> Consume()
    item 0: <tuple>
      0: <ndarray> shape:213x320x3 dtype:uint8 range:0..255

    >>> samples = [('tests/data/img_formats/nut_color.gif', 'class0')]
    >>> img_samples = samples >> ReadImage(0) >> Collect()

    >>> imagepath = 'tests/data/img_formats/*.gif'
    >>> samples = [(1, 'nut_color'), (2, 'nut_grayscale')]
    >>> samples >> ReadImage(1, imagepath) >> PrintColType() >> Consume()
    item 0: <tuple>
      0: <int> 1
      1: <ndarray> shape:213x320x3 dtype:uint8 range:0..255
    item 1: <tuple>
      0: <int> 2
      1: <ndarray> shape:213x320 dtype:uint8 range:20..235

    >>> pathfunc = lambda s: 'tests/data/img_formats/{1}.jpg'.format(*s)
    >>> img_samples = samples >> ReadImage(1, pathfunc) >> Collect()

    :param tuple|list sample: ('nut_color', 1)
    :param None|int|tuple columns: Indices of columns in sample to be replaced
                              by image (based on image id in that column)
                              If None then a flat samples is assumed and
                              a tuple with the image is returned.
    :param string|function|None pathfunc: Filepath with wildcard '*',
      which is replaced by the imageid provided in the sample, e.g.
      'tests/data/img_formats/*.jpg' for sample ('nut_grayscale', 2)
      will become 'tests/data/img_formats/nut_grayscale.jpg'
      or
      Function to compute path to image file from sample, e.g.
      lambda sample: 'tests/data/img_formats/{1}.jpg'.format(*sample)
      or
      None, in this case the image id is taken as the filepath.
    :param bool as_grey: If true, load as grayscale image.
    :param dtype dtype: Numpy data type of the image.
    :return: Sample with image ids replaced by image (=ndarray)
            of shape (h, w, c) or (h, w)
    :rtype: tuple
    """

    def load(filename):
        """Load image for given fileid"""
        filepath = reader_filepath(sample, filename, pathfunc)
        return load_image(filepath, as_grey=as_grey, dtype=dtype)

    if columns is None:
        return (load(sample),)  # image as tuple with one element

    colset = as_set(columns)
    elems = enumerate(sample)
    return tuple(load(e) if i in colset else e for i, e in elems)


class ReadPandas(NutSource):
    """
    Read data as Pandas table from file system.
    """

    def __init__(self, filepath, rows=None, colnames=None, dropnan=True,
                 replacenan=False, rowname='Row', **kwargs):
        """
        Create reader for Pandas tables.

        The reader returns the table contents as an interator over named tuples,
        where the column names are derived from the table columns. The order
        and selection of columns can be changed.

        >>> from nutsflow import Collect, Consume, Print
        >>> filepath = 'tests/data/pandas_table.csv'

        >>> ReadPandas(filepath) >> Print() >> Consume()
        Row(col1=1.0, col2=4.0)
        Row(col1=3.0, col2=6.0)

        >>> (ReadPandas(filepath, dropnan=False, rowname='Sample') >>
        ... Print() >> Consume())
        Sample(col1=1.0, col2=4.0)
        Sample(col1=2.0, col2=nan)
        Sample(col1=3.0, col2=6.0)

        >>> ReadPandas(filepath, replacenan=None) >> Print() >> Consume()
        Row(col1=1.0, col2=4.0)
        Row(col1=2.0, col2=None)
        Row(col1=3.0, col2=6.0)

        >>> colnames=['col2', 'col1']   # swap order
        >>> ReadPandas(filepath, colnames=colnames) >> Print() >> Consume()
        Row(col2=4.0, col1=1.0)
        Row(col2=6.0, col1=3.0)

        >>> ReadPandas(filepath, rows='col1 > 1', replacenan=0) >> Collect()
        [Row(col1=2.0, col2=0), Row(col1=3.0, col2=6.0)]

        :param str filepath: Path to a table in CSV, TSV, XLSX or
          Pandas pickle format. Depending on file extension (e.g. .csv)
          the table format is picked.
          Note tables must have a header with the column names.
        :param str rows: Rows to filter. Any Pandas filter expression. If
          rows = None all rows of the table are returned.
        :param list columns: List of names for the table columns to return.
          For columns = None all columns are returned.
        :param bool dropnan: If True all rows that contain NaN are dropped.
        :param object replacenan: If not False all NaNs are replaced by
             the value of replacenan
        :param str rowname: Name of named tuple return as rows.
        :param kwargs kwargs: Key word arguments passed on the the Pandas
          methods for data reading, e.g, header=None.
          See pandas/pandas/io/parsers.py for detais

        """
        self.filepath = filepath
        self.rows = rows
        self.colnames = colnames
        self.dropnan = dropnan
        self.replacenan = replacenan
        self.rowname = rowname
        self.kwargs = kwargs
        self.dataframe = self._load_table(filepath)

    @staticmethod
    def isnull(value):
        """
        Return true if values is NaN or None.

        >>> import numpy as np
        >>> ReadPandas.isnull(np.NaN)
        True

        >>> ReadPandas.isnull(None)
        True

        >>> ReadPandas.isnull(0)
        False

        :param value: Value to test
        :return: Return true for NaN or None values.
        :rtype: bool
        """
        return pd.isnull(value)

    def _replacenan(self, row):
        """
        Replace NaN values in row by None

        :param iterable row: Any iterable.
        :return: Row with None instead of NaN
        :rtype: tuple
        """
        value = self.replacenan
        return tuple(value if pd.isnull(v) else v for v in row)

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
            return pd.read_csv(filepath, sep='\t', **self.kwargs)
        if ext == '.csv':
            return pd.read_csv(filepath, **self.kwargs)
        if ext == '.xlsx':
            return pd.read_excel(filepath, engine='openpyxl', **self.kwargs)
        return pd.read_pickle(filepath, **self.kwargs)

    def __iter__(self):
        """
        Return iterator over rows in table.

        :return: Iterator over rows.
        :rtype: iterator
        """
        df = self.dataframe
        rows = df.query(self.rows) if self.rows else df
        series = rows[self.colnames] if self.colnames else rows
        Row = namedtuple(self.rowname, series.columns.to_list())

        if not self.replacenan is False:
            values = (self._replacenan(row) for row in series.values)
        elif self.dropnan:
            values = series.dropna().values
        else:
            values = series.values
        return (Row(*v) for v in values)
