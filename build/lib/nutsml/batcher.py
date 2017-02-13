"""
.. module:: batcher
   :synopsis: Collecting samples in mini-batches for GPU-based training.
"""

import numpy as np
import nutsml.imageutil as ni

from nutsflow.base import Nut
from nutsflow.iterfunction import take, PrefetchIterator


def build_number_batch(numbers, dtype):
    """
    Return numpy array with given dtype for given numbers.

    >>> numbers = (1, 2, 3, 1)
    >>> build_number_batch(numbers, 'uint8')
    array([1, 2, 3, 1], dtype=uint8)

    :param iterable number numbers: Numbers to create batch from
    :param numpy data type dtype: Data type of batch, e.g. 'uint8'
    :return: Numpy array for numbers
    :rtype: numpy.array
    """
    return np.array(numbers, dtype=dtype)


def build_one_hot_batch(class_ids, dtype, num_classes):
    """
    Return one hot vectors for class ids.

    >>> class_ids = [0, 1, 2, 1]
    >>> build_one_hot_batch(class_ids, 'uint8', 3)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [0, 1, 0]], dtype=uint8)

    :param iterable class_ids: Class indices in {0, ..., num_classes-1}
    :param numpy data type dtype: Data type of batch, e.g. 'uint8'
    :param num_classes: Number of classes
    :return: One hot vectors for class ids.
    :rtype: numpy.array
    """
    class_ids = np.array(class_ids, dtype=np.uint16)
    return np.eye(num_classes, dtype=dtype)[class_ids]


def build_vector_batch(vectors, dtype):
    """
    Return batch of vectors.

    >>> from datautil import shapestr
    >>> vectors = [np.array([1,2,3]), np.array([2, 3, 4])]
    >>> batch = build_vector_batch(vectors, 'uint8')
    >>> shapestr(batch)
    '2x3'

    >>> batch
    array([[1, 2, 3],
           [2, 3, 4]], dtype=uint8)

    :param iterable row_vectors: Numpy row vectors
    :param numpy data type dtype: Data type of batch, e.g. 'uint8'
    :return: vstack of vectors
    :rtype: numpy.array
    """
    if not len(vectors):
        raise ValueError('No vectors to build batch!')
    return np.vstack(vectors).astype(dtype)


def build_image_batch(images, dtype, channelfirst=True):
    """
    Return batch of images.

    If images have no channel a channel axis is added. For channelfirst=True
    it will be added/moved to front otherwise the channel comes last.
    All images in batch will have a channel axis. Batch is of shape
    (n, c, h, w) or (n, h, w, c) depending on channelfirst, where n is
    the number of images in the batch.

    >>> from datautil import shapestr
    >>> images = [np.zeros((2, 3)), np.ones((2, 3))]
    >>> batch = build_image_batch(images, 'uint8')
    >>> shapestr(batch)
    '2x1x2x3'

    >>> batch
    array([[[[0, 0, 0],
             [0, 0, 0]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[1, 1, 1],
             [1, 1, 1]]]], dtype=uint8)

    :param numpy array images: Images to batch. Must be of shape (w,h,c)
           or (w,h). Gray-scale with channel is fine (w,h,1) and also
           alpha channel is fine (w,h,4).
    :param numpy data type dtype: Data type of batch, e.g. 'uint8'
    :param bool channelfirst: If True, channel is added/moved to front.
    :return: Image batch with shape (n, c, h, w) or (n, h, w, c).
    :rtype: np.array
    """

    def _targetshape(image):
        shape = image.shape
        return (shape[0], shape[1], 1) if image.ndim == 2 else shape

    n = len(images)
    if not n:
        raise ValueError('No images to build batch!')
    h, w, c = _targetshape(images[0])  # shape of first(=all) images
    if c > 4:
        raise ValueError('Channel not at last axis: ' + str((h, w, c)))
    batch = np.empty((n, c, h, w) if channelfirst else (n, h, w, c))
    for i, image in enumerate(images):
        image = ni.add_channel(image, channelfirst)
        if image.shape != batch.shape[1:]:
            raise ValueError('Images vary in shape: ' + str(image.shape))
        batch[i, :, :, :] = image
    return batch.astype(dtype)


class BuildBatch(Nut):
    """
    Build batches for GPU-based neural network training.
    """

    def __init__(self, batchsize, prefetch=1):
        """
        iterable >> BuildBatch(batchsize, prefetch=1)

        Take samples in interable, extract specified columns, convert
        column data to numpy arrays of various types, aggregate converted
        samples into a batch.

        >>> from nutsflow import Collect
        >>> numbers = [4.1, 3.2, 1.1]
        >>> images = [np.zeros((5, 3)), np.ones((5, 3)) , np.ones((5, 3))]
        >>> class_ids = [1, 2, 1]
        >>> samples = zip(numbers, images, class_ids)

        >>> build_batch = (BuildBatch(2)
        ...                .by(0, 'number', float)
        ...                .by(1, 'image', np.uint8, True)
        ...                .by(2, 'one_hot', np.uint8, 3))
        >>> batches = samples >> build_batch >> Collect()

        :param int batchsize: Size of batch = number of rows in batch.
            Number of columns is determined by colspec.

        :param int prefetch: Number of batches to prefetch. This speeds up
           GPU based training.
        """
        self.batchsize = batchsize
        self.colspecs = []
        self.prefetch = prefetch
        self.builder = {'image': build_image_batch,
                        'number': build_number_batch,
                        'vector': build_vector_batch,
                        'one_hot': build_one_hot_batch}

    def by(self, col, name, *args, **kwargs):
        """
        Specify and add batch columns to create

        :param int col: column of the sample to extract and to create a
          batch column from.
        :param string name: Name of the column function to apply to create
            a batch column, e.g. 'image'
            See the following functions for more details:
            'image': nutsflow.batcher.build_image_batch
            'number': nutsflow.batcher.build_number_batch
            'vector': nutsflow.batcher.build_vector_batch
            'one_hot': nutsflow.batcher.build_one_hot_batch
        :param args args: Arguments for column function, e.g. dtype
        :param kwargs kwargs: Keyword arguments for column function
        :return: instance of BuildBatch
        :rtype: BuildBatch
        """
        self.colspecs.append((col, name, args, kwargs))
        return self

    def _batch_generator(self, iterable):
        """Return generator over batches for given iterable of samples"""
        while 1:
            batchsamples = list(take(iterable, self.batchsize))
            if not batchsamples:
                break
            cols = zip(*batchsamples)  # flip rows to cols
            batch = []  # columns of batch
            for colspec in self.colspecs:
                col, func, args, kwargs = colspec
                if not func in self.builder:
                    raise ValueError('Invalid builder: ' + func)
                batch.append(self.builder[func](cols[col], *args, **kwargs))
            yield batch

    def __rrshift__(self, iterable):
        """
        Convert samples in iterable into mini-batches.

        :param iterable iterable: Iterable over samples.
        :return: Mini-batches
        :rtype: list of np.array
        """
        prefetch = self.prefetch
        batch_gen = self._batch_generator(iter(iterable))
        return PrefetchIterator(batch_gen, prefetch) if prefetch else batch_gen
