"""
.. module:: batcher
   :synopsis: Collecting samples in mini-batches for GPU-based training.
"""
import warnings
import numpy as np
import nutsml.imageutil as ni

from nutsflow import nut_function
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

    >>> from nutsflow.common import shapestr
    >>> vectors = [np.array([1,2,3]), np.array([2, 3, 4])]
    >>> batch = build_vector_batch(vectors, 'uint8')
    >>> shapestr(batch)
    '2x3'

    >>> batch
    array([[1, 2, 3],
           [2, 3, 4]], dtype=uint8)

    :param iterable vectors: Numpy row vectors
    :param numpy data type dtype: Data type of batch, e.g. 'uint8'
    :return: vstack of vectors
    :rtype: numpy.array
    """
    if not len(vectors):
        raise ValueError('No vectors to build batch!')
    return np.vstack(vectors).astype(dtype)


def build_tensor_batch(tensors, dtype, axes=None, expand=None):
    """
    Return batch of tensors.

    >>> from nutsflow.common import shapestr
    >>> tensors = [np.zeros((2, 3)), np.ones((2, 3))]
    >>> batch = build_tensor_batch(tensors, 'uint8')
    >>> shapestr(batch)
    '2x2x3'

    >>> print(batch)
    [[[0 0 0]
      [0 0 0]]
    <BLANKLINE>
     [[1 1 1]
      [1 1 1]]]

    >>> batch = build_tensor_batch(tensors, 'uint8', expand=0)
    >>> shapestr(batch)
    '2x1x2x3'

    >>> print(batch)
    [[[[0 0 0]
       [0 0 0]]]
    <BLANKLINE>
     [[[1 1 1]
       [1 1 1]]]]

    >>> batch = build_tensor_batch(tensors, 'uint8', axes=(1, 0))
    >>> shapestr(batch)
    '2x3x2'

    >>> print(batch)
    [[[0 0]
      [0 0]
      [0 0]]
    <BLANKLINE>
     [[1 1]
      [1 1]
      [1 1]]]

    :param iterable tensors: Numpy tensors
    :param numpy data type dtype: Data type of batch, e.g. 'uint8'
    :param tuple|None axes: axes order, e.g. to move a channel axis to the
      last position. (see numpy transpose for details)
    :param int|None expand: Add empty dimension at expand dimension.
        (see numpy expand_dims for details).
    :return: stack of tensors, with batch axis first.
    :rtype: numpy.array
    """
    if not len(tensors):
        raise ValueError('No tensors to build batch!')
    if axes is not None:
        tensors = [np.transpose(t, axes) for t in tensors]
    if expand is not None:
        tensors = [np.expand_dims(t, expand) for t in tensors]
    return np.stack(tensors).astype(dtype)


def build_image_batch(images, dtype, channelfirst=False):
    """
    Return batch of images.

    If images have no channel a channel axis is added. For channelfirst=True
    it will be added/moved to front otherwise the channel comes last.
    All images in batch will have a channel axis. Batch is of shape
    (n, c, h, w) or (n, h, w, c) depending on channelfirst, where n is
    the number of images in the batch.

    >>> from nutsflow.common import shapestr
    >>> images = [np.zeros((2, 3)), np.ones((2, 3))]
    >>> batch = build_image_batch(images, 'uint8', True)
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
    if c > w or c > h:
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

        Take samples in iterable, extract specified columns, convert
        column data to numpy arrays of various types, aggregate converted
        samples into a batch.

        The format of a batch is a list of lists: [[inputs], [outputs]]
        where inputs and outputs are Numpy arrays.

        The following example uses PrintType() to print the shape of
        the batches constructed. This is useful for development and debugging
        but should be removed in production.

        >>> from nutsflow import Collect, PrintType

        >>> numbers = [4.1, 3.2, 1.1]
        >>> images = [np.zeros((5, 3)), np.ones((5, 3)) , np.ones((5, 3))]
        >>> class_ids = [1, 2, 1]
        >>> samples = list(zip(numbers, images, class_ids))

        >>> build_batch = (BuildBatch(batchsize=2)
        ...                .input(0, 'number', 'float32')
        ...                .input(1, 'image', np.uint8, True)
        ...                .output(2, 'one_hot', np.uint8, 3))
        >>> batches = samples >> build_batch >> PrintType() >> Collect()
        [[<ndarray> 2:float32, <ndarray> 2x1x5x3:uint8], [<ndarray> 2x3:uint8]]
        [[<ndarray> 1:float32, <ndarray> 1x1x5x3:uint8], [<ndarray> 1x3:uint8]]

        In the example above, we have multiple inputs and a single output,
        and the batch is of format [[number, image], [one_hot]], where each
        data element a Numpy array with the shown shape and dtype.

        Sample columns can be ignored or reused. Assuming an autoencoder, one
        might whish to reuse the sample image as input and output:

        >>> build_batch = (BuildBatch(2)
        ...                .input(1, 'image', np.uint8, True)
        ...                .output(1, 'image', np.uint8, True))
        >>> batches = samples >> build_batch >> PrintType() >> Collect()
        [[<ndarray> 2x1x5x3:uint8], [<ndarray> 2x1x5x3:uint8]]
        [[<ndarray> 1x1x5x3:uint8], [<ndarray> 1x1x5x3:uint8]]

        In the prediction phase no target outputs are needed. If the batch
        contains only inputs, the batch format is just [inputs].

        >>> build_pred_batch = (BuildBatch(2)
        ...                     .input(1, 'image', 'uint8', True))
        >>> batches = samples >> build_pred_batch >> PrintType() >> Collect()
        [<ndarray> 2x1x5x3:uint8]
        [<ndarray> 1x1x5x3:uint8]


        :param int batchsize: Size of batch = number of rows in batch.
        :param int prefetch: Number of batches to prefetch. This speeds up
           GPU based training, since one batch is built on CPU while the
           another is processed on the GPU.
           Note: if verbose=True, prefetch is set to 0 to simplify debugging.
        :param bool verbose: Print batch shape when True.
           (and sets prefetch=0)
        """
        self.batchsize = batchsize
        self.prefetch = prefetch
        self.colspecs = []
        self.builder = {'image': build_image_batch,
                        'number': build_number_batch,
                        'vector': build_vector_batch,
                        'tensor': build_tensor_batch,
                        'one_hot': build_one_hot_batch}

    def input(self, col, name, *args, **kwargs):
        """
        Specify and add input columns for batch to create

        :param int col: column of the sample to extract and to create a
          batch input column from.
        :param string name: Name of the column function to apply to create
            a batch column, e.g. 'image'
            See the following functions for more details:
            'image': nutsflow.batcher.build_image_batch
            'number': nutsflow.batcher.build_number_batch
            'vector': nutsflow.batcher.build_vector_batch
            'tensor': nutsflow.batcher.build_tensor_batch
            'one_hot': nutsflow.batcher.build_one_hot_batch
        :param args args: Arguments for column function, e.g. dtype
        :param kwargs kwargs: Keyword arguments for column function
        :return: instance of BuildBatch
        :rtype: BuildBatch
        """
        self.colspecs.append((col, name, True, args, kwargs))
        return self

    def output(self, col, name, *args, **kwargs):
        """
        Specify and add output columns for batch to create

        :param int col: column of the sample to extract and to create a
          batch output column from.
        :param string name: Name of the column function to apply to create
            a batch column, e.g. 'image'
            See the following functions for more details:
            'image': nutsflow.batcher.build_image_batch
            'number': nutsflow.batcher.build_number_batch
            'vector': nutsflow.batcher.build_vector_batch
            'tensor': nutsflow.batcher.build_tensor_batch
            'one_hot': nutsflow.batcher.build_one_hot_batch
        :param args args: Arguments for column function, e.g. dtype
        :param kwargs kwargs: Keyword arguments for column function
        :return: instance of BuildBatch
        :rtype: BuildBatch
        """
        self.colspecs.append((col, name, False, args, kwargs))
        return self

    def _batch_generator(self, iterable):
        """Return generator over batches for given iterable of samples"""
        while 1:
            batchsamples = list(take(iterable, self.batchsize))
            if not batchsamples:
                break
            cols = list(zip(*batchsamples))  # flip rows to cols
            batch = [[], []]  # in, out columns of batch
            for colspec in self.colspecs:
                col, func, isinput, args, kwargs = colspec
                if not func in self.builder:
                    raise ValueError('Invalid builder: ' + func)
                coldata = self.builder[func](cols[col], *args, **kwargs)
                batch[0 if isinput else 1].append(coldata)
            if not batch[1]:  # no output (prediction phase)
                batch = batch[0]  # flatten and take only inputs
            yield batch

    def __rrshift__(self, iterable):
        """
        Convert samples in iterable into mini-batches.

        Structure of output depends on fmt function used. If None
        output is a list of np.arrays

        :param iterable iterable: Iterable over samples.
        :return: Mini-batches
        :rtype: list of np.array if fmt=None
        """
        batch_gen = self._batch_generator(iter(iterable))
        if self.prefetch:
            batch_gen = PrefetchIterator(batch_gen, self.prefetch)
        return batch_gen


@nut_function
def Mixup(batch, alpha):
    """
    Mixup produces random interpolations between data and labels.

    Usage:
    ... >> BuildBatch() >> Mixup(0.1) >> network.train() >> ...

    Implementation based on the following paper:
    mixup: Beyond Empirical Risk Minimization
    https://arxiv.org/abs/1710.09412

    :param list batch: Batch consisting of list of input data and list of
           output data, where data must be numeric, e.g. images and
           one-hot-encoded class labels that can be interpolated between.
    :param float alpha: Control parameter for beta distribution the
           interpolation factors are sampled from. Range: [0,...,1]
           For alpha <= 0 no mixup is performed.
    :return:
    """
    if alpha <= 0:
        return batch

    ri = np.arange(len(batch[0][0]))
    np.random.shuffle(ri)
    lam = np.random.beta(alpha, alpha)
    mixup = lambda data: lam * data + (1 - lam) * data[ri]

    inputs = [mixup(i) for i in batch[0]]
    outputs = [mixup(o) for o in batch[1]]
    return [inputs, outputs]
