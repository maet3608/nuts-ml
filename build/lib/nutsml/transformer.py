"""
.. module:: transformer
   :synopsis: Data and image transformations
"""

import numpy as np
import os.path as path
import random as rnd
import nutsml.datautil as ut
import nutsml.imageutil as ni

from nutsflow import Nut, NutFunction, nut_processor, as_tuple, as_set


# map transformation to specified sample columns
def map_transform(sample, imagecols, spec):
    """
    Map transformation function on columns of sample.

    :param tuple sample: Sample with images
    :param int|tuple imagecols: Indices of sample columns the transformation
       should be applied to. Can be a single index or a tuple of indices.
    :param tuple spec: Transformation specification. Either a tuple with
       the name of the transformation function or a tuple with the
       name, arguments and keyword arguments of the transformation function.
    :return: Sample with transformations applied. Columns not specified
      remain unchained.
    :rtype: tuple
    """
    colset = as_tuple(imagecols)
    name, a, kw = spec if isinstance(spec, tuple) else (spec, [], {})
    f = TransformImage.transformations[name]
    enum_sample = enumerate(sample)
    return tuple(f(e, *a, **kw) if i in colset else e for i, e in enum_sample)


class TransformImage(NutFunction):
    """
    Transformation of images in samples.
    """

    transformations = {
        'identical': ni.identical,
        'rerange': ni.rerange,
        'crop': ni.crop,
        'crop_center': ni.crop_center,
        'normalize_histo': ni.normalize_histo,
        'resize': ni.resize,
        'rotate': ni.rotate,
        'contrast': ni.change_contrast,
        'sharpness': ni.change_sharpness,
        'brightness': ni.change_brightness,
        'color': ni.change_color,
        'fliplr': ni.fliplr,
        'flipud': ni.flipud,
        'shear': ni.shear,
    }

    def __init__(self, imagecols):
        """
        samples >> TransformImage(imagecols)

        Images are expect to be numpy array of the shape (h, w, c) or (h, w)
        with a range of [0,255] and a dtype of uint8. Transformation should
        result in images with the same properties.

        Example
        >>> transform = TransformImage(0).by('resize', 10, 20)

        :param int|tuple imagecols: Indices of sample columns the transformation
            should be applied to. Can be a single index or a tuple of indices.
        :param tuple transspec: Transformation specification. Either a
            tuple with the name of the transformation function or a tuple
            with the name, arguments and keyword arguments of the
            transformation function.
            The list of argument values and dictionaries provided in the
            transspec are simply passed on to the transformation function.
            See the relevant functions for details.
        """
        self.transspec = []
        self.imagecols = as_tuple(imagecols)

    def by(self, name, *args, **kwargs):
        """
        Specify and add tranformation to be performed.

        >>> transform = TransformImage(0).by('resize', 10, 20).by('fliplr')

        :param string name: Name of the transformation to apply, e.g. 'resize'
        :param args args: Arguments for the transformation, e.g. width and
           height for resize.
        :param kwargs kwargs: Keyword arguments passed on to the transformation
        :return: instance of TransformImage with added transformation
        :rtype: TransformImage
        """
        self.transspec.append((name, args, kwargs))
        return self

    @classmethod
    def register(cls, name, transformation):
        """
        Register new transformation function.

        >>> brighter = lambda image, c: image * c
        >>> TransformImage.register('brighter', brighter)
        >>> transform = TransformImage(0).by('brighter', 1.5)

        :param string name: Name of transformation
        :param function transformation: Transformation function.
        """
        cls.transformations[name] = transformation

    def __call__(self, sample):
        """
        Apply transformation to sample.

        :param tuple sample: Sample
        :return: Transformed sample
        :rtype: tuple
        """
        for spec in self.transspec:
            sample = map_transform(sample, self.imagecols, spec)
        return sample


class AugmentImage(Nut):
    """
    Random augmentation of images in samples
    """

    def __init__(self, imagecols, rand=rnd.Random()):
        """
        samples >> AugmentImage(imagecols, rand=rnd.Random())

        Randomly augment images, e.g. changing contrast. See TransformImage for
        a full list of available augmentations. Every transformation can be
        used as an augmentation. Note that the same (random) augmentation
        is applied to all images specified in imagecols. This ensures that
        an image and its mask are randomly rotated by the same angle,
        for instance.

        >>> augment_img = (AugmentImage(0)
        ...     .by('identical', 1.0)
        ...     .by('brightness', 0.5, [0.7, 1.3])
        ...     .by('contrast', 0.5, [0.7, 1.3])
        ...     .by('fliplr', 0.5)
        ...     .by('flipud', 0.5)
        ...     .by('rotate', 0.5, [0, 360]))

        Note that each augmentation is applied independently. This is in
        contrast to transformations which are applied in sequence and
        result in one image.
        Augmentation on the other hand are randomly applied and can result
        in many images. However, augmenters can be chained to achieve
        combinations of augmentation, e.g. contrast or brightness combined with
        rotation or shearing:

        >>> augment1 = (AugmentImage(0)
        ...     .by('brightness', 0.5, [0.7, 1.3])
        ...     .by('contrast', 0.5, [0.7, 1.3]))

        >>> augment2 = (AugmentImage(0)
        ...     .by('shear', 0.5, [0, 0.2])
        ...     .by('rotate', 0.5, [0, 360]))

        samples >> augment1 >> augment2 >> Consume()


        :param int|tuple imagecols: Indices of sample columns that contain
            images.
        :param Random rand: Random number generator to be used.
        """
        self.imagecols = imagecols
        self.rand = rand
        self.augmentations = []

    def by(self, name, prob, *ranges, **kwargs):
        """
        Specify and add augmentation to be performed.

        >>> augment_img = AugmentImage(0).by('rotate', 0.5, [0, 360])

        :param string name: Name of the augmentation/transformation, e.g.
          'rotate'
        :param float prob: Probability [0,1] that the augmentation is applied.
        :param list of lists ranges: Lists with ranges for each argument of
          the augmentation, e.g. [0, 360] degrees, where parameters are
            randomly sampled from.
        :param kwargs kwargs: Keyword arguments passed on the the augmentation.
        :return: instance of AugmentImage
        :rtype: AugmentImage
        """
        self.augmentations.append((name, prob, ranges, kwargs))
        return self

    def __rrshift__(self, iterable):
        """
        Apply augmentation to samples in iterable.

        :param iterable iterable: Samples
        :return: iterable with augmented samples
        :rtype: iterable
        """
        imagecols = as_tuple(self.imagecols)
        rand = self.rand
        for sample in iterable:
            for name, p, ranges, kwargs in self.augmentations:
                if rand.uniform(0, 1) < p:
                    args = [rand.uniform(r[0], r[1]) for r in ranges]
                    transformation = name, args, kwargs
                    yield map_transform(sample, imagecols, transformation)


@nut_processor
def RegularImagePatches(iterable, imagecols, pshape, stride):
    """
    samples >> RegularImagePatches(imagecols, shape, stride)

    Extract patches in a regular grid from images.

    >>> import numpy as np
    >>> img = np.reshape(np.arange(12), (3, 4))
    >>> samples = [(img, 0)]
    >>> get_patches = RegularImagePatches(0, (2, 2), 2)
    >>> for p in samples >> get_patches:
    ...     print p
    (array([[0, 1],
           [4, 5]]), 0)
    (array([[2, 3],
           [6, 7]]), 0)

    :param iterable iterable: Samples with images
    :param int|tuple imagecols: Indices of sample columns that contain
      images, where patches are extracted from.
      Images must be numpy arrays of shape h,w,c or h,w
    :param tuple shape: Shape of patch (h,w)
    :param int stride: Step size of grid patches are extracted from
    :return: Iterator over samples where images are replaced by patches.
    :rtype: iterator
    """
    colset = as_set(imagecols)
    for sample in iterable:
        patches = ut.col_map(sample, imagecols, ni.patch_iter, pshape, stride)
        while True:
            enum_iter = enumerate(patches)
            patched = tuple(next(p) if i in colset else p for i, p in enum_iter)
            if len(patched) == len(sample):
                yield patched
            else:
                break  # one or all patch iterators are depleted


@nut_processor
def RandomImagePatches(iterable, imagecols, pshape, npatches):
    """
    samples >> RandomImagePatches(imagecols, shape, npatches)

    Extract patches at random locations from images.

    >>> import numpy as np
    >>> np.random.seed(0)    # just to ensure stable doctest
    >>> img = np.reshape(np.arange(30), (5, 6))
    >>> samples = [(img, 0)]
    >>> get_patches = RandomImagePatches(0, (2, 3), 3)
    >>> for (p, l) in patches:
    ...     print p, l
    [[ 7  8  9]
     [13 14 15]] 0
    [[ 8  9 10]
     [14 15 16]] 0
    [[ 8  9 10]
     [14 15 16]] 0

    :param iterable iterable: Samples with images
    :param int|tuple imagecols: Indices of sample columns that contain
      images, where patches are extracted from.
      Images must be numpy arrays of shape h,w,c or h,w
    :param tuple shape: Shape of patch (h,w)
    :param int npatches: Number of patches to extract (per image)
    :return: Iterator over samples where images are replaced by patches.
    :rtype: iterator
    """
    imagecols = as_tuple(imagecols)
    for sample in iterable:
        image = sample[imagecols[0]]
        hi, wi = image.shape[:2]
        hs2, ws2 = pshape[0] // 2 + 1, pshape[1] // 2 + 1
        rr = np.random.randint(hs2, hi - hs2, npatches)
        cc = np.random.randint(ws2, wi - ws2, npatches)
        for r, c in zip(rr, cc):
            yield ut.col_map(sample, imagecols, ni.extract_patch, pshape, r, c)


@nut_processor
def ImagePatchesByMask(iterable, imagecol, maskcol, pshape, npos,
                       nneg=lambda npos: npos, pos=255, neg=0, retlabel=True):
    """
    samples >> ImagePatchesByMask(imagecol, maskcol, pshape, npos,
                                 nneg=lambda npos: npos,
                                 pos=255, neg=0, retlabel=True)


    Randomly sample positive/negative patches from image based on mask.

    A patch is positive if its center point has the value 'pos' in the
    mask (corresponding to the input image) and is negative for value 'neg'
    The mask must be of same size as image.

    >>> import numpy as np
    >>> np.random.seed(0)    # just to ensure stable doctest
    >>> img = np.reshape(np.arange(25), (5, 5))
    >>> mask = np.eye(5, dtype='uint8') * 255
    >>> samples = [(img, mask)]

    >>> get_patches = ImagePatchesByMask(0, 1, (3, 3), 2, 1)
    >>> for (p, l) in patches:
    ...     print p, l
    [[10 11 12]
     [15 16 17]
     [20 21 22]] 0
    [[12 13 14]
     [17 18 19]
     [22 23 24]] 1
    [[ 6  7  8]
     [11 12 13]
     [16 17 18]] 1

    >>> np.random.seed(0)    # just to ensure stable doctest
    >>> get_patches = ImagePatchesByMask(0, 1, (3, 3), 1, 1, retlabel=False)
    >>> for (p, m) in patches:
    ...     print p
    ...     print m
    [[12 13 14]
     [17 18 19]
     [22 23 24]]
    [[255   0   0]
     [  0 255   0]
     [  0   0 255]]
    [[10 11 12]
     [15 16 17]
     [20 21 22]]
    [[  0   0 255]
     [  0   0   0]
     [  0   0   0]]

    :param iterable iterable: Samples with images
    :param int imagecol: Index of sample column that contain image
    :param int maskcol: Index of sample column that contain mask
    :param tuple pshape: Shape of patch
    :param int npos: Number of positive patches to sample
    :param int|function nneg: Number of negative patches to sample or
        a function hat returns the number of negatives
        based on number of positives.
    :param int pos: Mask value indicating positives
    :param int neg: Mask value indicating negatives
    :param bool retlabel: True return label, False return mask patch
    :return: Iterator over samples where images are replaced by image patches
        and masks are replaced by labels [0,1] or mask patches
    :rtype: iterator
    """
    for sample in iterable:
        image, mask = sample[imagecol], sample[maskcol]
        if image.shape[:2] != mask.shape:
            raise ValueError('Image and mask size don''t match!')
        it = ni.sample_pn_patches(image, mask, pshape, npos, nneg, pos, neg)
        for img_patch, mask_patch, label in it:
            outsample = list(sample)[:]
            outsample[imagecol] = img_patch
            outsample[maskcol] = label if retlabel else mask_patch
            yield tuple(outsample)


@nut_processor
def ImagePatchesByAnnotation(iterable, imagecol, annocol, pshape, npos,
                             nneg=lambda npos: npos, pos=255, neg=0):
    """
    samples >> ImagePatchesByAnnotation(imagecol, annocol, pshape, npos,
                                 nneg=lambda npos: npos,
                                 pos=255, neg=0)

    Randomly sample positive/negative patches from image based on annotation.
    See imageutil.annotation2coords for annotation format.
    A patch is positive if its center point is within the annotated region
    and is negative otherwise

    >>> import numpy as np
    >>> np.random.seed(0)    # just to ensure stable doctest
    >>> img = np.reshape(np.arange(25), (5, 5))
    >>> anno = ('point', ((3, 2), (2, 3),))
    >>> samples = [(img, anno)]

    >>> get_patches = ImagePatchesByAnnotation(0, 1, (3, 3), 1, 1)
    >>> for (p, l) in patches:
    ...     print p, l
    [[12 13 14]
     [17 18 19]
     [22 23 24]] 0
    [[11 12 13]
     [16 17 18]
     [21 22 23]] 1
    [[7 8 9]
     [12 13 14]
     [17 18 19]] 1

    :param iterable iterable: Samples with images
    :param int imagecol: Index of sample column that contain image
    :param int annocol: Index of sample column that contain annotation
    :param tuple pshape: Shape of patch
    :param int npos: Number of positive patches to sample
    :param int|function nneg: Number of negative patches to sample or
        a function hat returns the number of negatives
        based on number of positives.
    :param int pos: Mask value indicating positives
    :param int neg: Mask value indicating negatives
    :return: Iterator over samples where images are replaced by image patches
        and annotations are replaced by labels [0,1]
    :rtype: iterator
    """
    for sample in iterable:
        image, annotations = sample[imagecol], sample[annocol]
        mask = ni.annotation2mask(image, annotations)
        n = len(annotations[1]) if annotations else 0
        it = ni.sample_pn_patches(image, mask, pshape, npos * n, nneg, pos, neg)
        for img_patch, _, label in it:
            outsample = list(sample)[:]
            outsample[imagecol] = img_patch
            outsample[annocol] = label
            yield tuple(outsample)


@nut_processor
def ImageAnnotationToMask(iterable, imagecol, annocol):
    """
    samples >> ImageAnnotationToMask(imagecol, annocol)

    Create mask for image annotation. Annotation are of the following formats.
    See imageutil.annotation2coords for details.
    ('point', ((x, y), ... ))
    ('circle', ((x, y, r), ...))
    ('rect', ((x, y, w, h), ...))
    ('polyline', (((x, y), (x, y), ...), ...))

    >>> import numpy as np
    >>> img = np.zeros((3, 3), dtype='uint8')
    >>> anno = ('point', ((0, 1), (2, 0)))
    >>> samples = [(img, anno)]
    >>> masks = samples >> ImageAnnotationToMask(0, 1) >> Collect()
    >>> masks[0][1]
    [[  0   0 255]
     [255   0   0]
     [  0   0   0]]

    :param iterable iterable: Samples with images and annotations
    :param int imagecol: Index of sample column that contain image
    :param int annocol: Index of sample column that contain annotation
    :return: Iterator over samples where annotations are replaced by masks
    :rtype: iterator
    """
    for sample in iterable:
        sample = list(sample)[:]  # going to mutate sample
        mask = ni.annotation2mask(sample[imagecol], sample[annocol])
        sample[annocol] = mask
        yield tuple(sample)


class ImageMean(NutFunction):
    """
    Compute, save mean over images and subtract from images.
    """

    def __init__(self, imagecol, filepath='image_means.npy'):
        """
         samples >> ImageMean(imagecol, filepath='image_means.npy')

         Construct ImageMean nut.

        :param int imagecol: Index of sample column that contain image
        :param string filepath: Path to file were mean values are saved
           and loaded from.
        """
        hasfile = path.exists(filepath)
        self.imagecol = imagecol
        self.filepath = filepath
        self.means = np.load(filepath, False) if hasfile else np.array([])

    def __call__(self, sample):
        """
        Subtract mean from images in samples.

        sub_mean = ImageMean(imagecol, filepath)
        samples >> sub_mean >> Consume()

        :param tuple sample: Sample that contains an image (at imagecol).
        :return: Sample with image where mean is subtracted.
                 Note that image will not be of dtype uint8 and
                 in range [0,255] anymore!
        :rtype: tuple
        """
        sample = list(sample)[:]  # going to mutate sample
        image = sample[self.imagecol].astype('float32')
        if not self.means.size:
            raise ValueError('Mean has not yet been computed!')
        if self.means.shape != image.shape:
            raise ValueError('Mean loaded was computed on different images?')
        sample[self.imagecol] = image - self.means
        return tuple(sample)

    def train(self):
        """
        Compute mean over images in samples.

        sub_mean = ImageMean(imagecol, filepath)
        samples >> sub_mean.train() >> Consume()

        :return: Input samples are returned unchanged
        :rtype: tuple
        """

        @nut_processor
        def Trainer(iterable, outer):
            img_mean = np.array([])
            for n_i, sample in enumerate(iterable):
                image = sample[outer.imagecol]
                if not img_mean.size:
                    img_mean = np.zeros(image.shape)
                img_mean += image
                yield sample
            outer.means = img_mean / (n_i + 1)
            if outer.filepath:
                np.save(outer.filepath, outer.means, False)

        return Trainer(self)


class ImageChannelMean(NutFunction):
    """
    Compute, save per-channel means over images and subtract from images.
    """

    def __init__(self, imagecol, filepath='image_channel_means.npy',
                 means=None):
        """
         samples >> ImageChannelMean(imagecol,
                                     filepath='image_channel_means.npy',
                                     means=None)

         Construct ImageChannelMean nut.

        :param int imagecol: Index of sample column that contain image
        :param string filepath: Path to file were mean values are saved
            and loaded from.
        :param list|tuple means: Mean values can be provided directly.
          In this case filepath will be ignored and training is not
          necessary.
        """
        hasfile = path.exists(filepath)
        self.imagecol = imagecol
        self.filepath = filepath
        if means:
            self.means = np.array([[means]])
        else:
            self.means = np.load(filepath, False) if hasfile else np.array([])

    def __call__(self, sample):
        """
        Subtract per-channel mean from images in samples.

        sub_mean = ImageChannelMean(imagecol, filepath='means.npy')
        samples >> sub_mean >> Consume()

        sub_mean = ImageChannelMean(imagecol, means=[197, 87, 101])
        samples >> sub_mean >> Consume()

        :param tuple sample: Sample that contains an image (at imagecol).
        :return: Sample with image where mean is subtracted.
                 Note that image will not be of dtype uint8 and
                 in range [0,255] anymore!
        :rtype: tuple
        """
        sample = list(sample)[:]  # going to mutate sample
        image = sample[self.imagecol].astype('float32')
        if not self.means.size:
            raise ValueError('Mean has not yet been computed!')
        if self.means.ndim != image.ndim:
            raise ValueError('Mean loaded was computed on different images?')
        sample[self.imagecol] = image - self.means
        return tuple(sample)

    def train(self):
        """
        Compute per-channel mean over images in samples.

        sub_mean = ImageChannelMean(imagecol, filepath)
        samples >> sub_mean.train() >> Consume()

        :return: Input samples are returned unchanged
        :rtype: tuple
        """

        @nut_processor
        def Trainer(iterable, outer):
            means = np.array([])
            for n_i, sample in enumerate(iterable):
                image = sample[outer.imagecol]
                img_means = np.mean(image, axis=(0, 1), keepdims=True)
                if not means.size:
                    means = np.zeros(img_means.shape)
                means += img_means
                yield sample
            outer.means = means / (n_i + 1)
            if outer.filepath:
                np.save(outer.filepath, outer.means, False)

        return Trainer(self)
