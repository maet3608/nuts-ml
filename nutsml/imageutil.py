"""
.. module:: imageutil
   :synopsis: Basic image processing utilities
"""
from __future__ import absolute_import, print_function

import numpy as np
import PIL as pil
import skimage.exposure as ske
import skimage.transform as skt
import skimage.color as skc
import skimage.util.shape as sks
import skimage.io as ski
import skimage.draw as skd
import matplotlib.patches as plp

from six.moves import range, map
from nutsflow.common import shapestr, isnan
from PIL import ImageEnhance as ie
from skimage.color import rgb2gray
from skimage import feature
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from warnings import warn


def load_image(filepath, as_grey=False, dtype='uint8', no_alpha=True):
    """
    Load image as numpy array from given filepath.

    Supported formats: gif, png, jpg, bmp, tif, npy

    >>> img = load_image('tests/data/img_formats/nut_color.jpg')
    >>> shapestr(img)
    '213x320x3'

    :param string filepath: Filepath to image file or numpy array.
    :param bool as_grey:
    :return: numpy array with shapes
             (h, w) for grayscale or monochrome,
             (h, w, 3) for RGB (3 color channels in last axis)
             (h, w, 4) for RGBA (for no_alpha = False)
             (h, w, 3) for RGBA (for no_alpha = True)
             pixel values are in range [0,255] for dtype = uint8
    :rtype: numpy ndarray
    """
    if filepath.endswith('.npy'):  # image as numpy array
        arr = np.load(filepath).astype(dtype)
        arr = rgb2gray(arr) if as_grey else arr
    else:
        img = ski.imread(filepath, as_gray=as_grey)
        arr = np.array(img, dtype=dtype)
    # https://github.com/scikit-image/scikit-image/issues/2406
    if arr.ndim == 1 and arr.shape[0] == 2:
        arr = arr[0]  # pragma: no cover
    if arr.ndim == 3 and arr.shape[2] == 4 and no_alpha:
        arr = arr[..., :3]  # cut off alpha channel
    return arr


def save_image(filepath, image):
    """
    Save numpy array as image (or numpy array) to given filepath.

    Supported formats: gif, png, jpg, bmp, tif, npy

    :param string filepath: File path for image file. Extension determines
        image file format, e.g. .gif
    :param numpy array image: Numpy array to save as image.
        Must be of shape (h,w) or (h,w,3) or (h,w,4)
    """
    if filepath.endswith('.npy'):  # image as numpy array
        np.save(filepath, image, allow_pickle=False)
    else:
        ski.imsave(filepath, image)


def arr_to_pil(image):
    """
    Convert numpy array to PIL image.

    >>> import numpy as np
    >>> rgb_arr = np.ones((5, 4, 3), dtype='uint8')
    >>> pil_img = arr_to_pil(rgb_arr)
    >>> pil_img.size
    (4, 5)

    :param ndarray image: Numpy array with dtype 'uint8' and dimensions
       (h,w,c) for RGB or (h,w) for gray-scale images.
    :return: PIL image
    :rtype: PIL.Image
    """
    if image.dtype != np.uint8:
        raise ValueError('Expect uint8 dtype but got: ' + str(image.dtype))
    if not (2 <= image.ndim <= 3):
        raise ValueError('Expect gray scale or RGB image: ' + str(image.ndim))

    return pil.Image.fromarray(image, 'RGB' if image.ndim == 3 else 'L')


def pil_to_arr(image):
    """
    Convert PIL image to Numpy array.

    >>> import numpy as np
    >>> rgb_arr = np.ones((5, 4, 3), dtype='uint8')
    >>> pil_img = arr_to_pil(rgb_arr)
    >>> arr = pil_to_arr(pil_img)
    >>> shapestr(arr)
    '5x4x3'

    :param PIL.Image image: PIL image (RGB or grayscale)
    :return: Numpy array
    :rtype: numpy.array with dtype 'uint8'
    """
    if image.mode not in {'L', 'RGB'}:
        raise ValueError('Expect RBG or grayscale but got:' + image.mode)
    return np.asarray(image)


def set_default_order(kwargs):
    """
    Set order parameter in kwargs for scikit-image functions.

    Default order is 1, which performs a linear interpolation of pixel values
    when images are rotated, resized and sheared. This is fine for images
    but causes unwanted pixel values in masks. This function set the default
    order to 0, which disables the interpolation.

    :param kwargs kwargs: Dictionary with keyword arguments.
    """
    if 'order' not in kwargs:
        kwargs['order'] = 0


def add_channel(image, channelfirst):
    """
    Add channel if missing and make first axis if requested.

    >>> import numpy as np
    >>> image = np.ones((10, 20))
    >>> image = add_channel(image, True)
    >>> shapestr(image)
    '1x10x20'

    :param ndarray image: RBG (h,w,3) or gray-scale image (h,w).
    :param bool channelfirst: If True, make channel first axis
    :return: Numpy array with channel (as first axis if makefirst=True)
    :rtype: numpy.array
    """
    if not 2 <= image.ndim <= 3:
        raise ValueError('Image must be 2 or 3 channel!')
    if image.ndim == 2:  # gray-scale image
        image = np.expand_dims(image, axis=-1)  # add channel axis
    return np.rollaxis(image, 2) if channelfirst else image


def floatimg2uint8(image):
    """
    Convert array with floats to 'uint8' and rescale from [0,1] to [0, 256].

    Converts only if image.dtype != uint8.

    >>> import numpy as np
    >>> image = np.eye(10, 20, dtype=float)
    >>> arr = floatimg2uint8(image)
    >>> np.max(arr)
    255

    :param numpy.array image: Numpy array with range [0,1]
    :return: Numpy array with range [0,255] and dtype 'uint8'
    :rtype: numpy array
    """
    return (image * 255).astype('uint8') if image.dtype != 'uint8' else image


def rerange(image, old_min, old_max, new_min, new_max, dtype):
    """
    Return image with values in new range.

    Note: The default range of images is [0, 255] and most image
    processing functions expect this range and will fail otherwise.
    However, as input to neural networks re-ranged images, e.g [-1, +1]
    are sometimes needed.

    >>> import numpy as np
    >>> image = np.array([[0, 255], [255, 0]])
    >>> rerange(image, 0, 255, -1, +1, 'float32')
    array([[-1.,  1.],
           [ 1., -1.]], dtype=float32)

    :param numpy.array image: Should be a numpy array of an image.
    :param int|float old_min: Current minimum value of image, e.g. 0
    :param int|float old_max: Current maximum value of image, e.g. 255
    :param int|float new_min: New minimum, e.g. -1.0
    :param int|float new_max: New maximum, e.g. +1.0
    :param numpy datatype dtype: Data type of output image,
           e.g. float32' or np.uint8
    :return: Image with values in new range.
    """
    image = image.astype('float32')
    old_range, new_range = old_max - old_min, new_max - new_min
    image = (image - old_min) / old_range * new_range + new_min
    return image.astype(dtype)


def identical(image):
    """
    Return input image unchanged.

    :param numpy.array image: Should be a numpy array of an image.
    :return: Same as input
    :rtype: Same as input
    """
    return image


def crop(image, x1, y1, x2, y2):
    """
    Crop image.

    >>> import numpy as np
    >>> image = np.reshape(np.arange(16, dtype='uint8'), (4, 4))
    >>> crop(image, 1, 2, 5, 5)
    array([[ 9, 10, 11],
           [13, 14, 15]], dtype=uint8)

    :param numpy array image: Numpy array.
    :param int x1: x-coordinate of left upper corner of crop (inclusive)
    :param int y1: y-coordinate of left upper corner of crop (inclusive)
    :param int x2: x-coordinate of right lower corner of crop (exclusive)
    :param int y2: y-coordinate of right lower corner of crop (exclusive)
    :return: Cropped image
    :rtype: numpy array
    """
    return image[y1:y2, x1:x2]


def crop_center(image, w, h):
    """
    Crop region with size w, h from center of image.

    Note that the crop is specified via w, h and not via shape (h,w).
    Furthermore if the image or the crop region have even dimensions,
    coordinates are rounded down.

    >>> import numpy as np
    >>> image = np.reshape(np.arange(16, dtype='uint8'), (4, 4))
    >>> crop_center(image, 3, 2)
    array([[ 4,  5,  6],
           [ 8,  9, 10]], dtype=uint8)

    :param numpy array image: Numpy array.
    :param int w: Width of crop
    :param int h: Height of crop
    :return: Cropped image
    :rtype: numpy array
    :raise: ValueError if image is smaller than crop region
    """
    iw, ih = image.shape[1], image.shape[0]
    dw, dh = iw - w, ih - h
    if dw < 0 or dh < 0:
        raise ValueError('Image too small for crop {}x{}'.format(iw, ih))
    return image[dh // 2:dh // 2 + h, dw // 2:dw // 2 + w]


def crop_square(image):
    """
    Crop image to square shape.

    Crops symmetrically left and right or top and bottom to achieve
    aspect ratio of one and preserves the largest dimension.

    :param numpy array image: Numpy array.
    :return: Cropped image
    :rtype: numpy array
    """
    iw, ih = image.shape[1], image.shape[0]
    if iw > ih:
        dw, mw = int((iw - ih) / 2), (iw - ih) % 2
        return crop(image, dw + mw, 0, iw - dw, ih)
    else:
        dh, mh = int((ih - iw) / 2), (ih - iw) % 2
        return crop(image, 0, dh + mh, iw, ih - dh)


def occlude(image, x, y, w, h, color=0):
    """
    Occlude image with a rectangular region.

    Occludes an image region with dimensions w,h centered on x,y with the
    given color. Invalid x,y coordinates will be clipped to ensure complete
    occlusion rectangle is within the image.

    >>> import numpy as np
    >>> image = np.ones((4, 5)).astype('uint8')
    >>> occlude(image, 2, 2, 2, 3)
    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 1, 1],
           [1, 0, 0, 1, 1],
           [1, 0, 0, 1, 1]], dtype=uint8)

    >>> image = np.ones((4, 4)).astype('uint8')
    >>> occlude(image, 0.5, 0.5, 0.5, 0.5)
    array([[1, 1, 1, 1],
           [1, 0, 0, 1],
           [1, 0, 0, 1],
           [1, 1, 1, 1]], dtype=uint8)

    :param numpy array image: Numpy array.
    :param int|float x: x coordinate for center of occlusion region.
                        Can be provided as fraction (float) of image width
    :param int|float y: y coordinate for center of occlusion region.
                        Can be provided as fraction (float) of image height
    :param int|float w: width of occlusion region.
                        Can be provided as fraction (float) of image width
    :param int|float h: height of occlusion region.
                        Can be provided as fraction (float) of image height
    :param int|tuple color: gray-scale or RGB color of occlusion.
    :return: Copy of input image with occluded region.
    :rtype: numpy array
    """
    frac = lambda c, m: int(m * c) if isinstance(c, float) else c
    iw, ih = image.shape[:2]
    x, y = frac(x, iw), frac(y, ih)
    w, h = frac(w, iw), frac(h, ih)
    r, c = int(y - h // 2), int(x - w // 2)
    r, c = max(min(r, ih - h), 0), max(min(c, iw - w), 0)
    image2 = image.copy()
    image2[r:r + h, c:c + w] = color
    return image2


def normalize_histo(image, gamma=1.0): # pragma no coverage
    """
    Perform histogram normalization on image.

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param float gamma: Factor for gamma adjustment.
    :return: Normalized image
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    image = ske.equalize_adapthist(image)
    image = ske.adjust_gamma(image, gamma=gamma)
    return floatimg2uint8(image)


def enhance(image, func, *args, **kwargs):
    """
    Enhance image using a PIL enhance function

    See the following link for details on PIL enhance functions:
    http://pillow.readthedocs.io/en/3.1.x/reference/ImageEnhance.html

    >>> from PIL.ImageEnhance import Brightness
    >>> image = np.ones((3,2), dtype='uint8')
    >>> enhance(image, Brightness, 0.0)
    array([[0, 0],
           [0, 0],
           [0, 0]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param function func: PIL ImageEnhance function
    :param args args: Argument list passed on to enhance function.
    :param kwargs kwargs: Key-word arguments passed on to enhance function
    :return: Enhanced image
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    image = arr_to_pil(image)
    image = func(image).enhance(*args, **kwargs)
    return pil_to_arr(image)


def change_contrast(image, contrast=1.0):
    """
    Change contrast of image.

    >>> image = np.eye(3, dtype='uint8') * 255
    >>> change_contrast(image, 0.5)
    array([[170,  42,  42],
           [ 42, 170,  42],
           [ 42,  42, 170]], dtype=uint8)

    See
    http://pillow.readthedocs.io/en/3.1.x/reference/ImageEnhance.html#PIL.ImageEnhance.Contrast

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param float contrast: Contrast [0, 1]
    :return: Image with changed contrast
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    return enhance(image, ie.Contrast, contrast)


def change_brightness(image, brightness=1.0):
    """
    Change brightness of image.

    >>> image = np.eye(3, dtype='uint8') * 255
    >>> change_brightness(image, 0.5)
    array([[127,   0,   0],
           [  0, 127,   0],
           [  0,   0, 127]], dtype=uint8)

    See
    http://pillow.readthedocs.io/en/3.1.x/reference/ImageEnhance.html#PIL.ImageEnhance.Brightness

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param float brightness: Brightness [0, 1]
    :return: Image with changed brightness
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    return enhance(image, ie.Brightness, brightness)


def change_sharpness(image, sharpness=1.0):
    """
    Change sharpness of image.

    >>> image = np.eye(3, dtype='uint8') * 255
    >>> change_sharpness(image, 0.5)
    array([[255,   0,   0],
           [  0, 196,   0],
           [  0,   0, 255]], dtype=uint8)

    See
    http://pillow.readthedocs.io/en/3.1.x/reference/ImageEnhance.html#PIL.ImageEnhance.Sharpness

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param float sharpness: Sharpness [0, ...]
    :return: Image with changed sharpness
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    return enhance(image, ie.Sharpness, sharpness)


def change_color(image, color=1.0):
    """
    Change color of image.

    >>> image = np.eye(3, dtype='uint8') * 255
    >>> change_color(image, 0.5)
    array([[255,   0,   0],
           [  0, 255,   0],
           [  0,   0, 255]], dtype=uint8)

    See
    http://pillow.readthedocs.io/en/3.1.x/reference/ImageEnhance.html#PIL.ImageEnhance.Color

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param float color: Color [0, 1]
    :return: Image with changed color
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    return enhance(image, ie.Color, color)


def extract_edges(image, sigma):   # pragma: no cover
    """
    Extract edges using the Canny algorithm.

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param float sigma: Standard deviation of the Gaussian filter.
    :return: Binary image with extracted edges
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    image = rgb2gray(image)
    params = dict(sigma=sigma, low_threshold=0.1, high_threshold=50.0,
                  mask=None, use_quantiles=False)
    image = feature.canny(image, **params)
    return image.astype('uint8') * 255


def gray2rgb(image):
    """
    Grayscale scale image to RGB image
    
    >>> image = np.eye(3, dtype='uint8') * 255
    >>> gray2rgb(image)
    array([[[255, 255, 255],
            [  0,   0,   0],
            [  0,   0,   0]],
    <BLANKLINE>
           [[  0,   0,   0],
            [255, 255, 255],
            [  0,   0,   0]],
    <BLANKLINE>
           [[  0,   0,   0],
            [  0,   0,   0],
            [255, 255, 255]]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'. 
    :return: RGB image
    :rtype:  numpy array with range [0,255] and dtype 'uint8'
    """
    return skc.gray2rgb(image)


def rgb2gray(image):
    """
    RGB scale image to grayscale image

    >>> image = np.eye(3, dtype='uint8') * 255
    >>> rgb2gray(image)
    array([[255,   0,   0],
           [  0, 255,   0],
           [  0,   0, 255]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'. 
    :return: grayscale image
    :rtype:  numpy array with range [0,255] and dtype 'uint8'
    """
    return floatimg2uint8(skc.rgb2gray(image))


def translate(image, dx, dy, **kwargs):
    """
    Shift image horizontally and vertically

    >>> image = np.eye(3, dtype='uint8') * 255
    >>> translate(image, 2, 1)
    array([[  0,   0,   0],
           [  0,   0, 255],
           [  0,   0,   0]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param dx: horizontal translation in pixels
    :param dy: vertical translation in pixels
    :param kwargs kwargs: Keyword arguments for the underlying scikit-image
       rotate function, e.g. order=1 for linear interpolation.
    :return: translated image
    :rtype:  numpy array with range [0,255] and dtype 'uint8'
    """
    set_default_order(kwargs)
    transmat = skt.AffineTransform(translation=(-dx, -dy))
    return skt.warp(image, transmat, preserve_range=True,
                    **kwargs).astype('uint8')


def rotate(image, angle=0, **kwargs):
    """
    Rotate image.

    For details see:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rotate

    For a smooth interpolation of images set 'order=1'. To rotate masks use
    the default 'order=0'.

    >>> image = np.eye(3, dtype='uint8')
    >>> rotate(image, 90)
    array([[0, 0, 1],
           [0, 1, 0],
           [1, 0, 0]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param float angle: Angle in degrees in counter-clockwise direction
    :param kwargs kwargs: Keyword arguments for the underlying scikit-image
       rotate function, e.g. order=1 for linear interpolation.
    :return: Rotated image
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    set_default_order(kwargs)
    return skt.rotate(image, angle, preserve_range=True,
                      **kwargs).astype('uint8')


def resize(image, w, h, anti_aliasing=False, **kwargs):
    """
    Resize image.

    Image can be up- or down-sized (using interpolation). For details see:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

    >>> image = np.ones((10,5), dtype='uint8')
    >>> resize(image, 4, 3)
    array([[1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param int w: Width in pixels.
    :param int h: Height in pixels.
    :param bool anti_aliasing: Toggle anti aliasing.
    :param kwargs kwargs: Keyword arguments for the underlying scikit-image
       resize function, e.g. order=1 for linear interpolation.
    :return: Resized image
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    set_default_order(kwargs)
    return skt.resize(image, (h, w), mode='constant', preserve_range=True,
                      anti_aliasing=anti_aliasing, **kwargs).astype('uint8')


def shear(image, shear_factor, **kwargs):
    """
    Shear image.

    For details see:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.AffineTransform

    >>> image = np.eye(3, dtype='uint8')
    >>> rotated = rotate(image, 45)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :param float shear_factor: Shear factor [0, 1]
    :param kwargs kwargs: Keyword arguments for the underlying scikit-image
       warp function, e.g. order=1 for linear interpolation.
    :return: Sheared image
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    set_default_order(kwargs)
    transform = skt.AffineTransform(shear=shear_factor)
    return skt.warp(image, transform, preserve_range=True,
                    **kwargs).astype('uint8')


def fliplr(image):
    """
    Flip image left to right.

    >>> image = np.reshape(np.arange(4, dtype='uint8'), (2,2))
    >>> fliplr(image)
    array([[1, 0],
           [3, 2]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :return: Flipped image
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    return np.fliplr(image)


def flipud(image):
    """
    Flip image up to down.

    >>> image = np.reshape(np.arange(4, dtype='uint8'), (2,2))
    >>> flipud(image)
    array([[2, 3],
           [0, 1]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
    :return: Flipped image
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    return np.flipud(image)


def distort_elastic(image, smooth=10.0, scale=100.0, seed=0):
    """
    Elastic distortion of images.

    Channel axis in RGB images will not be distorted but grayscale or
    RGB images are both valid inputs. RGB and grayscale images will be
    distorted identically for the same seed.

    Simard, et. al, "Best Practices for Convolutional Neural Networks
    applied to Visual Document Analysis",
    in Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    :param ndarray image: Image of shape [h,w] or [h,w,c]
    :param float smooth: Smoothes the distortion.
    :param float scale: Scales the distortion.
    :param int seed: Seed for random number generator. Ensures that for the
                     same seed images are distorted identically.
    :return: Distorted image with same shape as input image.
    :rtype: ndarray
    """
    # create random, smoothed displacement field
    rnd = np.random.RandomState(int(seed))
    h, w = image.shape[:2]
    dxy = rnd.rand(2, h, w, 3) * 2 - 1
    dxy = gaussian_filter(dxy, smooth, mode="constant")
    dxy = dxy / np.linalg.norm(dxy) * scale
    dxyz = dxy[0], dxy[1], np.zeros_like(dxy[0])

    # create transformation coordinates and deform image
    is_color = len(image.shape) == 3
    ranges = [np.arange(d) for d in image.shape]
    grid = np.meshgrid(*ranges, indexing='ij')
    add = lambda v, dv: v + dv if is_color else v + dv[:, :, 0]
    idx = [np.reshape(add(v, dv), (-1, 1)) for v, dv in zip(grid, dxyz)]
    distorted = map_coordinates(image, idx, order=1, mode='reflect')

    return distorted.reshape(image.shape)


def polyline2coords(points):
    """
    Return row and column coordinates for a polyline.

    >>> rr, cc = polyline2coords([(0, 0), (2, 2), (2, 4)])
    >>> list(rr)
    [0, 1, 2, 2, 3, 4]
    >>> list(cc)
    [0, 1, 2, 2, 2, 2]

    :param list of tuple points: Polyline in format [(x1,y1), (x2,y2), ...] 
    :return: tuple with row and column coordinates in numpy arrays
    :rtype: tuple of numpy array
    """
    coords = []
    for i in range(len(points) - 1):
        xy = list(map(int, points[i] + points[i + 1]))
        coords.append(skd.line(xy[1], xy[0], xy[3], xy[2]))
    return [np.hstack(c) for c in zip(*coords)]


def mask_where(mask, value):
    """
    Return x,y coordinates where mask has specified value

    >>> mask = np.eye(3, dtype='uint8')
    >>> mask_where(mask, 1).tolist()
    [[0, 0], [1, 1], [2, 2]]

    :param numpy array mask: Numpy array with range [0,255] and dtype 'uint8'.
    :return: Array with x,y coordinates
    :rtype: numpy array with shape Nx2 where each row contains x, y
    """
    return np.transpose(np.where(mask == value)).astype('int32')


def mask_choice(mask, value, n):
    """
    Random selection of n points where mask has given value

    >>> np.random.seed(1)   # ensure same random selection for doctest
    >>> mask = np.eye(3, dtype='uint8')
    >>> mask_choice(mask, 1, 2).tolist()
    [[0, 0], [2, 2]]

    :param numpy array mask: Numpy array with range [0,255] and dtype 'uint8'.
    :param int n: Number of points to select. If n is larger than the
      points available only the available points will be returned.
    :return: Array with x,y coordinates
    :rtype: numpy array with shape nx2 where each row contains x, y
    """
    points = mask_where(mask, value)
    n = min(n, points.shape[0])
    return points[np.random.choice(points.shape[0], n, replace=False), :]


def extract_patch(image, pshape, r, c):
    """
    Extract a patch of given shape, centered at r,c of given shape from image.

    Note that there is no checking if the patch region is inside the image.

    >>> image = np.reshape(np.arange(16, dtype='uint8'), (4, 4))
    >>> extract_patch(image, (2, 3), 2, 2)
    array([[ 5,  6,  7],
           [ 9, 10, 11]], dtype=uint8)

    :param numpy array image: Numpy array with range [0,255] and dtype 'uint8'.
       Can be of shapes MxN, MxNxC.
    :param tuple pshape: Shape of patch. #Dimensions must match image.
    :param int r: Row for center of patch
    :param int c: Column for center of patch
    :return: numpy array with shape pshape
    :rtype: numpy array with range [0,255] and dtype 'uint8'
    """
    h, w = pshape[0], pshape[1]
    r, c = int(r - h // 2), int(c - w // 2)
    return image[r:r + h, c:c + w]


def patch_iter(image, shape=(3, 3), stride=1):
    """
    Extracts patches from images with given shape.

    Patches are extracted in a regular grid with the given stride,
    starting in the left upper corner and then row-wise.
    Image can be gray-scale (no third channel dim) or color.

    >>> import numpy as np
    >>> img = np.reshape(np.arange(12), (3, 4))
    >>> for p in patch_iter(img, (2, 2), 2):
    ...     print(p)
    [[0 1]
     [4 5]]
    [[2 3]
     [6 7]]

    :param ndarray image: Numpy array of shape h,w,c or h,w.
    :param tuple shape: Shape of patch (h,w)
    :param int stride: Step size of grid patches are extracted from
    :return: Iterator over patches
    :rtype: Iterator
    """
    # view_as_windows requires contiguous array, which we ensure here
    if not image.flags['C_CONTIGUOUS']:
        warn('Image is not contiguous and will be copied!')
        image = np.ascontiguousarray(image)

    is_gray = image.ndim == 2
    wshape = shape if is_gray else (shape[0], shape[1], image.shape[2])
    views = sks.view_as_windows(image, wshape, stride)
    rows, cols = views.shape[:2]

    def patch_gen():
        for r in range(rows):
            for c in range(cols):
                yield views[r, c] if is_gray else views[r, c, 0]

    return patch_gen()


# Note that masked arrays don't work here since the where(pred) function ignores
# the mask (returns all coordinates) for where(mask == 0).
def centers_inside(centers, image, pshape):
    """
    Filter center points of patches ensuring that patch is inside of image.

    >>> centers = np.array([[1, 2], [0,1]])
    >>> image = np.zeros((3, 4))
    >>> centers_inside(centers, image, (3, 3)).astype('uint8')
    array([[1, 2]], dtype=uint8)

    :param ndarray(n,2) centers: Center points of patches.
    :param ndarray(h,w) image: Image the patches should be inside.
    :param tuple pshape: Patch shape of form (h,w)
    :return: Patch centers where the patch is completely inside the image.
    :rtype: ndarray of shape (n, 2)
    """
    if not centers.shape[0]:  # list of centers is empty
        return centers
    h, w = image.shape[:2]
    h2, w2 = pshape[0] // 2, pshape[1] // 2
    minr, maxr, minc, maxc = h2 - 1, h - h2, w2 - 1, w - w2
    rs, cs = centers[:, 0], centers[:, 1]
    return centers[np.all([rs > minr, rs < maxr, cs > minc, cs < maxc], axis=0)]


def sample_mask(mask, value, pshape, n):
    """
    Randomly pick n points in mask where mask has given value.

    Ensure that only points picked that can be center of a patch with
    shape pshape that is inside the mask.

    >>> mask = np.zeros((3, 4))
    >>> mask[1, 2] = 1
    >>> sample_mask(mask, 1, (1, 1), 1)
    array([[1, 2]], dtype=uint16)

    :param ndarray mask: Mask
    :param int value: Sample points in mask that have this value.
    :param tuple pshape: Patch shape of form (h,w)
    :param int n: Number of points to sample. If there is not enough points
       to sample from a smaller number will be returned. If there are not
       points at all np.empty((0, 2)) will be returned.
    :return: Center points of patches within the mask where the center point
      has the given mask value.
    :rtype: ndarray of shape (n, 2)
    """
    centers = np.transpose(np.where(mask == value))
    centers = centers_inside(centers, mask, pshape).astype('uint16')
    n = min(n, centers.shape[0])
    if not n:
        return np.empty((0, 2))
    return centers[np.random.choice(centers.shape[0], n, replace=False), :]


def sample_labeled_patch_centers(mask, value, pshape, n, label):
    """
    Randomly pick n points in mask where mask has given value and add label.

    Same as imageutil.sample_mask but adds given label to each center

    >>> mask = np.zeros((3, 4))
    >>> mask[1, 2] = 1
    >>> sample_labeled_patch_centers(mask, 1, (1, 1), 1, 0)
    array([[1, 2, 0]], dtype=uint16)

    :param ndarray mask: Mask
    :param int value: Sample points in mask that have this value.
    :param tuple pshape: Patch shape of form (h,w)
    :param int n: Number of points to sample. If there is not enough points
       to sample from a smaller number will be returned. If there are not
       points at all np.empty((0, 2)) will be returned.
    :param int label: Numeric label to append to each center point
    :return: Center points of patches within the mask where the center point
      has the given mask value and the label
    :rtype: ndarray of shape (n, 3)
    """
    centers = sample_mask(mask, value, pshape, n)
    labels = np.full((centers.shape[0], 1), label, dtype=np.uint8)
    return np.hstack((centers, labels))


def sample_patch_centers(mask, pshape, npos, nneg, pos=255, neg=0):
    """
    Sample positive and negative patch centers where mask value is pos or neg.

    The sampling routine ensures that the patch is completely inside the mask.

    >>> np.random.seed(0)   # just to ensure consistent doctest
    >>> mask = np.zeros((3, 4))
    >>> mask[1, 2] = 255
    >>> sample_patch_centers(mask, (2, 2), 1, 1)
    array([[1, 1, 0],
           [1, 2, 1]], dtype=uint16)

    :param ndarray mask: Mask
    :param tuple pshape: Patch shape of form (h,w)
    :param int npos: Number of positives to sample.
    :param int nneg: Number of negatives to sample.
    :param int pos: Value for positive points in mask
    :param int neg: Value for negative points in mask
    :return: Center points of patches within the mask where the center point
      has the given mask value (pos, neg) and the label (1, 0)
    :rtype: ndarray of shape (n, 3)
    """
    pcenters = sample_labeled_patch_centers(mask, pos, pshape, npos, 1)
    nneg = nneg(npos) if hasattr(nneg, '__call__') else nneg
    ncenters = sample_labeled_patch_centers(mask, neg, pshape, nneg, 0)

    # return all labeled patch center points in random order
    labeled_centers = np.vstack((pcenters, ncenters))
    np.random.shuffle(labeled_centers)
    return labeled_centers


def sample_pn_patches(image, mask, pshape, npos, nneg, pos=255, neg=0):
    """
    Sample positive and negative patches where mask value is pos or neg.

    The sampling routine ensures that the patch is completely inside the
    image and mask and that a patch a the same position is extracted from
    the image and the mask.

    >>> np.random.seed(0)   # just to ensure consistent doctest
    >>> mask = np.zeros((3, 4), dtype='uint8')
    >>> img = np.reshape(np.arange(12, dtype='uint8'), (3, 4))
    >>> mask[1, 2] = 255
    >>> for ip, mp, l in sample_pn_patches(img, mask, (2, 2), 1, 1):
    ...     print(ip)
    ...     print(mp)
    ...     print(l)
    [[0 1]
     [4 5]]
    [[0 0]
     [0 0]]
    0
    [[1 2]
     [5 6]]
    [[  0   0]
     [  0 255]]
    1

    :param ndarray mask: Mask
    :param tuple pshape: Patch shape of form (h,w)
    :param int npos: Number of positives to sample.
    :param int nneg: Number of negatives to sample.
    :param int pos: Value for positive points in mask
    :param int neg: Value for negative points in mask
    :return: Image and mask patches where the patch center point
      has the given mask value (pos, neg) and the label (1, 0)
    :rtype: tuple(image_patch, mask_patch, label)
    """
    for r, c, label in sample_patch_centers(mask, pshape, npos, nneg, pos, neg):
        img_patch = extract_patch(image, pshape, r, c)
        mask_patch = extract_patch(mask, pshape, r, c)
        yield img_patch, mask_patch, label


def annotation2coords(image, annotation):
    """
    Convert geometric annotation in image to pixel coordinates.

    For example, given a rectangular region annotated in an image as
    ('rect', ((x, y, w, h))) the function returns the coordinates of all pixels
    within this region as (row, col) position tuples.

    The following annotation formats are supported:
    ('point', ((x, y), ... ))
    ('circle', ((x, y, r), ...))
    ('ellipse', ((x, y, rx, ry, rot), ...))
    ('rect', ((x, y, w, h), ...))
    ('polyline', (((x, y), (x, y), ...), ...))

    Annotation regions can exceed the image dimensions and will be clipped.
    Note that annotation is in x,y order while output is r,c (row, col).


    >>> import numpy as np
    >>> img = np.zeros((5, 5), dtype='uint8')
    >>> anno = ('point', ((1, 1), (1, 2)))
    >>> for rr, cc in annotation2coords(img, anno):
    ...     print(list(rr), list(cc))
    [1] [1]
    [2] [1]

    :param ndarray image: Image
    :param annotation annotation: Annotation of an image region such as
      point, circle, rect or polyline
    :return: Coordinates of pixels within the (clipped) region.
    :rtype: generator over tuples (row, col)
    """
    if not annotation or isnan(annotation):
        return
    shape = image.shape[:2]
    kind, geometries = annotation
    for geo in geometries:
        if kind == 'point':
            if geo[1] < shape[0] and geo[0] < shape[1]:
                rr, cc = np.array([geo[1]]), np.array([geo[0]])
            else:
                rr, cc = np.array([]), np.array([])
        elif kind == 'circle':
            rr, cc = skd.circle(geo[1], geo[0], geo[2], shape=shape)
            #rr, cc = skd.disc((geo[1], geo[0]), geo[2], shape=shape)
        elif kind == 'ellipse':
            rr, cc = skd.ellipse(geo[1], geo[0], geo[3], geo[2],
                                 rotation=geo[4], shape=shape)
        elif kind == 'rect':
            x, y, w, h = geo
            rr, cc = skd.rectangle((y,x), extent=(h,w), shape=shape)
            rr, cc = rr.flatten('F'), cc.flatten('F')
        elif kind == 'polyline':
            if geo[0] == geo[-1]:  # closed polyline => draw fill polygon
                xs, ys = zip(*geo)
                rr, cc = skd.polygon(ys, xs, shape=shape)
            else:
                rr, cc = polyline2coords(geo)
        else:
            raise ValueError('Invalid kind of annotation: ' + kind)
        if not rr.size or not cc.size:
            err_msg = 'Annotation {}:{} '.format(kind, geo)
            err_msg += 'outside image {}! Image transformed?'.format(shape)
            raise ValueError(err_msg)
        yield rr, cc


def annotation2pltpatch(annotation, **kwargs):
    """
    Convert geometric annotation to matplotlib geometric objects (=patches)

    For details regarding matplotlib patches see:
    http://matplotlib.org/api/patches_api.html
    For annotation formats see:
    imageutil.annotation2coords

    :param annotation annotation: Annotation of an image region such as
      point, circle, rect or polyline
    :return: matplotlib.patches
    :rtype: generator over matplotlib patches
    """
    if not annotation or isnan(annotation):
        return
    kind, geometries = annotation
    for geo in geometries:
        if kind == 'point':
            pltpatch = plp.CirclePolygon((geo[0], geo[1]), 1, **kwargs)
        elif kind == 'circle':
            pltpatch = plp.Circle((geo[0], geo[1]), geo[2], **kwargs)
        elif kind == 'rect':
            x, y, w, h = geo
            pltpatch = plp.Rectangle((x, y), w, h, **kwargs)
        elif kind == 'polyline':
            pltpatch = plp.Polygon(geo, closed=False, **kwargs)
        else:
            raise ValueError('Invalid kind of annotation: ' + kind)
        yield pltpatch


def annotation2mask(image, annotations, pos=255):
    """
    Convert geometric annotation to mask.

    For annotation formats see:
    imageutil.annotation2coords

    >>> import numpy as np
    >>> img = np.zeros((3, 3), dtype='uint8')
    >>> anno = ('point', ((0, 1), (2, 0)))
    >>> annotation2mask(img, anno)
    array([[  0,   0, 255],
           [255,   0,   0],
           [  0,   0,   0]], dtype=uint8)

    :param annotation annotation: Annotation of an image region such as
      point, circle, rect or polyline
    :param int pos: Value to write in mask for regions defined by annotation
    :param numpy array image: Image annotation refers to.
      Returned mask will be of same size.
    :return: Mask with annotation
    :rtype: numpy array
    """
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for rr, cc in annotation2coords(image, annotations):
        mask[rr, cc] = pos
    return mask
