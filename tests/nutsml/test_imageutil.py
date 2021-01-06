"""
.. module:: test_imageutil
   :synopsis: Unit tests for imageutil module
"""
from __future__ import print_function

import pytest
import sys
import os

import os.path as op
import PIL as pil
import numpy as np
import skimage as ski
import nutsml.imageutil as ni
import numpy.testing as nt
import matplotlib.patches as plp

from warnings import warn
from glob import glob
from PIL import ImageEnhance as ie

# Set to True to create test data
CREATE_TEST_DATA = False


@pytest.fixture(scope="function")
def datadirs():
    imagedir = 'tests/data/img/'
    formatsdir = 'tests/data/img_formats/'
    arraysdir = 'tests/data/img_arrays/'
    processeddir = 'tests/data/img_processed/'

    return imagedir, formatsdir, arraysdir, processeddir


def assert_equal_image(imagepath, image, rtol=0.01, atol=0.01):
    if CREATE_TEST_DATA:
        ni.save_image(imagepath, image)
    expected = ni.load_image(imagepath)
    diff = abs(expected - image).sum()
    print('absolute difference', diff)
    print('relative difference', 100.0 * diff / expected.sum())
    nt.assert_allclose(expected, image, rtol=rtol, atol=atol)


def test_load_image(datadirs):
    h, w = 213, 320
    _, formatsdir, arraydir, _ = datadirs
    pathpattern = formatsdir + '*'

    for filepath in glob(pathpattern):
        img = ni.load_image(filepath)
        is_color = 'color' in filepath
        assert img.shape == (h, w, 3) if is_color else (h, w)
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert np.max(img) <= 255
        assert np.min(img) >= 0

    for filepath in glob(pathpattern):
        img = ni.load_image(filepath, as_grey=True)
        assert img.shape == (h, w)
        assert np.max(img) <= 255
        assert np.min(img) >= 0

    for filepath in glob(pathpattern):
        img = ni.load_image(filepath)
        fdir, fname = op.split(filepath)
        arr = np.load(arraydir + fname + '.npy')
        if not filepath.endswith('.tif'):
            nt.assert_allclose(img, arr, rtol=0.1, atol=255)


def test_save_image(datadirs):
    _, formatsdir, _, _ = datadirs
    formats = ['gif', 'png', 'jpg', 'bmp', 'tif', 'npy']
    for format in formats:
        inpath = formatsdir + 'nut_color.' + format
        outpath = formatsdir + 'temp_nut_color.' + format
        image = ni.load_image(inpath)
        ni.save_image(outpath, image)
        loaded = ni.load_image(outpath)
        os.remove(outpath)
        # Saved and loaded JPG images can vary greatly (lossy format) when using
        # skimage under different OS and a direct comparison often fails.
        # Therefore, only shape and mean value are verified for JPG images.
        if format == 'jpg':
            assert abs(np.mean(image) - np.mean(loaded)) < 0.1
            assert loaded.shape == (213, 320, 3)
        else:
            nt.assert_allclose(image, loaded, rtol=0.1, atol=255)


def test_arr_to_pil():
    rgb_arr = np.ones((5, 4, 3), dtype='uint8')
    pil_img = ni.arr_to_pil(rgb_arr)
    assert pil_img.size == (4, 5)
    assert pil_img.mode == 'RGB'

    gray_arr = np.ones((5, 4), dtype='uint8')
    pil_img = ni.arr_to_pil(gray_arr)
    assert pil_img.size == (4, 5)
    assert pil_img.mode == 'L'

    with pytest.raises(ValueError) as ex:
        image = np.ones((10,), dtype='float')
        ni.arr_to_pil(image)
    assert str(ex.value).startswith('Expect uint8 dtype but got')

    with pytest.raises(ValueError) as ex:
        image = np.ones((10,), dtype='uint8')
        ni.arr_to_pil(image)
    assert str(ex.value).startswith('Expect gray scale or RGB image')


def test_pil_to_arr():
    rgb_arr = np.ones((5, 4, 3), dtype='uint8')
    pil_img = ni.arr_to_pil(rgb_arr)
    arr = ni.pil_to_arr(pil_img)
    nt.assert_allclose(rgb_arr, arr)
    assert arr.dtype == np.uint8

    gray_arr = np.ones((5, 4), dtype='uint8')
    pil_img = ni.arr_to_pil(gray_arr)
    arr = ni.pil_to_arr(pil_img)
    nt.assert_allclose(gray_arr, arr)
    assert arr.dtype == np.uint8

    with pytest.raises(ValueError) as ex:
        rgb_arr = np.ones((5, 4, 3), dtype='uint8')
        hsv_img = pil.Image.fromarray(rgb_arr, 'HSV')
        ni.pil_to_arr(hsv_img)
    assert str(ex.value).startswith('Expect RBG or grayscale but got')


def test_set_default_order():
    kwargs = {}
    ni.set_default_order(kwargs)
    assert kwargs['order'] == 0
    kwargs = {'order': 1}
    ni.set_default_order(kwargs)
    assert kwargs['order'] == 1


def test_add_channel():
    image = np.ones((10, 20))
    assert ni.add_channel(image, True).shape == (1, 10, 20)
    assert ni.add_channel(image, False).shape == (10, 20, 1)

    image = np.ones((10, 20, 3))
    assert ni.add_channel(image, True).shape == (3, 10, 20)
    assert ni.add_channel(image, False).shape == (10, 20, 3)

    with pytest.raises(ValueError) as ex:
        image = np.ones((10,))
        ni.add_channel(image, True)
    assert str(ex.value).startswith('Image must be 2 or 3 channel!')

    with pytest.raises(ValueError) as ex:
        image = np.ones((10, 20, 3, 1))
        ni.add_channel(image, True)
    assert str(ex.value).startswith('Image must be 2 or 3 channel!')


def test_floatimg2uint8():
    image = np.eye(10, 20, dtype=float)
    arr = ni.floatimg2uint8(image)
    assert np.min(arr) == 0
    assert np.max(arr) == 255
    assert arr.shape == image.shape


def test_rerange():
    image = np.eye(3, 4, dtype=float)
    arr = ni.rerange(image, 0.0, 1.0, -10, +20, int)
    assert np.min(arr) == -10
    assert np.max(arr) == +20
    assert arr.dtype == int
    assert arr.shape == image.shape


def test_identical():
    image = np.ones((5, 4, 3), dtype='uint8')
    nt.assert_allclose(image, ni.identical(image))


def test_crop():
    rgb_arr = np.ones((6, 7, 3), dtype='uint8')
    cropped = ni.crop(rgb_arr, 1, 2, 5, 5)
    assert cropped.shape == (3, 4, 3)
    assert cropped.dtype == np.uint8

    gray_arr = np.ones((6, 7), dtype='uint8')
    cropped = ni.crop(gray_arr, 1, 2, 5, 5)
    assert cropped.shape == (3, 4)

    image = np.reshape(np.arange(16, dtype='uint8'), (4, 4))
    cropped = ni.crop(image, 1, 2, 5, 5)
    expected = np.array([[9, 10, 11], [13, 14, 15]])
    nt.assert_allclose(expected, cropped)


def test_crop_square():
    rgb_arr = np.ones((6, 7, 3), dtype='uint8')
    cropped = ni.crop_square(rgb_arr)
    assert cropped.shape == (6, 6, 3)
    assert cropped.dtype == np.uint8

    gray_arr = np.ones((6, 7), dtype='uint8')
    cropped = ni.crop_square(gray_arr)
    assert cropped.shape == (6, 6)

    image = np.reshape(np.arange(12, dtype='uint8'), (4, 3))
    cropped = ni.crop_square(image)
    expected = np.array([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
    nt.assert_allclose(expected, cropped)


def test_crop_center():
    image = np.reshape(np.arange(16, dtype='uint8'), (4, 4))
    cropped = ni.crop_center(image, 3, 2)
    expected = np.array([[4, 5, 6], [8, 9, 10]])
    nt.assert_allclose(expected, cropped)

    with pytest.raises(ValueError) as ex:
        ni.crop_center(image, 5, 6)
    assert str(ex.value).startswith('Image too small for crop')


def test_occlude():
    image = np.ones((4, 5)).astype('uint8')
    occluded = ni.occlude(image, 3, 2, 2, 3, color=3)
    expected = np.array([[1, 1, 1, 1, 1], [1, 1, 3, 3, 1],
                         [1, 1, 3, 3, 1], [1, 1, 3, 3, 1]], dtype='uint8')
    assert occluded.dtype == np.uint8
    nt.assert_allclose(expected, occluded)

    image = np.ones((4, 5)).astype('uint8')
    occluded = ni.occlude(image, 0.5, 0.5, 3, 2)
    expected = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1],
                         [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype='uint8')
    assert occluded.dtype == np.uint8
    nt.assert_allclose(expected, occluded)

    rgb_arr = np.ones((2, 2, 3)).astype('uint8')
    color = (2, 3, 4)
    occluded = ni.occlude(rgb_arr, 1, 1, 1, 1, color=color)
    assert tuple(occluded[1, 1, :]) == color


def test_enhance():
    image = np.ones((5, 4), dtype='uint8')
    expected = np.zeros((5, 4), dtype='uint8')
    enhanced = ni.enhance(image, ie.Brightness, 0.0)
    nt.assert_allclose(expected, enhanced)
    assert enhanced.dtype == np.uint8


def test_contrast(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.change_contrast(img_arr, 0.5)
    imagepath = processeddir + 'nut_color_contrast.bmp'
    assert_equal_image(imagepath, new_img)


def test_brightness(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.change_brightness(img_arr, 0.5)
    imagepath = processeddir + 'nut_color_brightness.bmp'
    assert_equal_image(imagepath, new_img)


def test_sharpness(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.change_sharpness(img_arr, 0.5)
    imagepath = processeddir + 'nut_color_sharpness.bmp'
    assert_equal_image(imagepath, new_img, rtol=0, atol=10)


def test_change_color(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.change_color(img_arr, 2.0)
    imagepath = processeddir + 'nut_color_color.bmp'
    assert_equal_image(imagepath, new_img, rtol=0.1, atol=255)


def test_extract_edges(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.extract_edges(img_arr, 2.0)
    imagepath = processeddir + 'nut_color_edges.bmp'
    assert_equal_image(imagepath, new_img, rtol=0.1, atol=255)


def test_gray2rgb(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_grayscale.bmp')
    new_img = ni.gray2rgb(img_arr)
    imagepath = processeddir + 'nut_grayscale_2rgb.bmp'
    assert_equal_image(imagepath, new_img)


def test_rgb2gray(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.rgb2gray(img_arr)
    imagepath = processeddir + 'nut_color_2gray.bmp'
    assert_equal_image(imagepath, new_img)


def test_translate(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.translate(img_arr, 50, 100)
    imagepath = processeddir + 'nut_color_translated.bmp'
    assert_equal_image(imagepath, new_img)


def test_translate(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.translate(img_arr, 50, 100)
    imagepath = processeddir + 'nut_color_translated.bmp'
    assert_equal_image(imagepath, new_img)


def test_rotate(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.rotate(img_arr, 45, order=1)
    imagepath = processeddir + 'nut_color_rotated.bmp'
    assert_equal_image(imagepath, new_img, rtol=0.1, atol=0.1)

    img_arr = np.eye(4, dtype=np.uint8)
    exp_img = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    new_img = ni.rotate(img_arr, 90)
    nt.assert_allclose(new_img, exp_img)


def test_shear(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.shear(img_arr, 0.5)
    imagepath = processeddir + 'nut_color_sheared.bmp'
    assert_equal_image(imagepath, new_img)


def test_distort_elastic(datadirs):
    imagedir, _, _, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.distort_elastic(img_arr, 10, 5000)
    imagepath = processeddir + 'nut_color_elastic.bmp'
    assert_equal_image(imagepath, new_img)

    img_arr = ni.load_image(imagedir + 'nut_grayscale.bmp')
    new_img = ni.distort_elastic(img_arr, 10, 5000)
    imagepath = processeddir + 'nut_grayscale_elastic.bmp'
    assert_equal_image(imagepath, new_img)


def test_mask_where():
    expected = np.array([[0, 0], [1, 1], [2, 2]], dtype='int64')
    mask = np.eye(3, dtype='uint8')
    points = ni.mask_where(mask, 1)
    nt.assert_allclose(expected, points)


def test_mask_choice():
    np.random.seed(1)
    expected = np.array([[0, 0], [2, 2]], dtype='int64')
    mask = np.eye(3, dtype='uint8')
    points = ni.mask_choice(mask, 1, 2)
    nt.assert_allclose(expected, points)


def test_extract_patch():
    expected = np.array([[5, 6, 7], [9, 10, 11]], dtype='uint8')
    image = np.reshape(np.arange(16, dtype='uint8'), (4, 4))
    patch = ni.extract_patch(image, (2, 3), 2, 2)
    nt.assert_allclose(expected, patch)


def test_patch_iter():
    img = np.reshape(np.arange(12), (3, 4))

    patches = list(ni.patch_iter(img, (2, 2), 2))
    expected = [np.array([[0, 1], [4, 5]]),
                np.array([[2, 3], [6, 7]])]
    for p, e in zip(patches, expected):
        nt.assert_allclose(p, e)

    patches = list(ni.patch_iter(img, (2, 2), 3))
    expected = [np.array([[0, 1], [4, 5]])]
    for p, e in zip(patches, expected):
        nt.assert_allclose(p, e)

    patches = list(ni.patch_iter(img, (1, 3), 1))
    expected = [np.array([[0, 1, 2]]), np.array([[1, 2, 3]]),
                np.array([[4, 5, 6]]), np.array([[5, 6, 7]]),
                np.array([[8, 9, 10]]), np.array([[9, 10, 11]])]
    for p, e in zip(patches, expected):
        nt.assert_allclose(p, e)


@pytest.mark.filterwarnings(
    'ignore:Image is not contiguous and will be copied!')
def test_patch_iter_notcontiguous():
    img = np.asfortranarray(np.reshape(np.arange(12), (3, 4)))

    patches = list(ni.patch_iter(img, (2, 2), 2))
    expected = [np.array([[0, 1], [4, 5]]),
                np.array([[2, 3], [6, 7]])]
    for p, e in zip(patches, expected):
        nt.assert_allclose(p, e)


def test_patch_iter_warning():
    with pytest.warns(UserWarning):
        img = np.asfortranarray(np.reshape(np.arange(12), (3, 4)))
        list(ni.patch_iter(img, (2, 2), 2))
        warn("Image is not contiguous and will be copied!", UserWarning)


def test_patch_iter_3channel():
    img = np.reshape(np.arange(12 * 3), (3, 4, 3))

    patches = list(ni.patch_iter(img, (2, 2), 2))
    expected = [
        np.array([[[0, 1, 2], [3, 4, 5]], [[12, 13, 14], [15, 16, 17]]]),
        np.array([[[6, 7, 8], [9, 10, 11]], [[18, 19, 20], [21, 22, 23]]])]
    for p, e in zip(patches, expected):
        nt.assert_allclose(p, e)


def test_centers_inside():
    centers = np.array([[1, 2], [0, 1]])
    image = np.zeros((3, 4))

    expected = np.array([[1, 2]])
    result = ni.centers_inside(centers, image, (3, 3))
    nt.assert_allclose(expected, result)

    result = ni.centers_inside(centers, image, (4, 3))
    assert not result.size


def test_sample_mask():
    mask = np.zeros((3, 4))
    mask[1, 2] = 1

    centers = ni.sample_mask(mask, 1, (1, 1), 1)
    nt.assert_allclose(centers, [[1, 2]])

    centers = ni.sample_mask(mask, 1, (1, 1), 0)
    nt.assert_allclose(centers, np.empty((0, 2)))

    centers = ni.sample_mask(mask, 2, (1, 1), 1)
    nt.assert_allclose(centers, np.empty((0, 2)))

    centers = ni.sample_mask(mask, 1, (4, 3), 1)
    nt.assert_allclose(centers, np.empty((0, 2)))


def test_sample_labeled_patch_centers():
    mask = np.zeros((3, 4))
    mask[1, 2] = 1

    centers = ni.sample_labeled_patch_centers(mask, 1, (1, 1), 1, 0)
    nt.assert_allclose(centers, [[1, 2, 0]])


def test_sample_pn_patches():
    np.random.seed(0)
    mask = np.zeros((3, 4), dtype='uint8')
    img = np.reshape(np.arange(12, dtype='uint8'), (3, 4))
    mask[1, 2] = 255
    results = list(ni.sample_pn_patches(img, mask, (2, 2), 1, 1))
    assert len(results) == 2

    img_patch, mask_patch, label = results[0]
    assert label == 0
    nt.assert_allclose(img_patch, [[0, 1], [4, 5]])
    nt.assert_allclose(mask_patch, [[0, 0], [0, 0]])

    img_patch, mask_patch, label = results[1]
    assert label == 1
    nt.assert_allclose(img_patch, [[1, 2], [5, 6]])
    nt.assert_allclose(mask_patch, [[0, 0], [0, 255]])


def test_polyline2coords():
    rr, cc = ni.polyline2coords([(0, 0), (2.1, 2.0), (2, 4)])
    nt.assert_allclose(rr, [0, 1, 2, 2, 3, 4])
    nt.assert_allclose(cc, [0, 1, 2, 2, 2, 2])


def test_annotation2coords():
    img = np.zeros((5, 5), dtype='uint8')

    assert list(ni.annotation2coords(img, np.NaN)) == []
    assert list(ni.annotation2coords(img, ())) == []

    anno = ('point', ((1, 1), (1, 2)))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_allclose(rr, [1])
    nt.assert_allclose(cc, [1])
    rr, cc = list(ni.annotation2coords(img, anno))[1]
    nt.assert_allclose(rr, [2])
    nt.assert_allclose(cc, [1])

    anno = ('circle', ((2, 2, 2),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_allclose(rr, [1, 1, 1, 2, 2, 2, 3, 3, 3])
    nt.assert_allclose(cc, [1, 2, 3, 1, 2, 3, 1, 2, 3])

    anno = ('ellipse', ((2, 2, 2, 2, 0),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_allclose(rr, [1, 1, 1, 2, 2, 2, 3, 3, 3])
    nt.assert_allclose(cc, [1, 2, 3, 1, 2, 3, 1, 2, 3])

    anno = ('ellipse', ((2, 2, 1, 3, 1.5),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_allclose(rr, [1, 1, 2, 2, 2, 2, 2, 3, 3])
    nt.assert_allclose(cc, [1, 2, 0, 1, 2, 3, 4, 2, 3])

    anno = ('rect', ((1, 2, 2, 3),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_allclose(rr, [2, 2, 3, 3, 4, 4])
    nt.assert_allclose(cc, [1, 2, 1, 2, 1, 2])

    anno = ('polyline', (((1, 2), (3, 2), (3, 4), (1, 4), (1, 2)),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    print('sys.version_info', sys.version_info)
    print('scikit-image.__version', ski.__version__)
    if ski.__version__ == '0.17.2':
        # skd.polyline behaves incorrectly for version 0.17.2 :(
        nt.assert_allclose(rr, [2, 2, 3, 3])
        nt.assert_allclose(cc, [1, 2, 1, 2])
    else:
        nt.assert_allclose(rr, [2, 2, 2, 3, 3, 3, 4, 4, 4])
        nt.assert_allclose(cc, [1, 2, 3, 1, 2, 3, 1, 2, 3])

    anno = ('polyline', (((0, 0), (2, 2), (2, 4)),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_allclose(rr, [0, 1, 2, 2, 3, 4])
    nt.assert_allclose(cc, [0, 1, 2, 2, 2, 2])

    with pytest.raises(ValueError) as ex:
        anno = ('point', ((10, 10),))
        rr, cc = list(ni.annotation2coords(img, anno))[0]
    assert str(ex.value).startswith('Annotation point:(10, 10) outside image')

    with pytest.raises(ValueError) as ex:
        anno = ('invalid', ((1,),))
        coords = list(ni.annotation2coords(img, anno))
    assert str(ex.value).startswith('Invalid kind of annotation')


def test_annotation2pltpatch():
    assert list(ni.annotation2pltpatch(np.NaN)) == []
    assert list(ni.annotation2pltpatch(())) == []

    anno = ('point', ((1, 1), (1, 2)))
    pltpatches = list(ni.annotation2pltpatch(anno))
    assert len(pltpatches) == 2
    assert isinstance(pltpatches[0], plp.CirclePolygon)
    assert isinstance(pltpatches[1], plp.CirclePolygon)

    anno = ('circle', ((2, 2, 2),))
    pltpatches = list(ni.annotation2pltpatch(anno))
    assert isinstance(pltpatches[0], plp.Circle)

    anno = ('rect', ((1, 2, 2, 3),))
    pltpatches = list(ni.annotation2pltpatch(anno))
    assert isinstance(pltpatches[0], plp.Rectangle)

    anno = ('polyline', (((1, 2), (3, 2), (3, 4), (1, 4), (1, 2)),))
    pltpatches = list(ni.annotation2pltpatch(anno))
    assert isinstance(pltpatches[0], plp.Polygon)

    anno = ('polyline', (((0, 0), (2, 2), (2, 4)),))
    pltpatches = list(ni.annotation2pltpatch(anno))
    assert isinstance(pltpatches[0], plp.Polygon)

    with pytest.raises(ValueError) as ex:
        anno = ('invalid', ((1,),))
        list(ni.annotation2pltpatch(anno))
    assert str(ex.value).startswith('Invalid kind of annotation')


def test_annotation2mask():
    img = np.zeros((3, 3), dtype='uint8')
    anno = ('point', ((0, 1), (2, 0)))
    mask = ni.annotation2mask(img, anno)
    expected = np.array([[0, 0, 255], [255, 0, 0], [0, 0, 0]], dtype='uint8')
    nt.assert_allclose(mask, expected)


# Note: For some reason this test causes the unrelated test_annotation2coords
# to fail when executed before. This is related to ske.equalize_adapthist but
# how is unknown.
@pytest.mark.skip(reason="Temporarily disabled :(")
@pytest.mark.filterwarnings('ignore:Possible precision loss')
def test_normalize_histo(datadirs):
    imagedir, _, arraydir, processeddir = datadirs
    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    new_img = ni.normalize_histo(img_arr)
    imagepath = processeddir + 'nut_color_normalize_histo.bmp'
    assert_equal_image(imagepath, new_img)
