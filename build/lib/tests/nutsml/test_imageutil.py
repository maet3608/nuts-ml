"""
.. module:: test_imageutil
   :synopsis: Unit tests for imageutil module
"""

import pytest
import os

import os.path as op
import PIL as pil
import numpy as np
import nutsml.imageutil as ni
import numpy.testing as nt
import matplotlib.patches as plp

from glob import glob
from PIL import ImageEnhance as ie

# Set to True to create test data
CREATE_DATA = False


@pytest.fixture(scope="function")
def testdatadirs():
    imagedir = 'tests/data/img/'
    formatsdir = 'tests/data/img_formats/'
    arraysdir = 'tests/data/img_arrays/'
    processeddir = 'tests/data/img_processed/'

    return imagedir, formatsdir, arraysdir, processeddir


def test_load_image(testdatadirs):
    h, w = 213, 320
    _, formatsdir, arraydir, _ = testdatadirs
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
        nt.assert_array_equal(img, arr)


def test_save_image(testdatadirs):
    _, formatsdir, _, _ = testdatadirs
    formats = ['gif', 'png', 'jpg', 'bmp', 'tif']
    for format in formats:
        inpath = formatsdir + 'nut_color.' + format
        outpath = formatsdir + 'temp_nut_color.' + format
        image = ni.load_image(inpath)
        ni.save_image(outpath, image)
        loaded = ni.load_image(outpath)
        # Unfortunately skimage loading and saving do not guarantee that the
        # same image that is saved is loaded again! The following assertion
        # fails for .jpg under Windows 7, Anaconda. Therefore only the shape
        # of the loaded image is tested.
        # nt.assert_array_equal(image, loaded)
        assert loaded.shape == (213, 320, 3)
        os.remove(outpath)


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
    nt.assert_array_equal(rgb_arr, arr)
    assert arr.dtype == np.uint8

    gray_arr = np.ones((5, 4), dtype='uint8')
    pil_img = ni.arr_to_pil(gray_arr)
    arr = ni.pil_to_arr(pil_img)
    nt.assert_array_equal(gray_arr, arr)
    assert arr.dtype == np.uint8

    with pytest.raises(ValueError) as ex:
        rgb_arr = np.ones((5, 4, 3), dtype='uint8')
        hsv_img = pil.Image.fromarray(rgb_arr, 'HSV')
        ni.pil_to_arr(hsv_img)
    assert str(ex.value).startswith('Expect RBG or grayscale but got')


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
    nt.assert_array_equal(image, ni.identical(image))


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
    nt.assert_array_equal(expected, cropped)


def test_crop_center():
    image = np.reshape(np.arange(16, dtype='uint8'), (4, 4))
    cropped = ni.crop_center(image, 3, 2)
    expected = np.array([[4, 5, 6], [8, 9, 10]])
    nt.assert_array_equal(expected, cropped)

    with pytest.raises(ValueError) as ex:
        ni.crop_center(image, 5, 6)
    assert str(ex.value).startswith('Image too small for crop')


def test_normalize_histo(testdatadirs):
    imagedir, _, arraydir, processeddir = testdatadirs

    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    normalized = ni.normalize_histo(img_arr)

    imagepath = processeddir + 'nut_color_normalize_histo.bmp'
    if CREATE_DATA:
        ni.save_image(imagepath, normalized)
    expected = ni.load_image(imagepath)
    nt.assert_array_equal(expected, normalized)


def test_enhance():
    image = np.ones((5, 4), dtype='uint8')
    expected = np.zeros((5, 4), dtype='uint8')
    enhanced = ni.enhance(image, ie.Brightness, 0.0)
    nt.assert_array_equal(expected, enhanced)
    assert enhanced.dtype == np.uint8


def test_contrast(testdatadirs):
    imagedir, _, _, processeddir = testdatadirs

    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    enhanced = ni.change_contrast(img_arr, 0.5)
    imagepath = processeddir + 'nut_color_contrast.bmp'
    if CREATE_DATA:
        ni.save_image(imagepath, enhanced)
    expected = ni.load_image(imagepath)
    nt.assert_array_equal(expected, enhanced)


def test_brightness(testdatadirs):
    imagedir, _, _, processeddir = testdatadirs

    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    enhanced = ni.change_brightness(img_arr, 0.5)
    imagepath = processeddir + 'nut_color_brightness.bmp'
    if CREATE_DATA:
        ni.save_image(imagepath, enhanced)
    expected = ni.load_image(imagepath)
    nt.assert_array_equal(expected, enhanced)


def test_sharpness(testdatadirs):
    imagedir, _, _, processeddir = testdatadirs

    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    enhanced = ni.change_sharpness(img_arr, 2.0)
    imagepath = processeddir + 'nut_color_sharpness.bmp'
    if CREATE_DATA:
        ni.save_image(imagepath, enhanced)
    expected = ni.load_image(imagepath)
    nt.assert_array_equal(expected, enhanced)


def test_change_color(testdatadirs):
    imagedir, _, _, processeddir = testdatadirs

    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    enhanced = ni.change_color(img_arr, 2.0)
    imagepath = processeddir + 'nut_color_color.bmp'
    if CREATE_DATA:
        ni.save_image(imagepath, enhanced)
    expected = ni.load_image(imagepath)
    nt.assert_array_equal(expected, enhanced)


def test_rotate(testdatadirs):
    imagedir, _, _, processeddir = testdatadirs

    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    enhanced = ni.rotate(img_arr, 45)
    imagepath = processeddir + 'nut_color_rotated.bmp'
    if CREATE_DATA:
        ni.save_image(imagepath, enhanced)
    expected = ni.load_image(imagepath)
    nt.assert_array_equal(expected, enhanced)


def test_shear(testdatadirs):
    imagedir, _, _, processeddir = testdatadirs

    img_arr = ni.load_image(imagedir + 'nut_color.bmp')
    enhanced = ni.shear(img_arr, 0.5)
    imagepath = processeddir + 'nut_color_sheared.bmp'
    if CREATE_DATA:
        ni.save_image(imagepath, enhanced)
    expected = ni.load_image(imagepath)
    nt.assert_array_equal(expected, enhanced)


def test_mask_where():
    expected = np.array([[0, 0], [1, 1], [2, 2]], dtype='int64')
    mask = np.eye(3, dtype='uint8')
    points = ni.mask_where(mask, 1)
    nt.assert_array_equal(expected, points)


def test_mask_choice():
    np.random.seed(1)
    expected = np.array([[0, 0], [2, 2]], dtype='int64')
    mask = np.eye(3, dtype='uint8')
    points = ni.mask_choice(mask, 1, 2)
    nt.assert_array_equal(expected, points)


def test_extract_patch():
    expected = np.array([[5, 6, 7], [9, 10, 11]], dtype='uint8')
    image = np.reshape(np.arange(16, dtype='uint8'), (4, 4))
    patch = ni.extract_patch(image, (2, 3), 2, 2)
    nt.assert_array_equal(expected, patch)


def test_patch_iter():
    img = np.asfortranarray(np.reshape(np.arange(12), (3, 4)))

    patches = list(ni.patch_iter(img, (2, 2), 2))
    expected = [np.array([[0, 1], [4, 5]]),
                np.array([[2, 3], [6, 7]])]
    for p, e in zip(patches, expected):
        nt.assert_array_equal(p, e)

    patches = list(ni.patch_iter(img, (2, 2), 3))
    expected = [np.array([[0, 1], [4, 5]])]
    for p, e in zip(patches, expected):
        nt.assert_array_equal(p, e)

    patches = list(ni.patch_iter(img, (1, 3), 1))
    expected = [np.array([[0, 1, 2]]), np.array([[1, 2, 3]]),
                np.array([[4, 5, 6]]), np.array([[5, 6, 7]]),
                np.array([[8, 9, 10]]), np.array([[9, 10, 11]])]
    for p, e in zip(patches, expected):
        nt.assert_array_equal(p, e)


def test_patch_iter_3channel():
    img = np.reshape(np.arange(12 * 3), (3, 4, 3))

    patches = list(ni.patch_iter(img, (2, 2), 2))
    expected = [
        np.array([[[0, 1, 2], [3, 4, 5]], [[12, 13, 14], [15, 16, 17]]]),
        np.array([[[6, 7, 8], [9, 10, 11]], [[18, 19, 20], [21, 22, 23]]])]
    for p, e in zip(patches, expected):
        nt.assert_array_equal(p, e)


def test_centers_inside():
    centers = np.array([[1, 2], [0, 1]])
    image = np.zeros((3, 4))

    expected = np.array([[1, 2]])
    result = ni.centers_inside(centers, image, (3, 3))
    nt.assert_array_equal(expected, result)

    result = ni.centers_inside(centers, image, (4, 3))
    assert not result


def test_sample_mask():
    mask = np.zeros((3, 4))
    mask[1, 2] = 1

    centers = ni.sample_mask(mask, 1, (1, 1), 1)
    nt.assert_array_equal(centers, [[1, 2]])

    centers = ni.sample_mask(mask, 1, (1, 1), 0)
    nt.assert_array_equal(centers, np.empty((0, 2)))

    centers = ni.sample_mask(mask, 2, (1, 1), 1)
    nt.assert_array_equal(centers, np.empty((0, 2)))

    centers = ni.sample_mask(mask, 1, (4, 3), 1)
    nt.assert_array_equal(centers, np.empty((0, 2)))


def test_sample_labeled_patch_centers():
    mask = np.zeros((3, 4))
    mask[1, 2] = 1

    centers = ni.sample_labeled_patch_centers(mask, 1, (1, 1), 1, 0)
    nt.assert_array_equal(centers, [[1, 2, 0]])


def test_sample_pn_patches():
    np.random.seed(0)
    mask = np.zeros((3, 4), dtype='uint8')
    img = np.reshape(np.arange(12, dtype='uint8'), (3, 4))
    mask[1, 2] = 255
    results = list(ni.sample_pn_patches(img, mask, (2, 2), 1, 1))
    assert len(results) == 2

    img_patch, mask_patch, label = results[0]
    assert label == 0
    nt.assert_array_equal(img_patch, [[0, 1], [4, 5]])
    nt.assert_array_equal(mask_patch, [[0, 0], [0, 0]])

    img_patch, mask_patch, label = results[1]
    assert label == 1
    nt.assert_array_equal(img_patch, [[1, 2], [5, 6]])
    nt.assert_array_equal(mask_patch, [[0, 0], [0, 255]])


def test_annotation2coords():
    img = np.zeros((5, 5), dtype='uint8')

    assert list(ni.annotation2coords(img, np.NaN)) == []
    assert list(ni.annotation2coords(img, ())) == []

    anno = ('point', ((1, 1), (1, 2)))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_array_equal(rr, [1])
    nt.assert_array_equal(cc, [1])
    rr, cc = list(ni.annotation2coords(img, anno))[1]
    nt.assert_array_equal(rr, [2])
    nt.assert_array_equal(cc, [1])

    anno = ('circle', ((2, 2, 2),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_array_equal(rr, [1, 1, 1, 2, 2, 2, 3, 3, 3])
    nt.assert_array_equal(cc, [1, 2, 3, 1, 2, 3, 1, 2, 3])

    anno = ('rect', ((1, 2, 2, 3),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_array_equal(rr, [2, 2, 3, 3, 4, 4])
    nt.assert_array_equal(cc, [1, 2, 1, 2, 1, 2])

    anno = ('polyline', (((1, 2), (3, 2), (3, 4), (1, 4), (1, 2)),))
    rr, cc = list(ni.annotation2coords(img, anno))[0]
    nt.assert_array_equal(rr, [2, 2, 3, 3])
    nt.assert_array_equal(cc, [1, 2, 1, 2])

    with pytest.raises(ValueError) as ex:
        anno = ('point', ((10, 10),))
        list(ni.annotation2coords(img, anno))[0]
    assert str(ex.value).startswith('Annotation outside image!')

    with pytest.raises(ValueError) as ex:
        anno = ('invalid', ((1,),))
        list(ni.annotation2coords(img, anno))
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

    with pytest.raises(ValueError) as ex:
        anno = ('invalid', ((1,),))
        list(ni.annotation2pltpatch(anno))
    assert str(ex.value).startswith('Invalid kind of annotation')


def test_annotation2mask():
    img = np.zeros((3, 3), dtype='uint8')
    anno = ('point', ((0, 1), (2, 0)))
    mask = ni.annotation2mask(img, anno)
    expected = np.array([[0, 0, 255], [255, 0, 0], [0, 0, 0]], dtype='uint8')
    nt.assert_array_equal(mask, expected)
