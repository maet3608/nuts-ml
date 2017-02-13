"""
.. module:: test_transformer
   :synopsis: Unit tests for transformer module
"""

import pytest
import os

import numpy as np
import numpy.testing as nt

from nutsflow import Collect
from nutsml.transformer import (map_transform, TransformImage, AugmentImage,
                                RegularImagePatches, RandomImagePatches,
                                ImagePatchesByMask, ImageAnnotationToMask,
                                ImagePatchesByAnnotation, ImageMean,
                                ImageChannelMean)


def test_map_transform():
    def fake_transformation(e, *a, **kw):
        return e, a, kw

    TransformImage.register('fake_trans', fake_transformation)
    sample = (1, 2, 3)

    transspec = ('fake_trans')
    assert map_transform(sample, 0, transspec) == ((1, (), {}), 2, 3)

    transspec = (('fake_trans', [2], {'o': 2}))
    assert map_transform(sample, 1, transspec) == (1, (2, (2,), {'o': 2}), 3)


def test_TransformImage():
    TransformImage.register('fake_trans1', lambda e: e + 1)
    TransformImage.register('fake_trans2', lambda e, x: e + x)
    samples = [(1, 2), (3, 4)]

    transform = TransformImage(0).by('fake_trans1')
    assert samples >> transform >> Collect() == [(2, 2), (4, 4)]

    transform = TransformImage((0, 1)).by('fake_trans1').by('fake_trans2', 3)
    assert samples >> transform >> Collect() == [(5, 6), (7, 8)]


def test_AugmentImage():
    TransformImage.register('fake_trans1', lambda e: e + 1)
    TransformImage.register('fake_trans2', lambda e: e + 2)
    samples = [(1, 2), (3, 4)]

    augment = AugmentImage(0).by('fake_trans1', 1.0).by('fake_trans2', 1.0)
    assert samples >> augment >> Collect() == [(2, 2), (3, 2), (4, 4), (5, 4)]

    augment = AugmentImage(0).by('fake_trans1', 1.0).by('fake_trans2', 0.0)
    assert samples >> augment >> Collect() == [(2, 2), (4, 4)]


def test_RegularImagePatches():
    img1 = np.reshape(np.arange(12), (3, 4))

    samples = [(img1, 0)]
    get_patches = RegularImagePatches(0, (2, 2), 2)
    expected = [(np.array([[0, 1], [4, 5]]), 0),
                (np.array([[2, 3], [6, 7]]), 0)]
    patches = samples >> get_patches >> Collect()
    for (p, ps), (e, es) in zip(patches, expected):
        nt.assert_array_equal(p, e)
        assert ps == es

    samples = [(img1, img1 + 1)]
    get_patches = RegularImagePatches((0, 1), (1, 1), 3)
    expected = [(np.array([[0]]), np.array([[1]])),
                (np.array([[3]]), np.array([[4]]))]
    patches = samples >> get_patches >> Collect()
    for p, e in zip(patches, expected):
        nt.assert_array_equal(p, e)


def test_RandomImagePatches():
    img = np.reshape(np.arange(30), (5, 6))
    samples = [(img, 0)]
    np.random.seed(0)
    get_patches = RandomImagePatches(0, (3, 2), 2)
    patches = samples >> get_patches >> Collect()
    assert len(patches) == 2

    p, l = patches[0]
    img_patch0 = np.array([[7, 8], [13, 14], [19, 20]])
    nt.assert_array_equal(p, img_patch0)

    p, l = patches[1]
    img_patch1 = np.array([[8, 9], [14, 15], [20, 21]])
    nt.assert_array_equal(p, img_patch1)


def test_ImagePatchesByMask():
    img = np.reshape(np.arange(25), (5, 5))
    mask = np.eye(5, dtype='uint8') * 255
    samples = [(img, mask)]

    np.random.seed(0)
    get_patches = ImagePatchesByMask(0, 1, (3, 3), 1, 1, retlabel=False)
    patches = samples >> get_patches >> Collect()
    assert len(patches) == 2

    p, m = patches[0]
    img_patch0 = np.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    mask_patch0 = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    nt.assert_array_equal(p, img_patch0)
    nt.assert_array_equal(m, mask_patch0)

    p, m = patches[1]
    img_patch1 = np.array([[10, 11, 12], [15, 16, 17], [20, 21, 22]])
    mask_patch1 = np.array([[0, 0, 255], [0, 0, 0], [0, 0, 0]])
    nt.assert_array_equal(p, img_patch1)
    nt.assert_array_equal(m, mask_patch1)

    with pytest.raises(ValueError) as ex:
        mask = np.eye(3, dtype='uint8') * 255
        samples = [(img, mask)]
        get_patches = ImagePatchesByMask(0, 1, (3, 3), 1, 1)
        samples >> get_patches >> Collect()
    assert str(ex.value).startswith('Image and mask size don''t match!')


def test_ImagePatchesByMask_3channel():
    img = np.reshape(np.arange(25 * 3), (5, 5, 3))
    mask = np.eye(5, dtype='uint8') * 255
    samples = [(img, mask)]

    np.random.seed(0)
    get_patches = ImagePatchesByMask(0, 1, (3, 3), 1, 1, retlabel=False)
    patches = samples >> get_patches >> Collect()
    assert len(patches) == 2

    p, m = patches[0]
    img_patch0 = np.array([[[36, 37, 38], [39, 40, 41], [42, 43, 44]],
                           [[51, 52, 53], [54, 55, 56], [57, 58, 59]],
                           [[66, 67, 68], [69, 70, 71], [72, 73, 74]]])
    mask_patch0 = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    nt.assert_array_equal(p, img_patch0)
    nt.assert_array_equal(m, mask_patch0)


def test_ImagePatchesByAnnotation():
    img = np.reshape(np.arange(25), (5, 5))
    anno = ('point', ((3, 2), (2, 3),))
    samples = [(img, anno)]

    np.random.seed(0)
    get_patches = ImagePatchesByAnnotation(0, 1, (3, 3), 1, 1)
    patches = samples >> get_patches >> Collect()
    assert len(patches) == 3
    p, l = patches[0]
    img_patch0 = np.array([[12, 13, 14], [17, 18, 19], [22, 23, 24]])
    assert l == 0
    nt.assert_array_equal(p, img_patch0)

    p, l = patches[1]
    img_patch1 = np.array([[11, 12, 13], [16, 17, 18], [21, 22, 23]])
    assert l == 1
    nt.assert_array_equal(p, img_patch1)

    p, l = patches[2]
    img_patch1 = np.array([[7, 8, 9], [12, 13, 14], [17, 18, 19]])
    assert l == 1
    nt.assert_array_equal(p, img_patch1)


def test_ImageAnnotationToMask():
    img = np.zeros((3, 3), dtype='uint8')
    anno = ('point', ((0, 1), (2, 0)))
    samples = [(img, anno)]
    masks = samples >> ImageAnnotationToMask(0, 1) >> Collect()
    expected = np.array([[0, 0, 255], [255, 0, 0], [0, 0, 0]], dtype='uint8')
    assert str(masks[0][1]) == str(expected)  # nt.assert_array_equal fails!


def test_ImageMean():
    meansfile = 'tests/data/temp_image_mean.npy'
    img1 = np.eye(3, dtype='uint8') * 2
    img2 = np.ones((3, 3), dtype='uint8') * 4
    samples = [(img1,), (img2,)]
    img_mean = ImageMean(0, meansfile)

    # compute means
    results = samples >> img_mean.train() >> Collect()
    expected = np.array([[3., 2., 2.], [2., 3., 2.], [2., 2., 3.]])
    nt.assert_array_equal(img_mean.means, expected)
    assert results == samples
    assert os.path.exists(meansfile)

    # re-loading means from file
    img_mean = ImageMean(0, meansfile)
    nt.assert_array_equal(img_mean.means, expected)

    # subtract means
    results = samples >> img_mean >> Collect()
    expected0 = np.array([[-1., -2., -2.], [-2., -1., -2.], [-2., -2., -1.]])
    expected1 = np.array([[1., 2., 2.], [2., 1., 2.], [2., 2., 1.]])
    nt.assert_array_equal(results[0][0], expected0)
    nt.assert_array_equal(results[1][0], expected1)

    with pytest.raises(ValueError) as ex:
        other_samples = [(np.eye(5),)]
        img_mean = ImageMean(0, meansfile)
        other_samples >> img_mean >> Collect()
    assert str(ex.value).startswith('Mean loaded was computed on different')

    with pytest.raises(ValueError) as ex:
        img_mean = ImageMean(0, 'file does not exist')
        samples >> img_mean >> Collect()
    assert str(ex.value).startswith('Mean has not yet been computed!')

    os.remove(meansfile)


def test_ImageChannelMean():
    meansfile = 'tests/data/temp_image_channel_mean.npy'
    img1 = np.dstack([np.ones((3, 3)), np.ones((3, 3))])
    img2 = np.dstack([np.ones((3, 3)), np.ones((3, 3)) * 3])
    samples = [(img1,), (img2,)]

    # provide means directly
    img_mean = ImageChannelMean(0, means=[1, 2])
    expected = np.array([[[1., 2.]]])
    nt.assert_array_equal(img_mean.means, expected)

    # compute means
    img_mean = ImageChannelMean(0, filepath=meansfile)
    results = samples >> img_mean.train() >> Collect()
    expected = np.array([[[1., 2.]]])
    nt.assert_array_equal(img_mean.means, expected)
    assert results == samples
    assert os.path.exists(meansfile)

    # re-loading means from file
    img_mean = ImageChannelMean(0, filepath=meansfile)
    nt.assert_array_equal(img_mean.means, expected)

    # subtract means
    results = samples >> img_mean >> Collect()
    expected0 = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    expected1 = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    nt.assert_array_equal(results[1][0][:, :, 0], expected0)
    nt.assert_array_equal(results[1][0][:, :, 1], expected1)

    with pytest.raises(ValueError) as ex:
        other_samples = [(np.eye(5),)]
        img_mean = ImageChannelMean(0, filepath=meansfile)
        other_samples >> img_mean >> Collect()
    assert str(ex.value).startswith('Mean loaded was computed on different')

    with pytest.raises(ValueError) as ex:
        img_mean = ImageChannelMean(0, 'file does not exist')
        samples >> img_mean >> Collect()
    assert str(ex.value).startswith('Mean has not yet been computed!')

    os.remove(meansfile)
