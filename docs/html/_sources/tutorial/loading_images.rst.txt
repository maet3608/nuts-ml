.. _reader:

Loading images
==============

Many image data sets are too large to be loaded into memory entirely and 
need to be processed lazily. This is the main reason for using **nuts-ml**.
Otherwise, pre-processing data sets via `scikit-learn <http://scikit-learn.org>`_
or `NumPy <http://www.numpy.org/>`_ is more efficient and simpler.

The common strategy in **nuts-ml** to deal with large image data is to use samples 
that contain image file paths and meta-data such as labels, and load images on demand. 
For instance, the `ReadImage()
<https://maet3608.github.io/nuts-ml/nutsml.html#module-nutsml.reader>`_ nut
reads images from a given path and returns Numpy arrays. Here a simple example
  
.. doctest::  

  >>> samples = [('nut_color.jpg', 'color'), ('nut_grayscale.jpg', 'gray')]
    
.. doctest::
  
  >>> imagepath = 'tests/data/img_formats/*'
  >>> samples >> ReadImage(0, imagepath) >> PrintColType() >> Consume()  
  item 0: <tuple>
    0: <ndarray> shape:213x320x3 dtype:uint8 range:0..248
    1: <str> color
  item 1: <tuple>
    0: <ndarray> shape:213x320 dtype:uint8 range:18..235
    1: <str> gray
    
where samples are composed of the image filename and a (class) label ('color', 'gray'). 
``ReadImage(0, imagepath)`` takes a sample, extracts the image filename from column 0 
of the sample, constructs the full file path by replacing ``*`` in  ``imagepath`` by
the image name, loads the image and replaces the image name by the actual image data.
    
Color images are loaded as Numpy arrays with shape ``(H, W, 3)`` and gray scale 
images with shape ``(H, W)`` both with data type ``uint8``. In this example 
the images are of size 213x320(x3) and ``range`` list the smallest and 
largest value of the image -- as printed by ``PrintColType()``.

``PrintColType()`` displays the data types of sample columns. Here, the first sample
(``item 0``) is a tuple and contains a Numpy array (the loaded image) in column 0 
and the class label as a string in column 1.
Similarly, the second sample (``item 1``) contains the loaded image as a Numpy array
and the class label. 

If the image filepaths are directly provided - and not stored in a column of a tuple -,
setting the column to ``None`` in ``ReadImage()`` enables loading those images:


.. doctest::

  >>> imagenames = ['nut_color', 'nut_grayscale']
  >>> imagepath = 'tests/data/img_formats/*.jpg'
  >>> imagenames >> ReadImage(None, imagepath) >> PrintColType() >> Consume()
  item 0: <tuple>
    0: <ndarray> shape:213x320x3 dtype:uint8 range:0..248
  item 1: <tuple>
    0: <ndarray> shape:213x320 dtype:uint8 range:18..235  
    
Note that ``ReadImage()`` still returns tuples, though, with the image as 
element in column 0 ``(<image>,)`` and not just the images. 

Instead of providing a base image path and image file names within samples it is
also possible to directly provide the full file path within the sample:

.. doctest::

   >>> images = ['tests/data/img_formats/nut_color.gif']
   >>> images >> ReadImage(None) >> PrintColType() >> Consume()
   item 0: <tuple>
     0: <ndarray> shape:213x320x3 dtype:uint8 range:0..255
     
Furthermore ``ReadImage()`` allows to read multiple images at the same time,
e.g. for samples that contain an image and a mask. In the following example we
read color images and their gray-scale version as pairs

.. doctest::

  >>> samples = [('color.jpg', 'grayscale.jpg'), ('color.png', 'grayscale.png')]
  >>> imagepath = 'tests/data/img_formats/nut_*'
  >>> samples >> ReadImage((0,1), imagepath) >> PrintType() >> Consume()  
  (<ndarray> 213x320x3:uint8, <ndarray> 213x320:uint8)
  (<ndarray> 213x320x3:uint8, <ndarray> 213x320:uint8) 
     
where ``ReadImage((0,1), ...)`` specifies that the image names are in columns
0 and 1 of the samples. Note that we moved the common prefix ``nut_`` of the
image file names to the image path.
     
The printout of ``PrintType`` confirms that we loaded
color images with 3 color channels (shape:213x320x3) and gray-scale images
that have no channel axis (shape:213x320) as pairs in tuple format.
     
The next section will show how to display loaded images conveniently.     
     



