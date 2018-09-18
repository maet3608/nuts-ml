.. _reader:

Loading images
==============

Typically image data sets are too large to be loaded into memory entirely and 
need to be processed lazily. The common strategy is to use samples that only 
contain file paths (and labels) and load images on demand.

The `ReadImage()
<https://maet3608.github.io/nuts-ml/nutsml.html#module-nutsml.reader>`_ nut of **nuts-ml**
reads images from a given path and returns Numpy arrays. Here a simple example

  >>> imagepath = 'tests/data/img_formats/*'
  >>> samples = [('nut_color.jpg', 'color'), ('nut_grayscale.jpg', 'gray')]
  >>> samples >> ReadImage(0, imagepath) >> PrintColType() >> Consume()
  item 0: <tuple>
    0: <ndarray> shape:213x320x3 dtype:uint8 range:0..248
    1: <str> color
  item 1: <tuple>
    0: <ndarray> shape:213x320 dtype:uint8 range:18..235
    1: <str> gray
    
where samples are composed of the image filename and a (class) label (``color``, ``gray``). 
``ReadImage(0, imagepath)`` takes a sample, extracts the image filename from column 0 
of the sample, constructs the full file path by replacing ``*`` in  ``imagepath`` by
the image name, loads the image and replaces the image name by the actual image data.
    
Color images are loaded as Numpy array with shape (H,W,3) and gray scale 
images with shape (H,W) both with data type ``uint8``. In this example the images are of 
size 213x320(x3) and ``range`` list the smallest and largest value of the image.

``PrintColType()`` in the above pipeline prints the data types of sample columns. In
this case, the first sample (``item 0``) is a tuple and contains a Numpy array (the image) 
in column 0 and the class label as a string in column 1.
Similarly, the second sample (``item 1``) contains the loaded image as a Numpy array
and the class label. 


TODO
- ReadLabelDirs
- ViewImage
- ViewImageAnnotation



