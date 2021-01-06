.. _image-transformation:

Transforming images
===================

Images are rarely in the shape, format or condition suitable for training
and need to be transformed in some fashion. **nuts-ml** provides a wide
and easily extensible range of transformation functions.

In the following example we resize two input images, read from disk, 
to width 64 and height 128, using ``TransformImage``:

.. doctest::

  >>> imagenames = ['color.jpg', 'grayscale.jpg']  
  >>> read_image = ReadImage(None, 'tests/data/img_formats/nut_*')
  >>> resize = TransformImage(0).by('resize', 64, 128)
  >>> imagenames >> read_image >> resize >> PrintColType() >> Consume()
  item 0: <tuple>
    0: <ndarray> shape:128x64x3 dtype:uint8 range:0..242
  item 1: <tuple>
    0: <ndarray> shape:128x64 dtype:uint8 range:23..235

``TransformImage`` extracts the images from column 0 of the tuples returned
by ``ReadImage`` and applies the transformation specified by ``by``. As the
output of ``PrintColType`` shows, the resulting images are indeed of the
specified shape with 128 rows, 64 columns and a channel axis in the case of
color images.

Transformation can be chained. For instance, we can easily resize the images,
adjust the contrast and convert all images to RGB format:

.. doctest::

  >>> normalize = TransformImage(0).by('resize', 64, 128).by('contrast', 1.1).by('gray2rgb')
  >>> imagenames >> read_image >> normalize >> PrintColType() >> Consume()
  item 0: <tuple>
    0: <ndarray> shape:128x64x3 dtype:uint8 range:0..250
  item 1: <tuple>
    0: <ndarray> shape:128x64x3 dtype:uint8 range:0..241

As you can see, all images now have a channel axis, have larger range 
(due to the contrast adjustment) and are of the specified dimensions.
    

See ``TransformImage`` in `transformer.py <https://github.com/maet3608/nuts-ml/blob/master/nutsml/transformer.py>`_
for a list of available transformations or run ``help(TransformImage.by)``. 
Each transformation can also be used for image augmentation (more of that later). 
Custom transformations can be added via ``register``

.. doctest::

  >>> def my_brightness(image, c): 
  >>> ... return (image * c).astype('uint8')
  >>> TransformImage.register('my_brightness', my_brightness)

  >>> normalize = TransformImage(0).by('resize', 64, 128).by('my_brightness', 0.5)
  >>> imagenames >> read_image >> normalize >> PrintColType() >> Consume()
  item 0: <tuple>
    0: <ndarray> shape:128x64x3 dtype:uint8 range:0..121
  item 1: <tuple>
    0: <ndarray> shape:128x64 dtype:uint8 range:11..117
  
.. note:: 

   In most cases image transformation expect RGB or grayscale images of 
   data type ``uint8`` -- though there are exceptions (e.g. ``rerange``). 
   When chaining transformations make sure that expected input and output 
   image formats of the transformations do match.
   
In addition, it is easy to implement custom nuts that can perform arbitrarily
complex operation. For instance, instead of using ``TransformImage`` we can
implement a custom transformation on the samples ourselves

.. code:: Python

    @nut_function
    def ChangeBrightness(sample, c):
        image, label = sample
        new_image = (image * c).astype('uint8')
        return new_image, label
        
    samples = [('nut_color.gif', 'color'), ('nut_monochrome.gif', 'mono')]  
    read_image = ReadImage(0, 'tests/data/img_formats/*')    
    samples >> read_image >> ChangeBrightness(0.5) >> PrintColType() >> Consume()  

however, in this case we also have to extract the image from sample column 0 
and return a new sample with the transformed image and the label. 
        
.. note:: 

   Style guide: names of (custom) nuts are in CamelCase to distinguish them from
   plain Python functions. Also nuts are implemented as classes, which agrees
   with the use of CamelCase.   
   
Transformations can be applied to multiple images in a sample. In the following code,
each sample contains two images (columns 0 and 1) that are resized and
converted to RGB:

  >>> samples = [('color.jpg', 'monochrome.jpg'), ('color.png', 'monochrome.png')]
  >>> read_image = ReadImage((0,1), 'tests/data/img_formats/nut_*')
  >>> normalize = TransformImage((0,1)).by('resize', 64, 128).by('gray2rgb')
  >>> samples >> read_image >> normalize >> PrintColType() >> Consume()
  item 0: <tuple>
    0: <ndarray> shape:128x64x3 dtype:uint8 range:0..242
    1: <ndarray> shape:128x64x3 dtype:uint8 range:0..255
  item 1: <tuple>
    0: <ndarray> shape:128x64x3 dtype:uint8 range:0..242
    1: <ndarray> shape:128x64x3 dtype:uint8 range:0..255  
   
``TransformImage`` converts each input image to a corresponding output image.
A common task, however, is to extend the training data set by creating multiple
output images for an input image. These so called *augmentations* are the topic
of the next section.
 
      