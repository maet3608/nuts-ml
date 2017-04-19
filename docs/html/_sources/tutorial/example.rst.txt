.. _cifar-example:

Example
=======

Prerequisites for this tutorial are a good knowledge of Python and
`nuts-flow <https://github.com/maet3608/nuts-flow>`_. Please read the 
`nuts-flow tutorial <https://maet3608.github.io/nuts-flow/tutorial/introduction.html>`_
if you haven't. Some knowledge of `Keras <https://keras.io/>`_,
and of course deep-learning, will be helpful.


Task
----

In this example we will implement a **nuts-ml** pipeline to classify CIFAR-10
images. `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ is a classical 
benchmark problem in image recognition. Given are 10 categories (airplane, dog, ship, ...) 
and the task is to classify small images of these objects accordingly.

.. image:: pics/cifar10.png

The CIFAR-10 dataset consists of 60000 RGB images of size 32x32. There are 6000 images 
per class and the dataset is split into 50000 training images and 10000 test images.
For more details see the `Tech report  <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>`_. 

In the following we will show how to use **nuts-flow/ml** and `Keras <https://keras.io/>`_ 
to train a Convolutional Neural Network (CNN) on the CIFAR-10 data. For readability some 
code will be omitted (e.g. import statements) but the complete code and more examples 
can be found under
`nutsml/examples <https://github.com/maet3608/nuts-ml/blob/master/nutsml/examples/cifar/cnn_train.py>`_.



Network
-------

The network architecture for the CNN is a slightly modified version of the Keras
`cifar10_cnn.py <https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py>`_ 
example (Keras version 2.x) with the notable exception of the last line, 
where the model is wrapped in a ``KerasNetwork``.

.. code:: Python

  INPUT_SHAPE = (32, 32, 3)
  NUM_CLASSES = 10

  def create_network():
      model = Sequential()
      model.add(Convolution2D(32, (3, 3), padding='same',
                              input_shape=INPUT_SHAPE))
      model.add(Activation('relu'))
      model.add(Convolution2D(32, (3, 3)))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Dropout(0.5))

      model.add(Convolution2D(64, (3, 3), padding='same'))
      model.add(Activation('relu'))
      model.add(Convolution2D(64, (3, 3))
      model.add(Activation('relu'))
      model.add(MaxPooling2D(pool_size=(2, 2)))
      model.add(Dropout(0.5))

      model.add(Flatten())
      model.add(Dense(512))
      model.add(Activation('relu'))
      model.add(Dropout(0.5))
      model.add(Dense(NUM_CLASSES))
      model.add(Activation('softmax'))

      model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

      return KerasNetwork(model, 'weights_cifar10.hd5')


The wrapping allows us using the CNN as a ``nut`` within a **nuts-flow**,
which simplifies training. The wrapper also takes a path to a weights file 
for check-pointing. Weights are saved in the standard Keras format as
`HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ file.

.. note:: 

  So far only wrappers for Keras and Lasagne models are provided. However, 
  any deep-learning library that accepts an iterable over mini-batches for 
  training will work with **nuts-ml**.



Loading data
------------

In many image processing applications the complete set of training images 
is too large to fit in memory and images are loaded in a streamed fashion. 
See `read_images.py 
<https://github.com/maet3608/nuts-ml/blob/master/nutsml/examples/cifar/read_images.py>`_ 
for an example that loads images sequentially.

CIFAR-10, however, is small benchmark data set and fits in memory. We therefore 
take advantage of the function ``cifar10.load_data()`` provided by
`Keras <https://github.com/fchollet/keras/blob/master/keras/datasets/cifar10.py>`_,
and load all images in memory but rearrange the data slightly

.. code:: Python

  def load_samples():
      (x_train, y_train), (x_test, y_test) = cifar10.load_data()
      train_samples = zip(x_train, map(int, y_train))
      test_samples = zip(x_test, map(int, y_test))
      return train_samples, test_samples

Specifically, we convert class labels from floats to integers, 
and zip inputs ``x`` and outputs ``y`` to create lists with training and test samples.
Sample are then tuples of format ``(image, label)``, where the image is a 
Numpy array of shape ``(32,32,3)``, and the label is an integer between 0 and 9, 
indicating the class. We can verify the type and shape of the samples 
by running the following flow
(`complete code here <https://github.com/maet3608/nuts-ml/blob/master/nutsml/examples/cifar/view_data.py>`_ )

.. code:: Python

  train_samples, val_samples = load_samples()
  train_samples >> Take(3) >> PrintColType() >> Consume()

which takes the first three samples and prints for each sample 
the data type and content information for the sample columns

.. code:: Python  

  0: <ndarray> shape:32x32x3 dtype:uint8 range:0-255
  1: <int> 6

  0: <ndarray> shape:32x32x3 dtype:uint8 range:5-254
  1: <int> 9

  0: <ndarray> shape:32x32x3 dtype:uint8 range:20-255
  1: <int> 9


.. note::

  The standard formats for image data in **nuts-ml** are Numpy arrays
  of shape ``(h,w,3)`` for RGB images, ``(h,w)`` for gray-scale images
  and ``(h,w,4)`` for RGBA image.

Not only can we inspect the type of the data but we can also have a look
at the images themselves

.. code:: Python

  train_samples, val_samples = load_samples()
  train_samples >> Take(3) >> PrintColType() >> ViewImage(0) >> Consume()

.. image:: ../pics/viewimage_cifar10.png


Training
--------

We will introduce the code for the network training in pieces before showing 
the complete code later. First, let us create the network and load the 
sample data using the functions introduced above

.. code:: Python

  network = create_network()
  train_samples, val_samples = load_samples()

Having a network and samples we can now train the network (for one epoch) 
with the following **nuts-flow**

.. code:: Python

  train_samples >> augment >> rerange >> Shuffle(100) \
                >> build_batch >> network.train() >> Consume()

The flow *augments* the training images by random transformations,
*re-ranges* pixel values to [0,1], *shuffles* the samples, *builds*
mini-batches, *trains* the network and *consumes* outputs of the training 
(losses, accuracies).

``Consume`` and ``Shuffle`` are *nuts* from **nuts-flow**. Image augmentation, 
re-ranging and batch-building are parts of **nuts-ml** that we describe
in detail in the next sections.


Augmentation
^^^^^^^^^^^^

Deep learning requires large data sets and a common strategy to increase the
amount of image data is to augment the data set with randomly perturbed
copies, e.g. rotated or blurred. Here we want augment the CIFAR-10 data set by 
flipping images horizontally and changing the brightness

.. code:: Python

      p = 0.1
      augment = (AugmentImage(0)
                 .by('identical', 1.0)
                 .by('fliplr', p)
                 .by('brightness', p, [0.7, 1.3]))

The ``AugmentImage`` nut takes as parameter the index of the image within the 
sample ``(image, label)``, here position 0 and augmentations are specified 
by invoking ``by(transformation, probability, *args)``.

We augment by passing the unchanged image (``'identical'``) through with 
probability 1.0 (all of them), flipping images horizontally for 10% 
of the samples (``p = 0.1``),  and randomly changing the brightness 
in range ``[0.7, 1.3]``, again with 10% probability ``p``. We could have
a look at the augmented images and their labels using the following flow
(`complete code here <https://github.com/maet3608/nuts-ml/blob/master/nutsml/examples/cifar/view_augmented_images.py>`_ )

.. code:: Python

  train_samples, val_samples = load_samples()
  train_samples >> augment >> ViewImageAnnotation(0, 1, pause=1) >> Consume()

In detail: for every sample processed by ``AugmentImage``, the image is
extracted from position 0 of the sample tuple and new samples with the same label
but with augmented images are outputted. For each input image the identical 
output image is generated (``identical``), and additional augmented samples 
(``fliplr``, ``brightness``) are created with 10% probability each, resulting
in 20% more training data.


Transformation
^^^^^^^^^^^^^^

Images returned by ``load_samples()`` are Numpy arrays with integers in range 
``[0, 255]``. The network, however, expects floating point numbers (``float32``) 
in range ``[0,1]``. We therefore transform images by *reranging*

.. code:: Python

  rerange = TransformImage(0).by('rerange', 0, 255, 0, 1, 'float32')

where ``TransformImage`` takes as parameter the index of the image within 
the sample and transformation are defined by invoking ``by(transformation, *args)``. 

.. note:: 

  Transformation are chained, meaning that an input image is transformed by sequentially
  applying all transformations to the image, resulting in one output image. Consequently, 
  the number of input and output images after transformation are the same. Augmentations, 
  on the other hand, are applied independently and the number of input and output images 
  can differ.

See ``TransformImage`` in `transformer.py <https://github.com/maet3608/nuts-ml/blob/master/nutsml/transformer.py>`_
for a list of available transformations. Each transformation can also be used for
augmentation. Custom transformations can be added via ``register``

  >>> from nutsml import TransformImage, AugmentImage
  >>> my_brightness = lambda image, c: image * c
  >>> TransformImage.register('my_brightness', my_brightness)

  >>> transform = TransformImage(0).by('my_brightness', 1.5)
  >>> augment = AugmentImage(0).by('my_brightness', [0.7, 1.3])

While transformations take a specific parameter values, e.g. ``1.5`` for brightness,
augmentations take ranges, e.g. ``[0.7, 1.3]``, where parameter values are
uniformly sampled from.



Batching
^^^^^^^^

Networks are trained with *mini-batches* of samples, e.g. a stack of images
with their corresponding class labels. ``BuildBatch(batchsize)``
is used to build these batches. The following example creates a batcher that 
extracts images from column 0 of the samples and class labels from column 1. 
Class labels are encode as one-hot vectors, while images are represented as 
Numpy arrays with dtype ``float32``.

.. code:: Python
      
  NUM_CLASSES = 10
  BATCH_SIZE = 32

  build_batch = (BuildBatch(BATCH_SIZE)
                  .by(0, 'image', 'float32')
                  .by(1, 'one_hot', 'uint8', NUM_CLASSES))

Having a batcher we can now build a complete pipeline that trains the network
for one epoch

.. code:: Python

  train_samples >> augment >> rerange >> build_batch >> network.train() >> Consume()

.. note::

  ``Consume()`` or some other data sink is needed. Without a consumer at the end of the 
  pipeline no data is processed.

Usually it is a good idea to shuffle the data (especially after augmentation) to ensure 
that each mini-batch contains a nice distribution of different class examples. 
Complete shuffling is not feasible if the training images do not fit in memory 
but we can perform a partial shuffling, e.g. over 100 samples. 
Let's also train for more than one epoch

.. code:: Python

  EPOCHS = 20
  for epoch in range(EPOCHS):
      (train_samples >> augment >> rerange >> Shuffle(100) >> build_batch >> 
       network.train() >> Consume())


Training results
^^^^^^^^^^^^^^^^

Instead of consuming (and throwing away) the outputs of the training we can collect 
and print the results (loss, accuracy)

.. code:: Python

  for epoch in range(EPOCHS):
      t_loss, t_acc = (train_samples >> augment >> rerange >> Shuffle(100) >>
                       build_batch >> network.train() >> Unzip())

      print("train loss  :", t_loss >> Mean())
      print("train acc   :", t_acc >> Mean())

``network.train()`` takes mini-batches as input and outputs loss and accuracy
(per mini-batch) as specified in ``create_network()``. ``Unzip()`` transforms the 
outputted sequence of ``(loss, accuracy)`` tuples into a sequence of losses 
``t_loss`` and a sequence of accuracies ``t_acc``. 
Finally we print the mean (over mini-batches) for training loss and accuracy.


Validation
----------

The performance of the network on the validation data can be computed analogous 
to the way the training results were computed. Important differences are 
that we are using the validation data, calling ``network.validate()`` instead of
``network.train(), do not perform augmentation and there is no need to shuffle the data

.. code:: Python

  for epoch in range(EPOCHS):
      v_loss, v_acc = val_samples >> rerange >> build_batch >> network.validate() >> Unzip()
      print("val loss  :", v_loss >> Mean())
      print("val acc   :", v_acc >> Mean())

Again, results are mean values over mini-batches per epoch.


Evaluation
----------

TODO


Prediction
----------

TODO


Writing
-------

TODO



Full example
------------

Complete code is 
`here <https://github.com/maet3608/nuts-ml/blob/master/nutsml/examples/cifar/cnn_train.py>`_.


.. code:: Python

  def train(train_samples, val_samples):
      from keras.metrics import categorical_accuracy

      rerange = TransformImage(0).by('rerange', 0, 255, 0, 1, 'float32')
      build_batch = (BuildBatch(BATCH_SIZE)
                     .by(0, 'image', 'float32')
                     .by(1, 'one_hot', 'uint8', NUM_CLASSES))
      p = 0.1
      augment = (AugmentImage(0)
                 .by('identical', 1.0)
                 .by('brightness', p, [0.7, 1.3])
                 .by('color', p, [0.7, 1.3])
                 .by('shear', p, [0, 0.1])
                 .by('fliplr', p)
                 .by('rotate', p, [-10, 10]))
      plot_eval = PlotLines((0, 1), layout=(2, 1))

      print('creating network...')
      network = create_network()

      print('training...', len(train_samples), len(val_samples))
      for epoch in xrange(NUM_EPOCHS):
          print('EPOCH:', epoch)

          t_loss, t_acc = (train_samples >> PrintProgress(train_samples) >>
                           Pick(PICK) >> augment >> rerange >> Shuffle(100) >>
                           build_batch >> network.train() >> Unzip())
          print("training loss  :\t\t{:.6f}".format(t_loss >> Mean()))
          print("training acc   :\t\t{:.1f}".format(100 * (t_acc >> Mean())))

          v_loss, v_acc = (val_samples >> rerange >>
                           build_batch >> network.validate() >> Unzip())
          print("validation loss :\t\t{:.6f}".format(v_loss >> Mean()))
          print("validation acc  :\t\t{:.1f}".format(100 * (v_acc >> Mean())))

          e_acc = (val_samples >> rerange >> build_batch >>
                   network.evaluate([categorical_accuracy]))
          print("evaluation acc  :\t\t{:.1f}".format(100 * e_acc))

          network.save_best(e_acc, isloss=False)
          plot_eval((t_acc >> Mean(), e_acc))

