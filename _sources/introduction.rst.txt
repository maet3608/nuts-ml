Introduction
============

Typical deep-learning code is characterized by

- data pre-processing on CPU and training on GPU
- mix of common and task-specific pre-processing steps
- training in epochs
- mini-batches of training data
- data augmentation to increase amount of training data
- check-pointing of network weights during training
- logging of training progress

These functions can be implemented as generic components and 
arranged in data processing pipelines.


Canonical pipeline
------------------

The *canonical pipeline* for deep-learning, specifically for image data,
is depicted below

.. image:: pics/pipeline.png

Data is processed in small batches or single images by a sequence of 
components such as

- *Reader*: sample data stored in CSV files, `Pandas <http://pandas.pydata.org/>`_ 
  tables, databases or other data sources is read,

- *Splitter*: samples are split into training, validation and sets, and stratified
  if necessary,

- *Loader*: image data is loaded for each sample when needed,

- *Transformer*: images are transformed, e.g. cropped or resized,

- *Augmenter*: images are augmented to increase data size by random rotations,
  flipping, changes to contrast, or others,

- *Batcher*: the transformed and augmented images are organized in mini-batches 
  for GPU processing,

- *Network*: a neural network is trained and evaluated on the GPU,

- *Logger*: the network performance (loss, accuracy, ...) is logged or plotted.

Depending on the specific application, the mode (training, testing, evaluation, ...) 
or data type (image, video, text) some of the processing steps will differ but 
many components can be shared between applications. 


Library
-------

**nuts-ml** is a library that provides common data-processing and machine learning 
components as so called ‘nuts’. **nuts-ml** is based on 
`nuts-flow <https://maet3608.github.io/nuts-flow/>`_, which itself is based on 
Python iterators and the `itertools <https://docs.python.org/2/library/itertools.html>`_
library.

.. image:: pics/architecture.png
   :align: center

**nuts-flow** wraps iterators and itertool functions into *nuts* that provide a 
``>>`` operator, which enables the composition of iterators in pipelines. 
For instance, a nested itertool expression such as the following

.. code:: Python

  >>> list(islice(ifilter(lambda x: x > 5, xrange(10)), 3))  # doctest: +SKIP
  [6, 7, 8]

can be flattened and more clearly written with **nuts-flow** as

.. code:: Python

  >>> Range(10) >> Filter(_ > 5) >> Take(3) >> Collect()  # doctest: +SKIP
  [6, 7, 8]

Nuts can be freely arranged to build data flows that are efficient, 
easy to understand and easy to modify.
**nuts-ml** adds nuts specifically for machine learning and (image) data 
processing. This excerpt shows the core of a **nuts-ml** pipeline

.. code:: python

  train_samples >> load_image >> transform >> augment >> Shuffle(100)
                >> build_batch >> network.train() >> Consume()

The following extended example demonstrates a network training with **nuts-ml**.


Example
-------

.. code:: Python

  def train(train_samples, val_samples):
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

      network = create_network()

      for epoch in xrange(NUM_EPOCHS):
          t_loss, t_acc = (train_samples >> PrintProgress(train_samples) >>
                           augment >> rerange >> Shuffle(100) >>
                           build_batch >> network.train() >> Unzip())
          print("training loss  :\t\t{:.6f}".format(np.mean(t_loss)))
          print("training acc   :\t\t{:.1f}".format(100 * np.mean(t_acc)))

          e_acc = (val_samples >> rerange >> build_batch >>
                   network.evaluate([categorical_accuracy]))
          print("evaluation acc  :\t\t{:.1f}".format(100 * e_acc))

          network.save_best(e_acc, isloss=False)
      print('finished.')


The complete code and more examples can be found under
`nutsml/examples <https://github.com/maet3608/nuts-ml/blob/master/nutsml/examples>`_ .
See the tutorial section for a detailed explanation of the code.
