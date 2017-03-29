Introduction
============

Software for GPU-based machine learning, specifically on image data,
has a common structure depicted by the following *canonical pipeline*

.. image:: pics/pipeline.png

Since large (image) data sets often cannot be loaded into memory, data is 
instead read in small batches or single images that are processed 
by a pipeline of components

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

Depending on the actual task (training, testing, evaluation, ...) or data type
(image, video, text) some of the processing steps may differ but often 
many components can be shared between applications. 

**nuts-ml** is a library that provides common data-processing and machine learning 
operations as encapsulated units, so called ‘nuts’. 
**nuts-ml** is based on `nuts-flow <https://maet3608.github.io/nuts-flow/>`_,
which itself is based on Python iterators and 
`itertools <https://docs.python.org/2/library/itertools.html>`_

.. image:: pics/architecture.png
   :align: center

**nuts-flow** wraps iterators and itertool functions into *nuts* that provide a 
``>>`` operator to arrange compositions of iterators as pipelines. For instance,
a nested itertool expression such as the following

.. code:: Python

  list(islice(ifilter(lambda x: x > 5, xrange(10)), 3))

can be flattened and more clearly written with **nuts-flow** as

.. code:: Python

  Range(10) >> Filter(_ > 5) >> Take(3) >> Collect()

**nuts-ml** adds nuts specifically for machine learning and (image) data 
processing. The following example gives a taste of a **nuts-ml** pipeline:

.. code:: python

  train_samples >> PrintProgress(train_samples) >>
    load_image >> transform >> augment >> Shuffle(100) >>
    build_batch >> network.train() >> Consume()

Nuts can be freely arranged to build data flows that are efficient, 
easy to understand and easy to modify.