.. Nutsml documentation master file

Welcome to nuts-ml
==================

.. image:: pics/nutsml_logo.gif
   :align: center

**nuts-ml** is a data pre-processing library for GPU-based deep learning
that provides common pre-processing functions as independent, reusable units.
Similar to the data loading and transformation pipelines in PyTorch or
Tensorflow but framework-agnostic and more flexible.

These units are called *nuts* and can be freely arranged to build data flows 
that are efficient, easy to read and modify.
The following example gives a taste of a **nuts-ml** data-flow that
trains a network on image data and prints training loss and accuracy

.. code:: python

   (train_samples >> Stratify(1) >> read_image >> transform >> augment >> 
      Shuffle(100) >> build_batch >> network.train() >>  
      Print('train loss:{} acc:{}') >> Consume())

It is easy to extend **nuts-ml** with your own
`custom nuts <https://maet3608.github.io/nuts-flow/tutorial/custom_nuts.html>`_ .
For instance, a nut that filters out (almost) black images could be implemented
as

.. code:: python

   @nut_filter
   def NotBlack(sample, threshold):
      image, label = sample
      return sum(image) > threshold

and then can be plugged into the flow

.. code:: python

   ... >> read_image >> NotBlack(10) >> transform >> ...


For a quick start read the :ref:`Introduction` and have a look at the code 
`examples <https://github.com/maet3608/nuts-ml/tree/master/nutsml/examples>`_ .
The :ref:`Tutorial` explains some of the examples in detail and if you are not
familiar with `nuts-flow <https://github.com/maet3608/nuts-flow>`_, the library 
**nuts-ml** is based on, reading its `documentation <https://maet3608.github.io/nuts-flow/>`_
is recommended. Skim over the short description of all nuts in the :ref:`Overview` 
for an overall impression of the available functionality.


.. toctree::
   :maxdepth: 1

   introduction
   installation
   overview   
   tutorial/introduction
   faq
   contributions
   nutsml


Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

