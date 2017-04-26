.. Nutsml documentation master file

Welcome to nuts-ml
==================

.. image:: pics/nutsml_logo.gif
   :align: center

**nuts-ml** is a data pre-processing library for GPU based deep learning
that provides common pre-processing functions as independent, reusable units. 
These so called 'nuts' can be freely arranged to build data flows that 
are efficient, easy to read and modify.
The following example gives a taste of a **nuts-ml** data-flow that
trains a network on image data and prints training loss and accuracy

.. code:: python

   (train_samples >> Stratify(1) >> read_image >> transform >> augment >> 
      Shuffle(100) >> build_batch >> network.train() >>  
      Print('train loss:{} acc:{}') >> Consume())

For a quick start have a look at the :ref:`Introduction` and for
more detailed information see the :ref:`Tutorial` .


.. toctree::
   :maxdepth: 1

   introduction
   installation   
   tutorial/introduction
   faq
   contributions
   nutsml


Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

