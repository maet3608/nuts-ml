
.. image:: pics/nutsml_logo.gif
   :align: center

- `Introduction <https://maet3608.github.io/nuts-ml/introduction.html>`_
- `Installation <https://maet3608.github.io/nuts-ml/installation.html>`_
- `Tutorial <https://maet3608.github.io/nuts-ml/tutorial/introduction.html>`_
- `Documentation <https://maet3608.github.io/nuts-ml/>`_
- `Examples <https://github.com/maet3608/nuts-ml/tree/master/nutsml/examples>`_
- `Github <https://github.com/maet3608/nuts-ml>`_

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

**nuts-ml** is based on `nuts-flow <https://github.com/maet3608/nuts-flow>`_,
which is described `here <https://maet3608.github.io/nuts-flow/>`_.

.. image:: https://badge.fury.io/py/nutsml.svg
   :target: https://badge.fury.io/py/nutsml

.. image:: https://img.shields.io/pypi/pyversions/nutsml.svg
   :target: https://pypi.python.org/pypi/nutsml/

.. image:: https://travis-ci.org/maet3608/nuts-ml.svg?branch=master
   :target: https://travis-ci.org/maet3608/nuts-ml

.. image:: https://coveralls.io/repos/github/maet3608/nuts-ml/badge.png?branch=master
   :target: https://coveralls.io/github/maet3608/nuts-ml?branch=master

.. image:: https://img.shields.io/github/issues/maet3608/nuts-ml.svg
   :target: https://github.com/maet3608/nuts-ml/issues

.. image:: https://img.shields.io/badge/license-Apache%202-blue.svg
   :target: https://github.com/maet3608/nuts-ml/blob/master/LICENSE






