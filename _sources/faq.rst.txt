FAQ
===

What is the default image representation in nuts-ml?
----------------------------------------------------

The standard formats for image data in **nuts-ml** are Numpy arrays
of shape ``(h,w,3)`` for RGB images, ``(h,w)`` for gray-scale images
and ``(h,w,4)`` for RGBA image.



What image formats can nuts-ml read?
------------------------------------

The ``ReadImage`` nut can read images in the following formats
GIF, PNG, JPG, BMP, TIF, NPY, where NPY are plain Numpy arrays.


How to flatten a batch of predictions?
--------------------------------------

Assuming the output of a network prediction is a batch
of labels, how can it be flattened into a flow of labels?

.. code:: Python

  >>> from nutsflow import Collect, Flatten
  >>> batched_labels = [(0, 1, 0), (1, 1, 0)]   # e.g. from network.predict()
  >>> batched_labels >> Flatten() >> Collect()
  [0, 1, 0, 1, 1, 0]

What if the batch has multiple columns, e.g. labels and
probabilities

.. code:: Python

  >>> from nutsflow import Collect, FlattenCol
  >>> batched_preds = [((0,1,0), (0.1,0.2,0.3)), ((1,1,0), (0.4,0.5,0.6))]
  >>> batched_preds >> FlattenCol((0,1)) >> Collect()
  [(0, 0.1), (1, 0.2), (0, 0.3), (1, 0.4), (1, 0.5), (0, 0.6)]
