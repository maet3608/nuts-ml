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

.. doctest::

  >>> from nutsflow import Collect, Flatten
  >>> batched_labels = [(0, 1, 0), (1, 1, 0)]   # e.g. from network.predict()
  >>> batched_labels >> Flatten() >> Collect()
  [0, 1, 0, 1, 1, 0]

What if the batch has multiple columns, e.g. labels and probabilities

.. doctest::

  >>> from nutsflow import Collect, FlattenCol
  >>> batched_preds = [((0,1,0), (0.1,0.2,0.3)), ((1,1,0), (0.4,0.5,0.6))]
  >>> batched_preds >> FlattenCol((0,1)) >> Collect()
  [(0, 0.1), (1, 0.2), (0, 0.3), (1, 0.4), (1, 0.5), (0, 0.6)]

  
Error: Only length-1 arrays can be converted to Python scalars
-------------------------------------------------------

If you see the following error message when running
``network.evaluate()`` under Keras you need to upgrade to Keras 2.x
and the latests ``nuts-ml`` version.

.. code::
  
  in compute_metric    
  return float(result.eval() if hasattr(result, 'eval') else result)
  TypeError: only length-1 arrays can be converted to Python scalars  
  
  
ImportError: No module named Tkinter
-------------------------------------------------------  

This means the computer nuts-ml is running on is not supporting
the default graphical backend for matplotlib. In this case create a file 
``~/.config/matplotlib/matplotlibrc`` with the following content:

.. code::
  
  backend : Agg

  
Alternatively add the following lines to your code:

.. code::

  import matplotlib
  matplotlib.use('Agg')
    
  
  
How to use class weights for imbalanced classes in Keras
--------------------------------------------------------

.. code:: Python

  class_weights = {0:1, 1:50} 

  for epoch in xrange(EPOCHS):              
    t_loss = samples >> build_batch >> network.train(class_weights) >> Mean()
    
If the samples are an iterable (and not an iterator that is consumed) the
class weights can also be computed directly. For instance, assuming that
the class labels are in the second column (index = 1) of the sample,
the following code can be used

.. code:: Python

  class_weight = samples >> Get(1) >> CountValues()
    