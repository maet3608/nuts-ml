Splitting and stratifying
=========================

Splitting data sets into training and test sets, and ensuring a balanced distribution 
of class labels are common preprocessing tasks for machine learning.


Splitting data
--------------

We start with a toy example, and randomly split a list of numbers
ranging from 0 to 9 into a training and a testing set with a size ratio 
of 70%:

.. doctest::

  >>> train, test = range(10) >> SplitRandom(ratio=0.7)
  >>> print('\n', train, '\n', test)
  [6, 3, 1, 7, 0, 2, 4] 
  [5, 9, 8]

Note that ``SplitRandom()`` is a sink and no ``Collect()`` or ``Consume()``
is required at the end of the pipeline. ``SplitRandom()`` returns a tuple 
containing the split data sets.
Often a three-fold split into a training, validation and testing set
is needed and this is easily done as well:

.. doctest::   

  >>> train, val, test = range(10) >> SplitRandom(ratio=(0.6, 0.3, 0.1)) 
  >>> print(train, val, test)
  ([6, 1, 4, 0, 3, 2], [8, 7, 9], [5])
  
``SplitRandom()`` randomizes the order of the samples in the split but
uses the same seed for the randomization for each call. You can provide a 
random number generator to create seed-dependent splits, e.g.
   
.. doctest:: 
  
  >>> from nutsflow.common import StableRandom
  >>> rand = StableRandom(seed=0)
  >>> range(10) >> SplitRandom(ratio=0.7, rand=rand)
  [[6, 3, 1, 7, 0, 2, 4], [5, 9, 8]]

   
.. note::
   
   Python's pseudo random number generator ``random.Random(0)`` returns different
   number sequences for Python 2.x and 3.x -- with the same seed! If you need 
   repeatable results across Python versions, e.g. for unit testing, 
   use ``StableRandom()``.
   
   
Occasionally, there are constraints on how the data can be split. For instance,
in medical data sets records originating from the same patient should not be
distributed across sets, since this would bias the results. ``SplitRandom()`` 
supports a ``constraint`` function and in the following example we ensure that
numbers with the same parity are not scattered across splits:

.. doctest::

  >>> same_parity = lambda x: x % 2 == 0
  >>> range(10) >> SplitRandom(ratio=0.5, constraint=same_parity)
  [[0, 2, 6, 8, 4], [3, 1, 7, 5, 9]]
  
Note that the constraint has precedence over the ratio. For instance, for a ratio of ``0.7`` the constraint holds (even or odd numbers are not scattered over splits) but
the first split contains all samples and the second split is empty, violating
the ``0.7`` ratio of split sizes:

.. doctest::

  >>> range(10) >> SplitRandom(ratio=0.7, constraint=same_parity)
  [[0, 5, 2, 6, 4, 9, 8, 7, 3, 1], []]
  
  
Let's close with a more realistic example. We load the `Iris flower data set
<https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ and split it into 
training and testing sets:

.. doctest::

  >>> train, test = ReadPandas('tests/data/iris.csv') >> SplitRandom(ratio=0.5)
  >>> len(train), len(test)
  (74, 75)
  
As you can see, with a split ratio of ``0.5`` training and test are roughly
about the same size. 

.. note::
  
   ``SplitRandom()`` loads all samples into memory. Splitting therefore has to
   occur before large data object (e.g. images) belonging to samples are
   loaded.
  
If your data set is very small, you likely will need a leave-one-out split, 
which can be performed via ``SplitLeaveOneOut()``:

.. doctest::

  >>> samples = [1, 2, 3]
  >>> for train, test in samples >> SplitLeaveOneOut():
  ...     print(train, ' : ', test)
  [2, 3]  :  [1]
  [1, 3]  :  [2]
  [1, 2]  :  [3]
  
  

Stratifying data
----------------
   
Real world data often contains considerably different numbers of samples for
the classes to learn (class imbalance). Training a classifier on such an unbalanced
data set could introduce a classification bias. Typically the classifier is
more accurate on the class with more samples. A common method to avoid this bias,
is to stratify the data by over- or under-sampling samples based on their class
labels.

In the following example, we create an artificial sample set with 10 samples
belonging to the ``good`` class and 100 samples for the ``bad`` class. 
``CountValues()`` returns a dictionary with the sample frequencies for the
class labels:

.. doctest::

  >>> samples = [(0, 'good')] * 10 + [(1, 'bad')] * 100
  >>> labelcol = 1
  >>> labeldist = samples >> CountValues(labelcol)
  >>> print(labeldist)
  {'good': 10, 'bad': 100}
  
Obviously, this is a strongly unbalanced data set. After stratification the samples
frequencies are much more balanced: 

.. doctest::

  >>> stratified = samples >> Stratify(labelcol, labeldist) >> Collect() 
  >>> print(stratified >> CountValues(labelcol))
  {'good': 10, 'bad': 9}

``Stratify()`` requires the label distribution of the unbalanced data set 
as input and down-sampling is based on the sample frequencies in ``labeldist``.
If the label distribution is known upfront, it can provided directly and 
there is no need to call ``CountValues()``.

.. note::
   ``Stratify()`` randomly selects samples but does not change the order
   of samples. Use ``Shuffle`` to ensure random ordering, e.g.
   ``Stratify() >> Shuffle(1000) >> Collect()``.

   
Splitting and stratifying
-------------------------

In this example we combine loading, splitting and stratification of sample data.
We take only 120 of the 150 samples of the `Iris flower data set
<https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ to create an 
artificially unbalanced sample set:

.. doctest::

  >>> filepath = 'tests/data/iris.csv'
  >>> train, test = ReadPandas(filepath) >> Take(120) >> SplitRandom(ratio=0.7)  
  >>> labelcol = 4
  >>> train >> CountValues(labelcol)
  {'Iris-versicolor': 33, 'Iris-setosa': 35, 'Iris-virginica': 16}
  
Next we stratify and shuffle the training data:
 
.. doctest:: 

  >>> labeldist = train >> CountValues(labelcol)
  >>> train >> Stratify(labelcol, labeldist) >> Shuffle(100) >> CountValues(labelcol)
  {'Iris-setosa': 23, 'Iris-virginica': 16, 'Iris-versicolor': 16}
   
As you can see, the training data is now balanced again. ``Shuffle(100)`` loads 100
samples in memory and shuffles them to  perform a (partial) randomization of the
sample order. Typically we would perform stratification and shuffling in the 
training loop. Here a template example:

.. code:: Python
 
   train, val, test = ReadPandas(filepath) >> SplitRandom((0.8, 0.1, 0.1)) 
   
   for epoch in range(EPOCHS):
       accuracy = (train >> Stratify(labelcol, labeldist) >> Shuffle(100) >> 
                   build_batch >> network.train() >> Mean())
        
Note that ``SplitRandom()`` creates the same split every time it is called,
while ``Stratify()`` will down-sample randomly. This ensures rerunning a training
operates on the same training and test data but in the training loop stratification
and shuffling randomizes the order of samples. This is usually what you want but 
you can provide random number generators with specific seeds to change this 
default behavior.

For the building of batches and network training see the later sections
of the tutorial.


   
