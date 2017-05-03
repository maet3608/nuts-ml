.. _reader:

Reading samples
===============

**nuts-ml** does not have a specific class or data structure defining data samples
buts many nuts return ``tuples`` and expect indexable data structures such as
``tuples`` or ``lists`` as input.

Here a toy example to demonstrate some basic operations before moving on to more advanced
and better methods to read sample data. Given the file ``tests/data/and.csv`` that 
contains the truth table of the logical ``and`` operation

.. code::

  x1,x2,y
  0,0,no
  0,1,no
  1,0,no
  1,1,yes

we can load its lines via Python's ``open`` function and collect them in a ``list``

  >>> from nutsflow import *

  >>> open('tests/data/and.csv') >> Collect()
  ['x1,x2,y\n', '0,0,no\n', '0,1,no\n', '1,0,no\n', '1,1,yes']

However, what is typically needed is tuples with the data and no header. 
We therefore drop the first line, split all remaining lines at ',' and remove the 
pesky newline character (``\n``) 

  >>> Split = nut_function(lambda s : s.strip().split(','))
  >>> load_data = open('tests/data/and.csv')
  >>> load_data >> Drop(1) >> Split() >> Print() >> Consume()
  ['0', '0', 'no']
  ['0', '1', 'no']
  ['1', '0', 'no']
  ['1', '1', 'yes']

Better! But all numbers in the first and second column are strings. We use
``MapCol``, which maps the ``int`` function on the specified colums ``(0, 1)`` 
and also make the loading of the data a bit more generic

  >>> Load = nut_source(lambda fname: open('tests/data/'+fname))
  >>> (Load('and.csv') >> Drop(1) >> Split() >> MapCol((0, 1), int) >>
  ... Print() >> Consume())
  (0, 0, 'no')
  (0, 1, 'no')
  (1, 0, 'no')
  (1, 1, 'yes')

There is a  `ReadCSV <https://github.com/maet3608/nuts-flow/blob/master/nutsflow/source.py>`_
nut in **nuts-flow** that could simplify or eliminate some of the steps above but
we are going to use `ReadPandas <https://github.com/maet3608/nuts-ml/blob/master/nutsml/reader.py>`_
from **nuts-ml**, which is even more powerfull 

  >>> from nutsml import *

  >>> ReadPandas('tests/data/and.csv') >> Print() >> Consume()
  (0L, 0L, 'no')
  (0L, 1L, 'no')
  (1L, 0L, 'no')
  (1L, 1L, 'yes')

It drops the header, splits the lines and converts numbers to (long) integers automatically.
But maybe the label column should be numeric as well. Here we go

  >>> label2int = MapCol(2, lambda label: 1 if label=='yes' else 0)
  >>> ReadPandas('tests/data/and.csv') >> label2int >> Print() >> Consume()
  (0L, 0L, 0)
  (0L, 1L, 0)
  (1L, 0L, 0)
  (1L, 1L, 1)


Note: loads all data into memory at once

TODO
-----
More details about ReadPandas
- reference to pandas
- kwargs
dplyr example

Reading image/audio/text samples  (filepath, label)
