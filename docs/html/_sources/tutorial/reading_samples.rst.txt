.. _reader:

Reading data samples
====================

**nuts-ml** does not have a specific type for data samples
but most functions operate on ``tuples``, ``lists`` or ``numpy arrays``,
with a preference for tuples. For instance, a sample of the `Iris data set <https://en.wikipedia.org/wiki/Iris_flower_data_set#Data_set>`_ could be
represented as the tuple ``(4.9, 3.1, 1.5, 0.2, 'Iris-setosa')`` and the entire
data set as a list of tuples.


Basics
------

**nuts-ml** is designed to read data in an iterative fashion to allow the processing
of arbitrarily large data sets. It largely relies on **nuts-flow**, which is
documented `here <https://maet3608.github.io/nuts-flow/>`_. In the following
a short introduction of the basic principles.
 
We start by importing **nuts-flow** and **nuts-ml** 

  >>> from nutsflow import *
  >>> from nutsml import *  
  
and create a tiny, in-memory example data set:
  
  >>> data = [(1,'odd'), (2, 'even'), (3, 'odd')]
  
Data pipelines in **nuts-ml** require a `sink <https://maet3608.github.io/nuts-flow/overview.html>`_ that pulls the data. The two most common ones are 
`Consume <https://maet3608.github.io/nuts-flow/nutsflow.html#nutsflow.sink.Consume>`_ and
`Collect <https://maet3608.github.io/nuts-flow/nutsflow.html#nutsflow.sink.Collect>`_ but
there are many `others <https://maet3608.github.io/nuts-flow/overview.html>`_.

``Consume()`` consumes all data and returns nothing, while ``Collect()`` collects all data in a list.
As an example we take the first two samples of the data set. Without a sink the pipeline is
not processing anything at all and only the generator (stemming from ``Take()``) is returned.

  >>> data >> Take(2)
  itertools.islice at 0xbf160e8>
  
Adding a ``Collect()`` results in the processing of the data and gives us what we want:

  >>> data >> Take(2) >> Collect()
  [(1, 'odd'), (2, 'even')]
  
The same pipeline using ``Consume()`` returns nothing

  >>> data >> Take(2) >> Consume()
  
but we can verify that samples are processed by inserting a 
`Print <https://maet3608.github.io/nuts-flow/nutsflow.html#nutsflow.function.Print>`_ nut:

  >>> data >> Print() >> Take(2) >> Consume()
  (1, 'odd')
  (2, 'even')

A broken pipeline or a pipeline without sink is a common problem that can be debugged
easily by inserting ``Print()`` functions. Two other very common and important functions are 
`Filter <https://maet3608.github.io/nuts-flow/nutsflow.html#nutsflow.processor.Filter>`_ and
`Map <https://maet3608.github.io/nuts-flow/nutsflow.html#nutsflow.processor.Map>`_.

As the name indicates, ``Filter`` is used to filter samples based on a provided
boolean function:

  >>> data >> Filter(lambda s: s[1] == 'odd') >> Print() >> Consume()
  (1, 'odd')
  (3, 'odd')
  
or maybe more clearly with additional printing

  >>> def is_odd(sample):
  ...     return sample[1] == 'odd'
  >>> data >> Print('before: {},{}') >> Filter(is_odd) >> Print('after : {},{}') >> Consume()
  before: 1,odd
  after : 1,odd
  before: 2,even
  before: 3,odd
  after : 3,odd
  
``Map`` applies a function to the samples of a data set, e.g.

  >>> def add_two(sample):
  ...     number, label = sample
  ...     return number + 2, label
  >>> data >> Map(add_two) >> Collect()
  [(3, 'odd'), (4, 'even'), (5, 'odd')]
    
There is a convenience nut `MapCol <https://maet3608.github.io/nuts-flow/nutsflow.html#nutsflow.processor.MapCol>`_  
that maps a function to a specific column (or columns) of a sample. This allows us
to write more succinctly
  
   >>> add_two = lambda number: number + 2
   >>> data >> MapCol(0, add_two) >> Collect()
   [(3, 'odd'), (4, 'even'), (5, 'odd')]
   
For simple expressions a Scala like syntax can be used to further shorten the code:

   >>> data >> MapCol(0, _ + 2) >> Collect()
   [(3, 'odd'), (4, 'even'), (5, 'odd')]
   
Let's combine what we learned and construct a pipeline that extracts the first number
in the data set that is even and converts the labels to upper case. 

   >>> to_upper = lambda label: label.upper()
   >>> is_even = lambda number: number % 2 == 0
   >>> first_even = (data >> FilterCol(0, is_even) >> 
   ... MapCol(1, to_upper) >> Take(1) >> Collect())
   [(2, 'EVEN')]
   
Here we used `FilterCol 
<https://maet3608.github.io/nuts-flow/nutsflow.html#nutsflow.processor.FilterCol>`_
instead of ``Filter`` to filter for the contents in column ``0`` (the numbers) of
the samples. Note that we wrap the pipeline into brackets allowing it to run over multiple lines.
Alternatively, we could refactor the code as follows to shorten the pipeline:

   >>> to_upper = MapCol(1, lambda label: label.upper())
   >>> is_even = FilterCol(0, lambda number: number % 2 == 0)
   >>> first_even = data >> is_even >> to_upper >> Head(1)
   [(2, 'EVEN')]

This concludes the basics. In the following examples we will read data sets 
in different formats from the file system and the web.     
   

TXT files
---------

Let us start with reading data from a simple text file. Here a tiny example file 
``tests/data/and.txt``

.. code::

  x1,x2,y
  0,0,no
  0,1,no
  1,0,no
  1,1,yes
  
We can loads the file content with Python's ``open`` function that returns an 
iterator over the lines and collect them in a ``list``  

  >>> from nutsflow import *

  >>> open('tests/data/and.txt') >> Collect()
  ['x1,x2,y\n', '0,0,no\n', '0,1,no\n', '1,0,no\n', '1,1,yes']
  
Of course, ``open('tests/data/and.txt').readlines()`` would have achieved the same.
However, samples as strings are not very useful. We would like samples to be
represented as tuples or lists containing column values. First, we therefore define a
nut function that strips white spaces from lines and splits a line into
its components:

>>> split = Map(lambda line : line.strip().split(','))

This as a ``Map`` because it will be applied to each line of the file. 
Let us try it out by reading the header of the file

  >>> lines = open('tests/data/and.txt')
  >>> lines >> split >> Head(1)
  [['x1', 'x2', 'y']]
  
where ``Head(n)`` is a sink that collects the first ``n`` lines in a list (here only one line).
As expected, we get the header with the column names.
Since ``open`` returns an iterator ``lines`` is ready to deliver the remaining
lines of the file. For instance, we could now write

  >>> lines >> split >> Print() >> Consume()
  ['0', '0', 'no']
  ['0', '1', 'no']
  ['1', '0', 'no']
  ['1', '1', 'yes']

which prints out the samples following the header.
Note that ``Consume`` does not collect the samples - it just consumes them and
returns nothing. Good for debugging but not suitable for further processing.
We therefore rerun the code and collect the samples in a list. But careful!
The ``lines`` iterator has been consumed. We have to reopen the file to
restart the iterator:

  >>> lines = open('tests/data/and.txt')
  >>> lines >> Drop(1) >> split >> Collect()
  [['0', '0', 'no'], ['0', '1', 'no'], ['1', '0', 'no'], ['1', '1', 'yes']]

We use ``Collect`` to collect the samples and ``Drop(1)`` means that we
skip the header line when reading the file.
  
Next we need to convert the strings containing numbers to actual numbers.
``MapCol`` can be used to map Python's ``int`` function on specific columns of the 
samples; here columns ``0`` and  ``1`` of the samples contain integers:

  >>> lines = open('tests/data/and.txt')
  >>> to_int = MapCol((0, 1), int)
  >>> skip_header = Drop(1)
  >>> samples = lines >> skip_header >> split >> to_int >> Collect()
  >>> print(samples)
  [(0, 0, 'no'), (0, 1, 'no'), (1, 0, 'no'), (1, 1, 'yes')]
      
Of course we had to reload ``lines`` again and just for readability gave the
``Drop(1)`` function a meaningful name (``skip_header``). We end up with a
nice pipeline that lazily processes individual lines, is modular and
easy to understand: ``lines >> skip_header >> split >> to_int >> Collect()``
The equivalent Python code without using **nuts-flow/ml** or ``itertools`` would be

.. code:: Python

   def split(line):
       return line.strip().split(',')
       
   def to_int(sample):
       x1, x2, label = sample
       return [int(x1), int(x2), label]
   
   lines = open('tests/data/and.txt')
   next(lines)   
   samples = [to_int(split(line)) for line in lines]  
   
If you prefer Python functions but still want to use pipelining, the
functions can be converted into nuts:

.. code:: Python

   @nut_function
   def Split(line):
       return line.strip().split(',')
       
   @nut_function    
   def ToInt(sample):
       x1, x2, label = sample
       return [int(x1), int(x2), label]
   
   lines = open('tests/data/and.txt') 
   samples = lines >> Drop(1) >> Split() >> ToInt() >> Collect()  
   
As a final example, we will convert the class labels that are currently strings to 
integer numbers -- usually needed for training a machine learning classifier. 
We could define the following nut and add it to the pipeline:
   
  >>> label2int = MapCol(2, lambda label: 1 if label=='yes' else 0)
  >>> open('tests/data/and.txt') >> skip_header >> split >> to_int >> label2int >> Collect()
  [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)] 
  
However, **nutsml** already has the `ConvertLabel <https://maet3608.github.io/nuts-ml/nutsml.html#nutsml.common.ConvertLabel>`_ nut
and we can simply write instead:

  >>> labels = ['no', 'yes']
  >>> convert = ConvertLabel(2, labels)
  >>> samples = (open('tests/data/and.txt') >> skip_header >> split >> to_int >> 
  ... convert >> Print() >> Collect())  
  (0, 0, 0)
  (0, 1, 0)
  (1, 0, 0)
  (1, 1, 1)
  
Using `ConvertLabel` has the additional advantage that the conversion back from 
integers to strings is trivial:
  
  >>> samples >> convert >> Print() >> Consume()
  (0, 0, 'no')
  (0, 1, 'no')
  (1, 0, 'no')
  (1, 1, 'yes')
  
`ConvertLabel(column, labels)` takes as parameter the column in a sample that contains
the class label (here column 2) and a list of labels. If the class label is a strings it
converts to an integer and vice versa.  `ConvertLabel` can also convert to one-hot-encoded
vectors and back:

  >>> convert = ConvertLabel(2, labels, onehot=True)
  >>> samples = (open('tests/data/and.txt') >> skip_header >> split >> to_int >> 
  ... convert >> Print() >> Collect())
  (0, 0, [1, 0])
  (0, 1, [1, 0])
  (1, 0, [1, 0])
  (1, 1, [0, 1])
  
  >>> samples >> convert >> Print() >> Consume()
  (0, 0, 'no')
  (0, 1, 'no')
  (1, 0, 'no')
  (1, 1, 'yes')  

   
   
CSV files
---------   

You will have noticed that the ``tests/data/and.txt`` file used above is actually a
file in CSV (Comma Separated Values) format. Reading of CSV files is so common that 
Python has a dedicated `CSV library
<https://docs.python.org/3/library/csv.html>`_ for it. Similarily, 
**nuts-flow** provides a `ReadCSV <https://github.com/maet3608/nuts-flow/blob/master/nutsflow/source.py>`_ nut,
and **nuts-ml** has the even more powerful `ReadPandas <https://github.com/maet3608/nuts-ml/blob/master/nutsml/reader.py>`_
nut. For instance, we could write

  >>> filepath = 'tests/data/and.csv'
  >>> with ReadCSV(filepath, skipheader=1, fmtfunc=(int,int,str)) as reader:
  >>>    samples = reader >> Collect()
  >>> print(samples)
  [(0, 0, 'no'), (0, 1, 'no'), (1, 0, 'no'), (1, 1, 'yes')]
  
which also properly closes the data file -- a detail we have neglected before.  
The code becomes even simpler with the ``ReadPandas`` nut but note that this
nut reads all data in memory:
  
  >>> from nutsml import ReadPandas
  >>> samples = ReadPandas('tests/data/and.csv') >> Collect()
  >>> print(samples)  
  [(0, 0, 'no'), (0, 1, 'no'), (1, 0, 'no'), (1, 1, 'yes')]

The advantage is that it drops the header, splits the lines and 
converts numbers to integers automatically. ``ReadPandas`` furthermore
can read TSV (Tab Separated Values) files and other format. Finally,
``ReadPandas`` can easily extract or reorder columns or filter rows:

  >>> columns = ['y', 'x1']
  >>> ReadPandas('tests/data/and.csv', columns=columns) >> Print() >> Consume()
  ('no', 0)
  ('no', 0)
  ('no', 1)
  ('yes', 1)
  
  >>> rows = 'x1>0'
  >>> ReadPandas('tests/data/and.csv', rows, columns) >> Print() >> Consume()  
  ('no', 1)
  ('yes', 1)

  
Numpy arrays
------------  
  
To use Numpy arrays as data sources we need to wrap them into an iterator.
In the following example we create an identity matrix, iterate over the rows, 
and print them:
  
  >>> import numpy as np
  >>> data = np.eye(4)
  >>> iter(data) >> Print() >> Consume()
  [1.0, 0.0, 0.0, 0.0]
  [0.0, 1.0, 0.0, 0.0]
  [0.0, 0.0, 1.0, 0.0]
  [0.0, 0.0, 0.0, 1.0]

Note that Numpy arrays larger than memory can be loaded and then processed with 
`np.load(filename, mmap_mode='r') <https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.load.html>`_.
  
  
Web files
---------  

**nuts-flow/ml** allows us to download and process data files from the web on the fly.
Alternatively you can download the file and the process its content as described above.
In the following example, however, we download and process the `Iris data set <https://en.wikipedia.org/wiki/Iris_flower_data_set#Data_set>`_
line by line. First, we open the URL to the data set located on the UCI
machine learning server:

  >>> import urllib
  >>> url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
  >>> lines = urllib.request.urlopen

We now can inspect the first two lines of the data set:
  
  >>> lines(url) >> Head(2)
  [b'5.1,3.5,1.4,0.2,Iris-setosa\n',
   b'4.9,3.0,1.4,0.2,Iris-setosa\n']
   
Here, ``lines`` is just a renaming of the ``urllib.request.urlopen`` function and ``Head(2)``
collects the first two lines. You will notice that (since Py3K) the lines are in binary (b) format.
The following code convert lines to strings, strips the the newline, and splits at comma
to give as samples with columns:

  >>> to_columns = Map(lambda l: l.decode('utf-8').strip().split(','))
  >>> lines(url) >> to_columns >> Head(2)
  [['5.1', '3.5', '1.4', '0.2', 'Iris-setosa'],
   ['4.9', '3.0', '1.4', '0.2', 'Iris-setosa']]

The four numeric features in columns 0 to 3 of the samples are still strings but we want floats.
Mapping the ``float`` function on those columns will do it:

  >>> to_float = MapCol((0,1,2,3), float)
  >>> lines(url) >> to_columns >> to_float >> Head(2)
  [(5.1, 3.5, 1.4, 0.2, 'Iris-setosa'), 
   (4.9, 3.0, 1.4, 0.2, 'Iris-setosa')]
   
Finally, we are going to replace the class labels (e.g. ``'Iris-setosa'``) by
numeric class indices. We could look the the names of the classes up, but
being lazy we extract them directly via

  >>> skip_empty = Filter(lambda cols: len(cols) == 5)
  >>> labels = lines(url) >> to_columns >> skip_empty >> Get(4) >> Dedupe() >> Collect()
  ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
  
where ``Get(4)`` gets the elements in the fourth column of the sample and ``Dedupe()``
removes all duplicate labels. We need ``skip_empty``, since the data set contains an
empty line at the end.

We now can use the extracted ``labels`` and the ``ConvertLabel`` nut to convert 
the class labels in column 4 from strings to class indices. For showcasing, we
download the entire data set put print only every 20-th sample.
  
  >>> (lines(url) >> to_columns >> skip_empty >> to_float >> 
  ...  ConvertLabel(4, labels) >> Print(every_n=20) >> Consume())
  (5.1, 3.8, 1.5, 0.3, 0)
  (5.1, 3.4, 1.5, 0.2, 0)
  (5.2, 2.7, 3.9, 1.4, 1)
  (5.7, 2.6, 3.5, 1.0, 1)
  (5.7, 2.8, 4.1, 1.3, 1)
  (6.0, 2.2, 5.0, 1.5, 2)
  (6.9, 3.1, 5.4, 2.1, 2)

  
Label directories  
-----------------

A common method to organize data and assign labels to large data objects such as
text files, audio recordings or images is to create directories with labels as names
and to store the data objects in the corresponding directories.

For an example let us assume two classes (``0`` and ``1``) and three text files
that are arranged in the following file structure

.. code:: 

  - books
    - 0
      - text0.txt
    - 1
      - text1.txt
      - text11.txt
      
**nuts-ml** supports the reading of such file structures via the `ReadLabelDirs()
<https://maet3608.github.io/nuts-ml/nutsml.html#module-nutsml.reader>`_ nut. The
following code demonstrates its usage:

  >>> samples = ReadLabelDirs('books', '*.txt')
  >>> samples >> Take(3) >> Print() >> Consume()
  ('books/0/text0.txt', '0')
  ('books/1/text1.txt', '1')
  ('books/1/text11.txt', '1')
  
Note that this code does not load the actual text data but the file paths only. 
However, we could easily implement a ``Process`` nut that loads and processes 
the text files individually without loading all texts in memory at once.
For instance, converting text files into word count dictionaries.

.. code:: 

  @nut_function
  def Process(sample):
      filepath, label = sample
      with open(filepath) as f:
        counts = f.read().split(' ') >> CountValues()
        return counts, labels
                
  samples = ReadLabelDirs('books', '*.txt')
  word_counts = samples >> Process() >> Collect()
        






    

    
   

