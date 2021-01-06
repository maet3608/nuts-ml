Logging data
============

Apart from printing loss, accuracy and other metrics during training it is
often useful to log these numbers to a file. **nuts-ml** provides logging
functionality within and outside of data pipelines via ``LogToFile``.
The following example demonstrates the basics. We have a list of samples
that we log to a CSV file:


.. code:: Python
 
    >>> filepath = 'tests/data/temp_logfile.csv'
    >>> samples = [(1, 2, 3), (4, 5, 6)]
    >>> with LogToFile(filepath) as logtofile:
    ...     samples >> logtofile >> Consume()
    
    >>> open(filepath).read()
    1,2,3
    4,5,6

``LogToFile`` allows to extract sample columns to log and to
specify column names for the log file. In this next example we also show
how to manually close and delete a created log file:
        
.. code:: Python
 
    >>> logtofile = LogToFile(filepath, cols=(2, 0), colnames=['A', 'B'])
    >>> samples >> logtofile >> Consume()
    
    >>> open(filepath).read()
    A,B
    3,1
    6,4

    >>> logtofile.close()
    >>> logtofile.delete()
    
    
In this more complex code sketch we will use ``LogToFile`` within a training a loop
and log loss and accuracy per batch, and epoch, mean loss and mean accuracy per
epoch:

.. code:: Python
 
    log_batch = LogToFile('batchlog.csv', colnames=['loss', 'acc'])
    log_epoch = LogToFile('epochlog.csv', colnames=['epoch', 'loss', 'acc'])
    mean = Mean()
 
    for epoch in range(EPOCHS):
        t_loss, t_acc = (train_samples >> ... >> build_batch >> 
                         network.train() >> log_batch >> Unzip())      
        log_epoch( (epoch, mean(t_loss), mean(t_acc) )
      
    log_batch.close()
    log_epoch.close()
  
The output of ``network.train()`` is a NumPy array, containing
the loss and accuracy per mini-batch (these are the outputs that Keras produces
during training). Note that we call ``log_epoch`` explicitly (outside of the pipeline) 
and can simply provide the values to log as a tuple, list or array. Of course, 
the number of values must match the number column names defined.
The same syntactical feature is used for ``Mean`` here. For instance, the following
three constructs are equivalent:

.. doctest::
  
    >>> [1, 2, 3] >> Mean()
    2.0
  
.. doctest::
  
    >>> Mean()([1, 2, 3])
    2.0
  
.. doctest::
  
    >>> mean = Mean()
    >>> mean([1, 2, 3])
    2.0
  

Similar to logging we can also plot data. This is the topic of the next section.



    
