Configuration files
===================

Frequently we want to store configuration information of our network architecture
or other training parameters in configuration files. **nuts-ml** provides a 
``Config`` dictionary to simplify this. The following example shows how to
create, access and update a configuration dictionary:

 .. doctest::
  
    >>> from nutsml import Config
    >>> cfg = Config({'epochs':100, 'layer1':{'stride':2, 'filters':32}})
    
 .. doctest:: 
 
    >>> cfg.epochs
    100

 .. doctest::
  
    >>> cfg.layer1.filters
    32
    
 .. doctest::
  
    >>> cfg.layer1
    {'stride':2, 'filters':32}

 .. doctest::
  
    >>> cfg.layer1.filters = 64
    >>> cfg.layer1.filters
    64
    
 .. doctest::
  
    >>> cfg.layer2 = Config({'stride':4, 'filters':16})
    >>> cfg.layer2.stride
    4   
    
Configuration data can easily be saved and loaded to the file system in 
JSON or YAML format:

 .. code:: Python

    cfg = Config({'epochs':100, 'mode':'TRAIN'})
    cfg.save('tests/data/config.yaml')
                
 .. code:: Python
 
    cfg = Config().load('tests/data/config.json')

