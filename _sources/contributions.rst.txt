Contributions
=============

Contributions to **nuts-ml** are welcome. Please document the code following
the examples, provide unit tests and ensure that ``pytest`` runs without
errors. 


Unit tests
^^^^^^^^^^

.. code::

  $ cd nutsml
  $ pytest

  ============================= test session starts =============================
  platform win32 -- Python 2.7.13, pytest-3.0.3, py-1.4.31, pluggy-0.4.0
  rootdir: C:\Maet\Projects\Python\nuts-ml, inifile: pytest.ini
  plugins: cov-2.3.1
  collected 170 items

  nutsml\batcher.py .....
  nutsml\common.py ...
  nutsml\config.py ..
  nutsml\datautil.py .......
  nutsml\imageutil.py .............................
  nutsml\logger.py .
  nutsml\network.py ssss
  nutsml\plotter.py .
  nutsml\reader.py s....
  nutsml\stratify.py .
  nutsml\transformer.py ..........
  nutsml\viewer.py ..
  nutsml\writer.py .
  sphinx\source\faq.rst .
  sphinx\source\installation.rst .
  sphinx\source\introduction.rst s
  sphinx\source\tutorial\example.rst .
  sphinx\source\tutorial\reading_samples.rst .
  tests\nutsml\test_batcher.py ........
  tests\nutsml\test_booster.py .
  tests\nutsml\test_common.py ......
  tests\nutsml\test_config.py ..
  tests\nutsml\test_datautil.py .......
  tests\nutsml\test_fileutil.py ........
  tests\nutsml\test_imageutil.py ..............................
  tests\nutsml\test_logger.py ....
  tests\nutsml\test_network.py ........
  tests\nutsml\test_reader.py ......
  tests\nutsml\test_stratify.py .
  tests\nutsml\test_transformer.py ...........
  tests\nutsml\test_viewer.py .
  tests\nutsml\test_writer.py .

  ==================== 164 passed, 6 skipped in 8.06 seconds ====================


We are aiming at a code coverage of 100%. Run ``pytest --cov`` for verification.

.. code::

  $ cd nutsml
  $ pytest --cov

  ---------- coverage: platform win32, python 2.7.13-final-0 -----------
  Name                    Stmts   Miss  Cover
  -------------------------------------------
  nutsml\batcher.py          56      0   100%
  nutsml\booster.py          16      0   100%
  nutsml\common.py           36      0   100%
  nutsml\config.py           14      0   100%
  nutsml\datautil.py         38      0   100%
  nutsml\fileutil.py         29      0   100%
  nutsml\imageutil.py       184      0   100%
  nutsml\logger.py           33      0   100%
  nutsml\network.py          53      0   100%
  nutsml\plotter.py           5      0   100%
  nutsml\reader.py           64      0   100%
  nutsml\stratify.py         12      0   100%
  nutsml\transformer.py     152      0   100%
  nutsml\viewer.py           17      0   100%
  nutsml\writer.py           20      0   100%
  -------------------------------------------
  TOTAL                     729      0   100%


Documentation
^^^^^^^^^^^^^

Update Sphinx/HTML documentation as follows

.. code::

  cd sphinx
  make clean
  make html

  cd ..
  ./push_docs


Style guide
^^^^^^^^^^^

Code should be formatted following the `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_
style guide. 

Names of *nuts* shoulds be in CamelCase (just like class names) and describe an action,
e.g. ``ReadCSV``, ``BuildBatch`` but not ``CSVReader`` or ``Batcher``.

Prefer *immutable* data types, e.g. tuples over lists, for outputs of nuts and
avoid nuts with *side-effects. Nuts should not *mutate* their input data but create
copies.

If a nut has no input it should be a *source*, for instance like ``Range``. 
If it doesn't output a generator or iterator it should be a *sink*, 
see ``Collect`` for example.
If a nut outputs the same number of elements it reads, it probably
is a *function* (e.g. ``Square``) otherwise a *processor*.