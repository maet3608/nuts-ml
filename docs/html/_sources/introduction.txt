Introduction
============

**nuts-ml** is a library for flow-based data pre-processing for machine learning.
It allows to construct pre-processing pipelines such as the following

.. code:: python

   train_sel = train_samples >> Stratify(1) >> Pick(PSEL) >> Collect()
   t_results = (train_sel >> PrintProgress(train_sel) >> read_image >>
                     crop >> augment >> Shuffle(100) >> normalize >>
                     build_batch >> network.train() >> log_train >>
                     Print('train loss:{} acc:{}') >> Collect())
   t_loss, t_acc = t_results >> Unzip() >> Collect()
   print "training loss    :\t\t{:.3f}".format(np.mean(t_loss))
   print "training acc     :\t\t{:.1f}%".format(np.mean(t_acc))

**nuts-ml** is based on `nuts-flow <https://github.com/maet3608/nuts-flow>`_
and reading its `documentation <https://maet3608.github.io/nuts-flow/>`_ is
recommended.