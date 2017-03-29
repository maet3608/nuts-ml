Example
=======

CIFAR10 example

Training
--------

Validation
^^^^^^^^^^

Augmentation
^^^^^^^^^^^^

Prediction
----------


Writing
-------

Writing image data


.. code::

  def train(train_samples, val_samples):
      from keras.metrics import categorical_accuracy

      rerange = TransformImage(0).by('rerange', 0, 255, 0, 1, 'float32')
      build_batch = (BuildBatch(BATCH_SIZE)
                     .by(0, 'image', 'float32')
                     .by(1, 'one_hot', 'uint8', NUM_CLASSES))
      p = 0.1
      augment = (AugmentImage(0)
                 .by('identical', 1.0)
                 .by('brightness', p, [0.7, 1.3])
                 .by('color', p, [0.7, 1.3])
                 .by('shear', p, [0, 0.1])
                 .by('fliplr', p)
                 .by('rotate', p, [-10, 10]))
      plot_eval = PlotLines((0, 1), layout=(2, 1))

      print('creating network...')
      network = create_network()

      print('training...', len(train_samples), len(val_samples))
      for epoch in xrange(NUM_EPOCHS):
          print('EPOCH:', epoch)

          t_loss, t_acc = (train_samples >> PrintProgress(train_samples) >>
                           Pick(PICK) >> augment >> rerange >> Shuffle(100) >>
                           build_batch >> network.train() >> Unzip())
          print("training loss  :\t\t{:.6f}".format(np.mean(t_loss)))
          print("training acc   :\t\t{:.1f}".format(100 * np.mean(t_acc)))

          v_loss, v_acc = (val_samples >> rerange >>
                           build_batch >> network.validate() >> Unzip())
          print("validation loss :\t\t{:.6f}".format(np.mean(v_loss)))
          print("validation acc  :\t\t{:.1f}".format(100 * np.mean(v_acc)))

          e_acc = (val_samples >> rerange >> build_batch >>
                   network.evaluate([categorical_accuracy]))
          print("evaluation acc  :\t\t{:.1f}".format(100 * e_acc))

          network.save_best(e_acc, isloss=False)
          plot_eval((np.mean(t_acc), e_acc))
      print('finished.')


   
