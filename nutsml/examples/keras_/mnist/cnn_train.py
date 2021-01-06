"""
.. module:: cnn_train
   :synopsis: Example nuts-ml pipeline for training a CNN on MNIST

This is code is based on a Keras example (see here)
https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
to train a Multi-layer perceptron on the MNIST data and modified to
use nuts for the data-preprocessing.
"""

from __future__ import print_function

from six.moves import zip, range
from nutsflow import PrintProgress, Collect, Unzip, Shuffle, Pick, Mean, NOP
from nutsml import (KerasNetwork, TransformImage, AugmentImage, BuildBatch,
                    Boost, PrintColType, PlotLines)

PICK = 0.1  # Pick 10% of the data for a quick trial
NUM_EPOCHS = 10
INPUT_SHAPE = (28, 28, 1)
BATCH_SIZE = 128
NUM_CLASSES = 10


def load_samples():
    from tensorflow.python.keras.datasets import mnist
    h, w, c = INPUT_SHAPE
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], h, w)
    x_test = x_test.reshape(x_test.shape[0], h, w)
    return list(zip(x_train, y_train)), list(zip(x_test, y_test))


def create_network():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return KerasNetwork(model, 'cnn_weights.hd5')


def train():
    from tensorflow.keras.metrics import categorical_accuracy

    print('creating network ...')
    network = create_network()

    print('loading data...')
    train_samples, test_samples = load_samples()

    augment = (AugmentImage(0)
               .by('identical', 1)
               .by('translate', 0.5, [-3, +3], [-3, +3])
               .by('rotate', 0.5, [-5, +5])
               .by('shear', 0.5, [0, 0.2])
               .by('elastic', 0.5, [5, 5], [100, 100], [0, 100]))
    transform = (TransformImage(0)
                 .by('rerange', 0, 255, 0, 1, 'float32'))
    build_batch = (BuildBatch(BATCH_SIZE, prefetch=0)
                   .input(0, 'image', 'float32')
                   .output(1, 'one_hot', 'uint8', NUM_CLASSES))
    plot = PlotLines((0, 1), layout=(2, 1), every_sec=1)

    print('training...', NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        print('EPOCH:', epoch)

        t_loss, t_acc = (train_samples >> PrintProgress(train_samples) >>
                         Pick(PICK) >> augment >> transform >> Shuffle(100) >>
                         build_batch >> network.train() >> plot >> Unzip())
        print('train loss : {:.6f}'.format(t_loss >> Mean()))
        print('train acc  : {:.1f}'.format(100 * (t_acc >> Mean())))

        e_acc = (test_samples >> transform >> build_batch >>
                 network.evaluate([categorical_accuracy]))
        print('test acc   : {:.1f}'.format(100 * e_acc))

        network.save_best(e_acc, isloss=False)


if __name__ == "__main__":
    train()
