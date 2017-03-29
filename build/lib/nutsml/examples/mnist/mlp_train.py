"""
This is code is based on a Keras example (see here)
https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
to train a Multi-layer perceptron on the MNIST data and modified to
use nuts for the data-preprocessing.
"""

from __future__ import print_function

import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.metrics import categorical_accuracy

from nutsflow import PrintProgress, Collect, Unzip
from nutsml import KerasNetwork, TransformImage, BuildBatch, PlotLines

NUM_EPOCHS = 3
BATCH_SIZE = 128
NUM_CLASSES = 10


def load_samples():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return zip(X_train, y_train), zip(X_test, y_test)


def create_network():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return KerasNetwork(model, filepath='mlp_weights.hd5')


def train():
    TransformImage.register('flatten', lambda img: img.flatten())
    transform = (TransformImage(0)
                 .by('rerange', 0, 255, 0, 1, 'float32')
                 .by('flatten'))
    build_batch = (BuildBatch(BATCH_SIZE)
                   .by(0, 'vector', 'float32')
                   .by(1, 'one_hot', 'uint8', NUM_CLASSES))
    plot = PlotLines((0, 1), layout=(2, 1), every_sec=1)

    print('loading data...')
    train_samples, val_samples = load_samples()

    print('constructing network ...')
    network = create_network()

    print('training...', NUM_EPOCHS)
    for epoch in xrange(NUM_EPOCHS):
        print('EPOCH:', epoch)

        t_loss, t_acc = (train_samples >> PrintProgress(train_samples) >>
                         transform >> build_batch >>
                         network.train() >> plot >> Unzip())
        print("training loss  : {:.6f}".format(np.mean(t_loss)))
        print("training acc   : {:.1f}".format(100 * np.mean(t_acc)))

        e_acc = (val_samples >> transform >> build_batch
                 >> network.evaluate([categorical_accuracy]))
        print("evaluation acc : {:.1f}".format(100 * e_acc))

        network.save_best(e_acc, isloss=False)


if __name__ == "__main__":
    train()
