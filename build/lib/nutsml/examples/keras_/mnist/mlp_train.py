"""
.. module:: mlp_train
   :synopsis: Example nuts-ml pipeline for training and evaluation

This is code is based on a Keras example (see here)
https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
to train a Multi-layer perceptron on the MNIST data and modified to
use nuts for the data-preprocessing.
"""

from __future__ import print_function

from six.moves import zip, range
from nutsflow import PrintProgress, Collect, Unzip, Mean
from nutsml import (KerasNetwork, TransformImage, BuildBatch, PlotLines,
                    PrintType)

NUM_EPOCHS = 5
BATCH_SIZE = 128
NUM_CLASSES = 10


def load_samples():
    from tensorflow.python.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_samples = list(zip(x_train, map(int, y_train)))
    test_samples = list(zip(x_test, map(int, y_test)))
    return train_samples, test_samples


def create_network():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation

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
    return KerasNetwork(model, 'mlp_weights.hd5')


def train():
    from tensorflow.keras.metrics import categorical_accuracy

    TransformImage.register('flatten', lambda img: img.flatten())
    transform = (TransformImage(0)
                 .by('rerange', 0, 255, 0, 1, 'float32')
                 .by('flatten'))
    build_batch = (BuildBatch(BATCH_SIZE)
                   .input(0, 'vector', 'float32')
                   .output(1, 'one_hot', 'uint8', NUM_CLASSES))
    plot = PlotLines((0, 1), layout=(2, 1), every_sec=1)

    print('loading data...')
    train_samples, test_samples = load_samples()

    print('creating network ...')
    network = create_network()

    print('training...', NUM_EPOCHS)
    for epoch in range(NUM_EPOCHS):
        print('EPOCH:', epoch)

        t_loss, t_acc = (train_samples >> PrintProgress(train_samples) >>
                         transform >> build_batch >>
                         network.train() >> plot >> Unzip())
        print('train loss : {:.6f}'.format(t_loss >> Mean()))
        print('train acc  : {:.1f}'.format(100 * (t_acc >> Mean())))

        e_acc = (test_samples >> transform >> build_batch >>
                 network.evaluate([categorical_accuracy]))
        print('test acc   : {:.1f}'.format(100 * e_acc))

        network.save_best(e_acc, isloss=False)


if __name__ == "__main__":
    train()
