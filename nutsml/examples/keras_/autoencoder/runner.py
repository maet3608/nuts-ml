"""
Runs training and prediction.

Trains an autoencoder on MNIST and in the prediction phase shows
the original image, the decoded images and the difference.
"""

from __future__ import print_function

import numpy as np

from six.moves import zip, range
from nutsflow import *
from nutsml import *

NUM_EPOCHS = 10   # need more epochs for good results!
BATCH_SIZE = 128
INPUT_SHAPE = (28, 28, 1)


def create_network():
    import conv_autoencoder as cae
    return cae.create_network()


def load_samples():
    from tensorflow.keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    h, w, c = INPUT_SHAPE
    x_train = np.reshape(x_train, (len(x_train), h, w, c))
    x_test = np.reshape(x_test, (len(x_test), h, w, c))
    return list(zip(x_train, x_train)), list(zip(x_test, x_test))


@nut_function
def Diff(sample):
    x, y = sample
    return x, y, abs(x - y)  # Add difference image to sample


def train():
    print('\n\nTRAIN...')
    rerange = TransformImage((0, 1)).by('rerange', 0, 255, 0, 1, 'float32')
    build_batch = (BuildBatch(BATCH_SIZE)
                   .input(0, 'image', 'float32')
                   .output(1, 'image', 'float32'))

    print('creating network and loading data...')
    network = create_network()
    train_samples, test_samples = load_samples()

    print('training...', len(train_samples), len(test_samples))
    for epoch in range(NUM_EPOCHS):
        print('EPOCH:', epoch)

        t_loss = (train_samples >> PrintProgress(train_samples) >> rerange >>
                  Shuffle(1000) >> build_batch >> network.train() >> Mean())
        print("train loss : {:.6f}".format(t_loss))

        network.save_best(t_loss, isloss=True)


def predict():
    print('\n\nPREDICT...')
    rerange = TransformImage((0, 1)).by('rerange', 0, 255, 0, 1, 'float32')
    build_batch = (BuildBatch(BATCH_SIZE).input(0, 'image', 'float32'))
    view_images = ViewImage((0, 1, 2), pause=0.5,
                            titles=['Input', 'Output', 'Difference'])

    print('creating network ...')
    network = create_network()
    network.load_weights()

    print('loading data...')
    _, test_samples = load_samples()

    print('predicting...')
    preds = test_samples >> rerange >> build_batch >> network.predict()

    (test_samples >> Take(100) >> rerange >> Get(0) >> Zip(preds) >> Diff() >>
     view_images >> Consume())


def view():
    print('\n\nVIEW...')
    train_samples, test_samples = load_samples()
    (train_samples >> Take(10) >> PrintColType() >> ViewImage(0, pause=1) >>
     Consume())


if __name__ == "__main__":
    view()
    train()
    predict()
