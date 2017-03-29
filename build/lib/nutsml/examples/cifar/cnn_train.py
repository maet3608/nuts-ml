"""
Example nuts-ml pipeline for CIFAR-10 training and prediction
"""
from __future__ import print_function

import cPickle

import numpy as np
import os.path as osp

from keras.datasets import cifar10
from keras.utils.data_utils import get_file
from nutsflow import (PrintProgress, Collect, Zip, Unzip, Pick, Take, Map,
                      ArgMax, Get, Consume, Shuffle, nut_function)
from nutsml import (KerasNetwork, TransformImage, AugmentImage, BuildBatch,
                    PlotLines)

PICK = 0.1   # Pick 10% of the data for a quick trial
NUM_EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)


def load_samples():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_train = map(int, y_train)
    y_test = map(int, y_test)
    return zip(x_train, y_train), zip(x_test, y_test)


def load_names():
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)
    with open(osp.join(path, 'batches.meta'), 'rb') as f:
        return cPickle.load(f)['label_names']


def create_network():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D

    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return KerasNetwork(model, filepath='weights_cifar10.hd5')


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


if __name__ == "__main__":
    train_samples, val_samples = load_samples()
    train(train_samples, val_samples)
