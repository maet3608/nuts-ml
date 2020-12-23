"""
.. module:: mlp_view_misclassified
   :synopsis: Example for showing misclassified examples
"""

from __future__ import print_function

import pickle

import os.path as osp

from six.moves import zip, map, range
from nutsflow import PrintProgress, Zip, Unzip, Pick, Shuffle, Mean
from nutsml import (KerasNetwork, TransformImage, AugmentImage, BuildBatch,
                    SplitRandom, PlotLines, PrintType)

PICK = 0.1  # Pick 10% of the data for a quick trial
NUM_EPOCHS = 10
BATCH_SIZE = 128
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


def load_samples():
    from tensorflow.python.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_samples = list(zip(x_train, map(int, y_train)))
    test_samples = list(zip(x_test, map(int, y_test)))
    return train_samples, test_samples


def load_names():
    from tensorflow.python.keras.utils.data_utils import get_file
    dirname = 'cifar-10-batches-py'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)
    with open(osp.join(path, 'batches.meta'), 'rb') as f:
        return pickle.load(f)['label_names']


def create_network():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D

    model = Sequential()
    model.add(Convolution2D(32, (3, 3), padding='same',
                            input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3)))
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
    return KerasNetwork(model, 'weights_cifar10.hd5')


def train():
    from tensorflow.keras.metrics import categorical_accuracy

    rerange = TransformImage(0).by('rerange', 0, 255, 0, 1, 'float32')
    build_batch = (BuildBatch(BATCH_SIZE)
                   .input(0, 'image', 'float32')
                   .output(1, 'one_hot', 'uint8', NUM_CLASSES))
    p = 0.1
    augment = (AugmentImage(0)
               .by('identical', 1.0)
               .by('elastic', p, [5, 5], [100, 100], [0, 100])
               .by('brightness', p, [0.7, 1.3])
               .by('color', p, [0.7, 1.3])
               .by('shear', p, [0, 0.1])
               .by('fliplr', p)
               .by('rotate', p, [-10, 10]))
    plot_eval = PlotLines((0, 1), layout=(2, 1), titles=['train', 'val'])

    print('creating network...')
    network = create_network()

    print('loading data...')
    train_samples, test_samples = load_samples()
    train_samples, val_samples = train_samples >> SplitRandom(0.8)

    print('training...', len(train_samples), len(val_samples))
    for epoch in range(NUM_EPOCHS):
        print('EPOCH:', epoch)

        t_loss, t_acc = (train_samples >> PrintProgress(train_samples) >>
                         Pick(PICK) >> augment >> rerange >> Shuffle(100) >>
                         build_batch >> network.train() >> Unzip())
        t_loss, t_acc = t_loss >> Mean(), t_acc >> Mean()
        print("train loss : {:.6f}".format(t_loss))
        print("train acc  : {:.1f}".format(100 * t_acc))

        v_loss, v_acc = (val_samples >> rerange >>
                         build_batch >> network.validate() >> Unzip())
        v_loss, v_acc = v_acc >> Mean(), v_acc >> Mean()
        print('val loss   : {:.6f}'.format(v_loss))
        print('val acc    : {:.1f}'.format(100 * v_acc))

        network.save_best(v_acc, isloss=False)
        plot_eval((t_acc, v_acc))

    print('testing...', len(test_samples))
    e_acc = (test_samples >> rerange >> build_batch >>
             network.evaluate([categorical_accuracy]))
    print('test acc   : {:.1f}'.format(100 * e_acc))


if __name__ == "__main__":
    train()
