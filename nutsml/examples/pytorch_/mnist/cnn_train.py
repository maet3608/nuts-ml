"""
.. module:: cnn_train
   :synopsis: Example nuts-ml pipeline for training a CNN on MNIST
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import nutsflow as nf
import nutsml as nm
import numpy as np

from nutsml.network import PytorchNetwork
from utils import download_mnist, load_mnist

BATCHSIZE = 64
EPOCHS = 3


class Model(nn.Module):
    """Pytorch model"""

    def __init__(self, device='cpu'):
        """Construct model on given device, e.g. 'cpu' or 'cuda'"""
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.BatchNorm2d(20),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(True),
            nn.Linear(50, 10),
        )
        self.to(device)  # set device before constructing optimizer

        # required properties of a model to be wrapped as PytorchNetwork!
        self.device = device  # 'cuda', 'cuda:0' or 'gpu'
        self.losses = F.cross_entropy  # can be list of loss functions
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        """Forward pass through network for input x"""
        return self.layers(x)


build_batch = (nm.BuildBatch(BATCHSIZE)
               .input(0, 'image', 'float32', True)
               .output(1, 'number', 'int64'))
build_pred_batch = (nm.BuildBatch(BATCHSIZE)
                    .input(0, 'image', 'float32', True))
augment = (nm.AugmentImage(0)
           .by('identical', 1)
           .by('translate', 0.2, [-3, +3], [-3, +3])
           .by('rotate', 0.2, [-30, +30])
           .by('shear', 0.2, [0, 0.2])
           .by('elastic', 0.2, [5, 5], [100, 100], [0, 100])
           )
vec2img = nf.MapCol(0, lambda x: (x.reshape([28, 28]) * 255).astype('uint8'))


def accuracy(y_true, y_pred):
    """Compute accuracy"""
    from sklearn.metrics import accuracy_score
    y_pred = [yp.argmax() for yp in y_pred]
    return 100 * accuracy_score(y_true, y_pred)


def train(network, x, y, epochs):
    """Train network for given number of epochs"""
    for epoch in range(epochs):
        print('epoch', epoch + 1)
        losses = (zip(x, y) >> nf.PrintProgress(x) >> vec2img >>
                  augment >> nf.Shuffle(1000) >> build_batch >>
                  network.train() >> nf.Collect())
        print('train loss: %.4f' % np.mean(losses))


def validate(network, x, y):
    """Compute validation/test loss (= mean over batch losses)"""
    losses = (zip(x, y) >> nf.PrintProgress(x) >> vec2img >>
              build_batch >> network.validate() >> nf.Collect())
    print('val loss: %.4f' % np.mean(losses))


def predict(network, x, y):
    """Compute network outputs and print accuracy"""
    preds = (zip(x, y) >> nf.PrintProgress(x) >> vec2img >>
             build_pred_batch >> network.predict() >> nf.Collect())
    acc = accuracy(y, preds)
    print('test acc %.1f %%' % acc)


def evaluate(network, x, y):
    """Evaluate network performance (here accuracy)"""
    metrics = [accuracy]
    result = (zip(x, y) >> nf.PrintProgress(x) >> vec2img >>
              build_batch >> network.evaluate(metrics))
    return result


def view_misclassified_images(network, x, y):
    """Show misclassified images"""
    make_label = nf.Map(lambda s: (s[0], 'true:%d  pred:%d' % (s[1], s[2])))
    filter_error = nf.Filter(lambda s: s[1] != s[2])
    view_image = nm.ViewImageAnnotation(0, 1, pause=1)

    preds = (zip(x, y) >> vec2img >> build_pred_batch >>
             network.predict() >> nf.Map(np.argmax) >> nf.Collect())
    (zip(x, y, preds) >> vec2img >> filter_error >> make_label >>
     view_image >> nf.Consume())


def view_augmented_images(x, y, n=10):
    """Show n augmented images"""
    view_image = nm.ViewImageAnnotation(0, 1, pause=1)
    zip(x, y) >> vec2img >> augment >> nf.Take(n) >> view_image >> nf.Consume()


if __name__ == '__main__':
    print('loading data...')
    filepath = download_mnist()
    x_train, y_train, x_test, y_test = load_mnist(filepath)

    print('creating model...')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Model(device)
    network = PytorchNetwork(model)
    # network.load_weights()
    network.print_layers((1, 28, 28))

    print('training ...')
    train(network, x_train, y_train, EPOCHS)
    network.save_weights()

    print('evaluating ...')
    print('train acc:', evaluate(network, x_train, y_train))
    print('test  acc:', evaluate(network, x_test, y_test))

    print('validating ...')
    validate(network, x_test, y_test)

    print('predicting ...')
    predict(network, x_test, y_test)

    # print('viewing images...')
    # view_augmented_images(x_test, y_test)

    # print('showing errors ...')
    # view_misclassified_images(network, x_test, y_test)
