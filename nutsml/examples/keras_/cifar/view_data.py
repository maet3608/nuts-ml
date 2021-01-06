"""
.. module:: view_data
   :synopsis: Example nuts-ml pipeline viewing CIFAR-10 image data
"""

from nutsflow import Take, Consume
from nutsml import ViewImage

if __name__ == "__main__":
    from cnn_train import load_samples

    train_samples, test_samples = load_samples()
    samples = train_samples + test_samples
    show_image = ViewImage(0, pause=1, figsize=(2, 2), interpolation='spline36')

    samples >> Take(10) >> show_image >> Consume()
