"""
.. module:: view_augmented_images
   :synopsis: Example nuts-ml pipeline for viewing augmented image data
"""

from nutsflow import Take, Consume
from nutsml import ViewImageAnnotation, AugmentImage

if __name__ == "__main__":
    from cnn_train import load_samples

    train_samples, _ = load_samples()

    p = 0.5
    augment = (AugmentImage(0)
               .by('identical', 1.0)
               .by('brightness', p, [0.7, 1.3])
               .by('rotate', p, [-10, 10])
               .by('fliplr', p))
    show_image = ViewImageAnnotation(0, 1, pause=1, figsize=(2, 2),
                                     interpolation='spline36')

    (train_samples >> Take(10) >> augment >> show_image >> Consume())
