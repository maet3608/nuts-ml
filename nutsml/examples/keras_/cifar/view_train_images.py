"""
.. module:: view_train_images
   :synopsis: Example nuts-ml pipeline reading and viewing image data
"""

from nutsflow import Take, Consume, MapCol
from nutsml import ViewImageAnnotation, PrintColType, ConvertLabel

if __name__ == "__main__":
    from cnn_train import load_samples, load_names

    train_samples, _ = load_samples()

    convert_label = ConvertLabel(1, load_names())
    show_image = ViewImageAnnotation(0, 1, pause=1, figsize=(2, 2),
                                     interpolation='spline36')

    (train_samples >> Take(10) >> convert_label >> PrintColType() >>
     show_image >> Consume())
