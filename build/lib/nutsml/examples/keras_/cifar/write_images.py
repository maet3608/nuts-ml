"""
.. module:: write_images
   :synopsis: Example nuts-ml pipeline for writing of images 
"""

from nutsflow import (Take, Print, Consume, Enumerate, Zip, Format, MapCol,
                      Get)
from nutsml import WriteImage, ConvertLabel

if __name__ == "__main__":
    from cnn_train import load_samples, load_names

    train_samples, _ = load_samples()

    convert_label = ConvertLabel(1, load_names())
    fnames = (Enumerate() >> Zip(train_samples >> Get(1)) >> convert_label >>
              Format('{0}_{1}') >> Print())
    imagepath = 'images/img*.png'
    train_samples >> Take(10) >> WriteImage(0, imagepath, fnames) >> Consume()

