"""
.. module:: read_images
   :synopsis: Example nuts-ml pipeline for reading and viewing image data
"""

from glob import glob
from nutsflow import Consume
from nutsml import ReadImage, ViewImage, PrintColType

if __name__ == "__main__":
    show_image = ViewImage(0, pause=1, figsize=(2, 2), interpolation='spline36')
    paths = glob('images/*.png')

    paths >> ReadImage(None) >> PrintColType() >> show_image >> Consume()
