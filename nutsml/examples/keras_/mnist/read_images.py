"""
.. module:: read_images
   :synopsis: Example for reading and viewing of image data
"""

from nutsflow import Consume, Print
from nutsml import ReadLabelDirs, ReadImage, ViewImageAnnotation, PrintColType

if __name__ == "__main__":
    show_image = ViewImageAnnotation(0, 1, pause=1, figsize=(3, 3))

    (ReadLabelDirs('images', '*.png') >> Print() >> ReadImage(0) >>
     PrintColType() >> show_image >> Consume())
