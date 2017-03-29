from glob import glob
from nutsflow import Consume, Print
from nutsml import ReadLabelDirs, ReadImage, ViewImageAnnotation, PrintColType

show_image = ViewImageAnnotation(0, 1, pause=1, figsize=(3, 3))

ReadLabelDirs('images', '*.png') >> Print() >> ReadImage(0) >> \
PrintColType() >> show_image >> Consume()
