from glob import glob
from nutsflow import Consume
from nutsml import ReadImage, ViewImage, PrintColType

show_image = ViewImage(0, pause=1, figsize=(2, 2), interpolation='spline36')
paths = glob('images/*.png')

paths >> ReadImage(None) >> PrintColType() >> show_image >> Consume()
