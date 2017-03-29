from __future__ import print_function

from cnn_train import load_samples, load_names
from nutsflow import Take, Consume, MapCol
from nutsml import ViewImageAnnotation, PrintColType

train_samples, val_samples = load_samples()
names = load_names()

id2name = MapCol(1, lambda i: names[i])
show_image = ViewImageAnnotation(0, 1, pause=1, figsize=(2, 2),
                                 interpolation='spline36')

train_samples >> Take(10) >> id2name >> PrintColType() >> show_image >> Consume()