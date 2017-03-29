from __future__ import print_function

from glob import glob
from cnn_train import create_network, load_names
from nutsflow import Collect, Consume, Get, Zip, Map, ArgMax, Print
from nutsml import TransformImage, BuildBatch, ReadImage, ViewImageAnnotation

BATCH_SIZE = 128

names = load_names()

transform = TransformImage(0).by('rerange', 0, 255, 0, 1, 'float32')
show_image = ViewImageAnnotation(0, 1, pause=1, figsize=(3, 3),
                                 interpolation='spline36')
pred_batch = BuildBatch(BATCH_SIZE).by(0, 'image', 'float32')
id2name = Map(lambda i: names[i])

print('loading network...')
network = create_network()
network.load_weights()

print('predicting...')
samples = glob('images/*.png') >> Print() >> ReadImage(None) >> Collect()

predictions = (samples >> transform >> pred_batch >>
               network.predict() >> Map(ArgMax()) >> id2name)
samples >> Get(0) >> Zip(predictions) >> show_image >> Consume()
