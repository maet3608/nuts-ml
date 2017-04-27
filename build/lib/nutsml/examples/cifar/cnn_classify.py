"""
.. module:: cnn_predict
   :synopsis: Example pipeline for viewing annotations and classification
"""

from __future__ import print_function

from glob import glob
from nutsflow import Collect, Consume, Get, Zip, Map, ArgMax, Print
from nutsml import TransformImage, BuildBatch, ReadImage, ViewImageAnnotation

BATCH_SIZE = 128

if __name__ == "__main__":
    from cnn_train import create_network, load_names

    names = load_names()

    rerange = TransformImage(0).by('rerange', 0, 255, 0, 1, 'float32')
    show_image = ViewImageAnnotation(0, 1, pause=1, figsize=(3, 3),
                                     interpolation='spline36')
    pred_batch = BuildBatch(BATCH_SIZE).by(0, 'image', 'float32')

    print('loading network...')
    network = create_network()
    network.load_weights()

    print('predicting...')
    samples = glob('images/*.png') >> Print() >> ReadImage(None) >> Collect()

    predictions = (samples >> rerange >> pred_batch >>
                   network.predict() >> Map(ArgMax()) >> Map(names.__getitem__))
    samples >> Get(0) >> Zip(predictions) >> show_image >> Consume()
