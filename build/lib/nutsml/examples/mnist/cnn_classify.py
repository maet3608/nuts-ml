"""
.. module:: cnn_predict
   :synopsis: Example nuts-ml pipeline for network predictions 
"""

from nutsflow import Collect, Consume, Get, Zip, Map, ArgMax, Format
from nutsml import (TransformImage, BuildBatch, ReadLabelDirs, ReadImage,
                    ViewImageAnnotation)

BATCH_SIZE = 128

if __name__ == "__main__":
    from cnn_train import create_network

    transform = TransformImage(0).by('rerange', 0, 255, 0, 1, 'float32')
    show_image = ViewImageAnnotation(0, 1, pause=1, figsize=(3, 3))
    pred_batch = BuildBatch(BATCH_SIZE).by(0, 'image', 'float32')

    print('loading network...')
    network = create_network()
    network.load_weights()

    print('predicting...')
    samples = ReadLabelDirs('images', '*.png') >> ReadImage(0) >> Collect()
    predictions = (samples >> transform >> pred_batch >>
                   network.predict() >> Map(ArgMax()))
    samples >> Get(0) >> Zip(predictions) >> show_image >> Consume()
