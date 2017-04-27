"""
.. module:: mlp_predict
   :synopsis: Example nuts-ml pipeline for classification
"""

from __future__ import print_function

from nutsflow import Collect, Consume, Get, Zip, Map, Format, ArgMax
from nutsml import (TransformImage, BuildBatch, ReadImage, ReadLabelDirs,
                    ViewImageAnnotation)

BATCH_SIZE = 128

if __name__ == "__main__":
    from mlp_train import create_network

    TransformImage.register('flatten', lambda img: img.flatten())
    transform = (TransformImage(0)
                 .by('rerange', 0, 255, 0, 1, 'float32')
                 .by('flatten'))
    show_image = ViewImageAnnotation(0, (1, 2), pause=1, figsize=(4, 4))
    pred_batch = BuildBatch(BATCH_SIZE).by(0, 'vector', 'float32')

    print('loading network...')
    network = create_network()
    network.load_weights()

    print('predicting...')
    samples = ReadLabelDirs('images', '*.png') >> ReadImage(0) >> Collect()
    truelabels = samples >> Get(1) >> Format('true: {}')
    predictions = (samples >> transform >> pred_batch >>
                   network.predict() >> Map(ArgMax()) >> Format('pred: {}'))
    samples >> Get(0) >> Zip(predictions, truelabels) >> show_image >> Consume()
