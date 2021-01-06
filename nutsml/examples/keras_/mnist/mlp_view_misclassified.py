"""
.. module:: mlp_precit
   :synopsis: Example nuts-ml pipeline for prediction
"""

from __future__ import print_function

from nutsflow import Consume, Zip, Unzip, Map, ArgMax, nut_filter
from nutsml import TransformImage, BuildBatch, ViewImageAnnotation

BATCH_SIZE = 128

if __name__ == "__main__":
    from mlp_train import create_network, load_samples

    TransformImage.register('flatten', lambda img: img.flatten())
    transform = (TransformImage(0)
                 .by('rerange', 0, 255, 0, 1, 'float32')
                 .by('flatten'))
    show_image = ViewImageAnnotation(0, (1, 2), pause=3, figsize=(3, 3))
    pred_batch = BuildBatch(BATCH_SIZE).by(0, 'vector', 'float32')
    IsMisclassified = nut_filter(lambda t: t[1] != t[2])

    print('loading samples ...')
    train_samples, test_samples = load_samples()

    print('loading network...')
    network = create_network()
    network.load_weights()

    print('predicting...')
    samples = train_samples + test_samples
    images, trues = samples >> Unzip()
    preds = (samples >> transform >> pred_batch >>
             network.predict() >> Map(ArgMax()))
    images >> Zip(trues, preds) >> IsMisclassified() >> show_image >> Consume()
