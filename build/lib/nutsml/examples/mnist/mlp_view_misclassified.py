from __future__ import print_function

from mlp_train import create_network, load_samples
from nutsflow import Consume, Get, Zip, Unzip, Map, ArgMax, nut_filter
from nutsml import TransformImage, BuildBatch, ViewImageAnnotation

BATCH_SIZE = 128

TransformImage.register('flatten', lambda img: img.flatten())
transform = (TransformImage(0)
             .by('rerange', 0, 255, 0, 1, 'float32')
             .by('flatten'))
show_image = ViewImageAnnotation(0, (1, 2), pause=3, figsize=(3, 3))
pred_batch = BuildBatch(BATCH_SIZE).by(0, 'vector', 'float32')
IsMisclassified = nut_filter(lambda (i, t, p): p != t)

print('loading samples ...')
train_samples, val_samples = load_samples()

print('loading network...')
network = create_network()
network.load_weights()

print('predicting...')
samples = train_samples + val_samples
images, trues = samples >> Unzip()
preds = samples >> transform >> pred_batch >> network.predict() >> Map(ArgMax())
images >> Zip(trues, preds) >> IsMisclassified() >> show_image >> Consume()
