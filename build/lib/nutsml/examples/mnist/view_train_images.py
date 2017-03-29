from __future__ import print_function

from mlp_train import load_samples
from nutsflow import Take, Consume
from nutsml import ViewImageAnnotation

train_samples, val_samples = load_samples()
(train_samples >> Take(10) >> ViewImageAnnotation(0, 1, pause=1) >> Consume())
