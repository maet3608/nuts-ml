"""
.. module:: stratify
   :synopsis: Stratification of sample sets
"""

import random as rnd

from nutsflow import nut_processor
from datautil import upsample, random_downsample


@nut_processor
def Stratify(iterable, labelcol, mode='downrnd', rand=rnd.Random()):
    samples = list(iterable)
    if mode == 'up':
        stratified = upsample(samples, labelcol, rand)
    elif mode == 'downrnd':
        stratified = random_downsample(samples, labelcol, rand)
    else:
        raise ValueError('Unknown mode: ' + mode)
    return iter(stratified)


