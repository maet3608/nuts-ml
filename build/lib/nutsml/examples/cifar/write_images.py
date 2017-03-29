from cnn_train import load_samples, load_names
from nutsflow import (Take, Print, Consume, Map, Enumerate, Zip, Format, MapCol,
                      Get)
from nutsml import WriteImage

train_samples, val_samples = load_samples()
names = load_names()

id2name = MapCol(1, lambda i: names[int(i)])
fnames = (Enumerate() >> Zip(train_samples >> Get(1)) >> id2name >>
          Format('{0}_{1}') >> Print())
imagepath = 'images/img*.png'
train_samples >> Take(10) >> WriteImage(0, imagepath, fnames) >> Consume()
