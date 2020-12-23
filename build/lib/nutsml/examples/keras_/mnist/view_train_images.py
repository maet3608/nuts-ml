"""
.. module:: view_train_images
   :synopsis: Example for showing images with transformation
"""

from nutsflow import Take, Consume, GetCols
from nutsml import ViewImage, TransformImage

if __name__ == "__main__":
    from mlp_train import load_samples

    samples, _ = load_samples()
    transform = (TransformImage(0).by('elastic', 5, 100))
    (samples >> GetCols(0,0,1) >> Take(1000) >> transform >>
     ViewImage((0,1), pause=1) >> Consume())
