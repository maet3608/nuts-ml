"""
Utility functions for downloading the MNIST data set.
"""

import requests
import pickle
import gzip
from pathlib import Path


def download_mnist():
    """Download MNIST from web to data folder."""
    folder = Path("data/mnist")
    filename = "mnist.pkl.gz"
    fullpath = folder / filename
    url = "http://deeplearning.net/data/mnist/" + filename
    folder.mkdir(parents=True, exist_ok=True)
    if not fullpath.exists():
        content = requests.get(url).content
        fullpath.open("wb").write(content)
    return fullpath


def load_mnist(filepath):
    """Load MNIST data from filepath"""
    with gzip.open(filepath.as_posix(), "rb") as f:
        data = pickle.load(f, encoding="latin-1")
    (x_train, y_train), (x_valid, y_valid), _ = data
    return x_train, y_train, x_valid, y_valid
