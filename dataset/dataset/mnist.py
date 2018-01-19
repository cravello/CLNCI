#!/usr/bin/env python

import os

import numpy as np
import hashlib
from urllib.request import urlretrieve

train_images = {"url": "http://yann.lecun.com/exdb/\
train-images-idx3-ubyte.gz",
                "filename": "train-images-idx3-ubyte.gz"}
train_labels = {"url": "http://yann.lecun.com/exdb/mnist/\
train-labels-idx1-ubyte.gz",
                "filename": "train-labels-idx1-ubyte.gz"}
test_images = {"url": "http://yann.lecun.com/exdb/mnist/\
t10k-images-idx3-ubyte.gz",
               "filename": "t10k-images-idx3-ubyte.gz"}
test_labels = {"url": "http://yann.lecun.com/exdb/mnist/\
t10k-labels-idx1-ubyte.gz",
               "filename": "t10k-labels-idx1-ubyte.gz"}


def load_train_images(path="./"):
    with open(path + "train-images-idx3-ubyte",  "rb") as file:
        train_images = np.frombuffer(file.read(), np.uint8, offset=16)
        train_images = train_images.reshape((60000, 28, 28))
    return train_images


def load_train_labels(path="./"):
    with open(path + "train-labels-idx1-ubyte",  "rb") as file:
        train_labels = np.frombuffer(file.read(), np.uint8, offset=8)
    return train_labels


def load_test_images(path="./"):
    with open(path + "t10k-images-idx3-ubyte",  "rb") as file:
        test_images = np.frombuffer(file.read(), np.uint8, offset=16)
        test_images = test_images.reshape((10000, 28, 28))
    return test_images


def load_test_labels(path="./"):
    with open(path + "t10k-labels-idx1-ubyte",  "rb") as file:
        test_labels = np.frombuffer(file.read(), np.uint8, offset=8)
    return test_labels


def load_train(path="./"):
    return load_train_images(path=path), load_train_labels(path=path)


def load_test(path="./"):
    return load_test_images(path=path), load_test_labels(path=path)


def download_train_labels(directory="./"):
    path = os.path.join(directory, train_labels["filename"])
    urlretrieve(train_labels["url"], path)
    return path


def check():
    pass


def check_train_images():
    pass


def check_train_labels(directory="./"):
    path = os.path.join(directory, train_labels["filename"])
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == "d53e105ee54ea40749a09fcbcd1e9432"


def check_train():
    pass


def check_test_images():
    pass


def check_test_labels():
    pass


def check_test():
    pass


if __name__ == "__main__":
    pass
