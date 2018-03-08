#!/usr/bin/env python

# Based on Alec Radford code (https://github.com/Newmu/Theano-Tutorials)
# Created by Raul Peralta-Lozada

import os
import numpy as np


def load_mnist(mnist_path, n_train=60000, n_test=10000, onehot=True):
    fd = open(os.path.join(mnist_path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(np.float32)

    fd = open(os.path.join(mnist_path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000, ))

    fd = open(os.path.join(mnist_path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(np.float32)

    fd = open(os.path.join(mnist_path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000, ))

    trX /= 255.0
    teX /= 255.0

    trX = trX[:n_train]
    trY = trY[:n_train]

    teX = teX[:n_test]
    teY = teY[:n_test]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n), dtype=np.float32)
    o_h[np.arange(len(x)), x] = 1
    return o_h
