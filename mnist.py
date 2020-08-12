#!/usr/bin/env python

# Based on Alec Radford code (https://github.com/Newmu/Theano-Tutorials)
# Created by Raul Peralta-Lozada

import os
import numpy as np


def load_mnist(mnist_path, n_train=60000, n_test=10000, onehot=True):
    """ Loads training and testing data from NMIST

        It loads all data, normalizes it, converts the columns of labels
        in a binary matrix, and finally returns the amount of data
        requested.
    """

    # Reads full training data: 60,000 images of 28x28 bytes.
    fd = open(os.path.join(mnist_path, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(np.float32)

    # Reads full training labels.
    fd = open(os.path.join(mnist_path, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000, ))

    # Reads full testing data: 10,000 images of 28x28 bytes.
    fd = open(os.path.join(mnist_path, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(np.float32)

    # Reads full testing labels.
    fd = open(os.path.join(mnist_path, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000, ))

    # Normalizes values in range(256)
    trX /= 255.0
    teX /= 255.0

    trX = trX[:n_train]
    trY = trY[:n_train]

    teX = teX[:n_test]
    teY = teY[:n_test]

    if onehot:
        # Converts a column of values in range(10) into a matrix where
        # [i, value] = 1, and otherwise [i,j] = 0.
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


def one_hot(x, n):
    """ Transforms an column of values in a binary matrix.

        From a column of m values in range(n), it produces a binary matrix
        of size m x n, where entry at [k,k_value] = 1, and otherwise [k,j] = 0
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n), dtype=np.float32)
    o_h[np.arange(len(x)), x] = 1
    return o_h
