#!/usr/bin/env python

# Based on Alec Radford code (https://github.com/Newmu/Theano-Tutorials)
# Created by Raul Peralta-Lozada

import theano

import numpy as np
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d


srng = RandomStreams()


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def rectify(X):
    return T.maximum(X, 0.)


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def dropout(X, p=0.0):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


def convnet_model(X, w1, w2, w3, w4, w5, p_drop_conv, p_drop_hidden):
    # l1_relu = rectify(conv2d(X, w1, border_mode='full'))
    l1_relu = T.nnet.relu(conv2d(X, w1, border_mode='full'), alpha=0)
    l1_pool = pool_2d(l1_relu, (2, 2), ignore_border=False)
    l1_out = dropout(l1_pool, p_drop_conv)

    l2_relu = T.nnet.relu(conv2d(l1_out, w2))
    l2_pool = pool_2d(l2_relu, (2, 2), ignore_border=False)
    l2_out = dropout(l2_pool, p_drop_conv)

    l3_relu = T.nnet.relu(conv2d(l2_out, w3))
    l3_pool = pool_2d(l3_relu, (2, 2), ignore_border=False)
    l3_flat = T.flatten(l3_pool, outdim=2)
    l3_out = dropout(l3_flat, p_drop_conv)

    l4_relu = T.nnet.relu(T.dot(l3_out, w4))
    l4_out = dropout(l4_relu, p_drop_hidden)

    pyx = softmax(T.dot(l4_out, w5))
    return l1_out, l2_out, l3_out, l4_out, pyx
