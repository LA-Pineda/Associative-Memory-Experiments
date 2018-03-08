#!/usr/bin/env bash

# Adapted from Alec's script (https://raw.githubusercontent.com/Newmu/Theano-Tutorials/master/download_mnist.sh)

mkdir -p ./mnist

if ! [ -e ./mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P ./mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d ./mnist/train-images-idx3-ubyte.gz

if ! [ -e ./mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P ./mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d ./mnist/train-labels-idx1-ubyte.gz

if ! [ -e ./mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P ./mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d ./mnist/t10k-images-idx3-ubyte.gz

if ! [ -e ./mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P ./mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d ./mnist/t10k-labels-idx1-ubyte.gz