# Associative-Memory-Experiments
This repository contains the data and procedures to replicate the expermients presented in the paper "A Distributed Extension of the Turing Machine" by Luis A. Pineda at IIMAS, UNAM, México

This program was written in Python 3 and was run on a personal computer with the following specifications:
* CPU: Intel Core i7-6700
* GPU: Nvidia GeForce GTX 1080
* OS: Ubuntu 16.04.2 LTS
* RAM: 16GB

### Requeriments
The following libraries need to be installed beforehand:
* joblib
* matplotlib
* numpy
* theano

### Data
The data that was used for this test was obtained from http://yann.lecun.com/exdb/mnist/ 

This data is downloaded by the script "download_mnist.sh".

All files are provided with the exception of

/mnist/train-images-idx3-ubyte.gz

due to its size.

However, this file is downloades directly by the script.

### Use

To use just run the following command in the source directory

    python3 main_test_associative.py



