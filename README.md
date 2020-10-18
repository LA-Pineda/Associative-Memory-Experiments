# Associative Memories Experiments
This repository contains the data and procedures to replicate the expermients presented in the paper “[An Entropic Associative Memory](https://arxiv.org/abs/2009.13058)” by Luis A. Pineda and Gibrán Fuentes (IIMAS, UNAM, Mexico), and Rafael Morales (SUV, UDG, Mexico).

This program was written in Python 3 and was run on a desktop computer with the following specifications:
* CPU: Intel Core i7-6700 at 3.40 GHz
* GPU: Nvidia GeForce GTX 1080
* OS: Ubuntu 16.04 Xenial
* RAM: 64GB

### Requeriments
The following libraries need to be installed beforehand:
* joblib
* matplotlib
* numpy
* png
* TensorFlow 2.3

The experiments were run using the Anaconda 3 distribution.

### Data
The MNIST database of handwritten digits, available throught TensorFlow 2.3, was used for all the experiments.

### Use

To see how to use the code, just run the following command in the source directory

```shell
python3 main_test_associative.py -h
```



