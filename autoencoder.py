import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

import constants

ENCODER = 1
DECODER = 2

def get_data(one_hot = False):

   # Load MNIST data, as part of TensorFlow.
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    all_data = np.concatenate((train_images, test_images), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis= 0)

    all_data = all_data.reshape((len(all_data), 28, 28, 1))
    all_data = all_data.astype('float32') / 255

    if one_hot:
        # Changes labels to binary rows. Each label correspond to a column, and only
        # the column for the corresponding label is set to one.
        all_labels = to_categorical(all_labels)

    return (all_data, all_labels)


def get_encoder():

    # Creates a two parts neural network: a convolutional neural network (CNN) with
    # three main layers, and a full connected neural network.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32,kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Conv2D(32,kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=5, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(constants.domain*2, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4)),
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    print('Encoder:')
    model.summary()
    return model


def get_decoder():
    model = tf.keras.models.Sequential()
    input_img = tf.keras.layers.Input(shape=(64, ))
    model.add(tf.keras.layers.Dense(units=7*7*32, activation='relu', input_shape=(64, )))
    model.add(tf.keras.layers.Reshape((7, 7, 32)))
    model.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(1, kernel_size=3, activation='sigmoid', padding='same'))
    # model.add(tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='same'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print('Decoder:')
    model.summary()
    return model


def get_model(type = ENCODER):
    if type == ENCODER:
        return get_encoder()
    elif type == DECODER:
        return get_decoder()
    else:
        return None


def train_networks(pops):

    EPOCHS = 5

    (data, labels) = get_data(one_hot=True)

    total = len(data)
    step = int(total/constants.training_stages)

    # Amount of data used for testing
    ntd = total - int(total*constants.nn_training_percent)

    n = 0
    enstats = []
    destats = []

    for i in range(0, total, step):
        j = (i + ntd) % total

        if j > i:
            testing_data = data[i:j]
            testing_labels = labels[i:j]
            training_data = np.concatenate((data[0:i], data[j:total]), axis=0)
            training_labels = np.concatenate((labels[0:i], labels[j:total]), axis=0)
        else:
            testing_data = np.concatenate((data[i:total], data[0:j]), axis=0)
            testing_labels = np.concatenate((labels[i:total], labels[0:j]), axis=0)
            training_data = data[j:i]
            training_labels = labels[j:i]
        
        encoder = get_model(ENCODER)
        encoder.fit(training_data, training_labels,
            batch_size=100,
            epochs=EPOCHS,
            verbose=2)

        test_loss, test_acc = encoder.evaluate(testing_data, testing_labels)
        enstats.append((test_loss, test_acc))

        # Drop the last layers of the full connected neural network part.
        for i in range(pops):
            encoder.pop()
 
        training_features = encoder.predict(training_data)

        decoder = get_model(DECODER)
        decoder.fit(training_features, training_data,
            batch_size=100,
            epochs=EPOCHS,
            verbose=2)

        testing_features = encoder.predict(testing_data)
        test_loss, test_acc = decoder.evaluate(testing_features, testing_data)
        destats.append((test_loss, test_acc))

        prod_images = decoder.predict(testing_features)
        prod_images *= 255

        for i in range(len(prod_images)):
            plt.imshow(testing_data[i].reshape(28,28))
            plt.gray()
            plt.show()
            plt.imshow(prod_images[i].reshape((28,28)))
            plt.gray()
            plt.show()
            fig = plt.figure()
            fig.savefig('image.png')

        encoder.save(constants.model_filename(n))
        n += 1
    
    return np.array(enstats), np.array(destats)


def obtain_features(features_fn_prefix, labels_fn_prefix, pops):
 
    (data, labels) = get_data()

    total = len(data)
    step = int(total/constants.training_stages)

    # Amount of data used for testing
    ntd = total - int(total*constants.nn_training_percent)

    n = 0
    for i in range(0, total, step):
        j = (i + ntd) % total

        if j > i:
            testing_data = data[i:j]
            testing_labels = labels[i:j]
        else:
            testing_data = np.concatenate((data[0:j], data[i:total]), axis=0)
            testing_labels = np.concatenate((labels[0:j], labels[i:total]), axis=0)
 
        # Recreate the exact same model, including its weights and the optimizer
        model = tf.keras.models.load_model(constants.model_filename(n))

        # Drop the last layers of the full connected neural network part.
        for i in range(pops):
            model.pop()
 
        features = model.predict(testing_data)

        feat_filename = constants.data_filename(features_fn_prefix, n)
        labl_filename = constants.data_filename(labels_fn_prefix, n)
        np.save(feat_filename, features)
        np.save(labl_filename, testing_labels)

        n += 1

