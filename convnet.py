import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


import constants


EPOCHS = 5


def get_data(one_hot = False):

   # Load MNIST data, as part of TensorFlow.
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # MNIST dataset includes 60,000 training elements.
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    # MNIST dataset includes 10,000 testing elements.
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    if one_hot:
        # Changes labels to binary rows. Each label correspond to a column, and only
        # the column for the corresponding label is set to one.
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def train_network():

    (train_images, train_labels), (test_images, test_labels) = get_data(one_hot= True)

    # Creates a two parts neural network: a convolutional neural network (CNN) with three
    # main layers, and a full connected neural network.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32,(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5)),
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='RMSprop',
                metrics=['accuracy'])

    model.fit(train_images, train_labels,
            batch_size=100,
            epochs=EPOCHS,
            verbose=1)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    model.save(constants.full_model_filename)


def obtain_features(tr_filename, te_filename, model_filename, pops):

    (train_images, train_labels), (test_images, test_labels) = get_data()

    # Recreate the exact same model, including its weights and the optimizer
    model = tf.keras.models.load_model(constants.full_model_filename)

    # Drop the last two layers of the full connected neural network part.
    for i in range(pops):
        model.pop()
    model.summary()

    features = model.predict(train_images, batch_size=100)
    np.save(tr_filename, features)
    
    features = model.predict(test_images)
    np.save(te_filename, features)

    np.save(constants.train_labels_filename, train_labels)
    np.save(constants.test_labels_filename, test_labels)

    # Save model with a Dense layer as the last one.
    model.save(model_filename)

