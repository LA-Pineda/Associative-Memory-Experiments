import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import constants


def get_data(one_hot = False):

   # Load MNIST data, as part of TensorFlow.
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    all_data = np.concatenate((train_images, test_images), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis= 0)

    all_data = all_data.reshape((70000, 28, 28, 1))
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
    return model


def get_network():

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
    
    model.compile(loss='categorical_crossentropy',
                optimizer='RMSprop',
                metrics=['accuracy'])

    model.summary()

    return model


def train_encoders(training_percentage, filename):

    EPOCHS = constants.encoders_epochs
    stages = constants.training_stages

    (data, labels) = get_data(one_hot=True)

    total = len(data)
    step = int(total/stages)

    # Amount of testing data
    atd = total - int(total*training_percentage)

    n = 0
    stats = []
    for i in range(0, total, step):
        j = (i + atd) % total

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
        
        model = get_encoder()
        print('Encoder:')
        model.summary()

        model.fit(training_data, training_labels,
                batch_size=100,
                epochs=EPOCHS,
                verbose=2)

        test_loss, test_acc = model.evaluate(testing_data, testing_labels)
        stats.append((test_loss, test_acc))
        model.save(constants.model_filename(filename, n))
        n += 1
    
    return np.array(stats)


def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
            training_percentage, am_filling_percentage, pops):
    (data, labels) = get_data()

    total = len(data)
    step = int(total/constants.training_stages)

    # Amount of data used for training the networks
    trdata = int(total*training_percentage)

    # Amount of data to be used to fill the memories
    fldata = int(total*am_filling_percentage)

    # Amount of data used for testing memories
    tedata = total - trdata - fldata

    n = 0
    for i in range(0, total, step):
        j = (i + tedata) % total

        if j > i:
            testing_data = data[i:j]
            testing_labels = labels[i:j]
            other_data = np.concatenate((data[0:i], data[j:total]), axis=0)
            other_labels = np.concatenate((labels[0:i], labels[j:total]), axis=0)
            training_data = other_data[:trdata]
            training_labels = other_labels[:trdata]
            filling_data = other_data[trdata:]
            filling_labels = other_labels[trdata:]
        else:
            testing_data = np.concatenate((data[0:j], data[i:total]), axis=0)
            testing_labels = np.concatenate((labels[0:j], labels[i:total]), axis=0)
            training_data = data[j:j+trdata]
            training_labels = labels[j:j+trdata]
            filling_data = data[j+trdata:i]
            filling_labels = labels[j+trdata:i]
 
        # Recreate the exact same model, including its weights and the optimizer
        model = tf.keras.models.load_model(constants.model_filename(model_prefix, n))

        # Drop the last layers of the full connected neural network part.
        for i in range(pops):
            model.pop()
        
        training_features = model.predict(training_data)
        if len(filling_data) > 0:
            filling_features = model.predict(filling_data)
        else:
            r, c = training_features.shape
            filling_features = np.zeros((0, c))
        testing_features = model.predict(testing_data)

        dict = {
            constants.training_suffix: (training_data, training_features, training_labels),
            constants.filling_suffix : (filling_data, filling_features, filling_labels),
            constants.testing_suffix : (testing_data, testing_features, testing_labels)
            }

        for suffix in dict:
            data_fn = constants.data_filename(data_prefix+suffix, n)
            features_fn = constants.data_filename(features_prefix+suffix, n)
            labels_fn = constants.data_filename(labels_prefix+suffix, n)

            d, f, l = dict[suffix]
            np.save(data_fn, d)
            np.save(features_fn, f)
            np.save(labels_fn, l)

        n += 1

