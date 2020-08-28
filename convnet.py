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


def train_network():

    EPOCHS = 5

    (data, labels) = get_data(one_hot=True)

    total = len(data)
    step = int(total/constants.training_stages)

    # Amount of data used for testing
    ntd = total - int(total*constants.nn_training_percent)

    n = 0
    stats = []
    for i in range(0, total, step):
        print("Training neural network #",n)
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
        
        model = get_network()

        model.fit(training_data, training_labels,
                batch_size=100,
                epochs=EPOCHS,
                verbose=2)

        test_loss, test_acc = model.evaluate(testing_data, testing_labels)
        stats.append((test_loss, test_acc))

        model.save(constants.model_filename(n))
        n += 1
    
    return np.array(stats)


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
            training_data = np.concatenate((data[0:i], data[j:total]), axis=0)
            training_labels = np.concatenate((labels[0:i], labels[j:total]), axis=0)
        else:
            testing_data = np.concatenate((data[i:total], data[0:j]), axis=0)
            testing_labels = np.concatenate((labels[i:total], labels[0:j]), axis=0)
            training_data = data[j:i]
            training_labels = labels[j:i]
 
        # Recreate the exact same model, including its weights and the optimizer
        model = tf.keras.models.load_model(constants.model_filename(n))

        # Drop the last layers of the full connected neural network part.
        for i in range(pops):
            model.pop()
 
        ftr = model.predict(training_data)
        fte = model.predict(testing_data)

        all_features = np.concatenate((ftr,fte), axis=0)
        all_labels = np.concatenate((training_labels, testing_labels), axis=0)

        feat_filename = constants.data_filename(features_fn_prefix, n)
        labl_filename = constants.data_filename(labels_fn_prefix, n)
        np.save(feat_filename, all_features)
        np.save(labl_filename, all_labels)

        n += 1

