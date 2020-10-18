# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    LayerNormalization, Reshape, Conv2DTranspose
from tensorflow.keras.utils import to_categorical
from joblib import Parallel, delayed
import png

import constants

img_rows = 28
img_columns = 28

def get_data(one_hot = False):

   # Load MNIST data, as part of TensorFlow.
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    all_data = np.concatenate((train_images, test_images), axis=0)
    all_labels = np.concatenate((train_labels, test_labels), axis= 0)

    all_data = all_data.reshape((70000, img_columns, img_rows, 1))
    all_data = all_data.astype('float32') / 255

    if one_hot:
        # Changes labels to binary rows. Each label correspond to a column, and only
        # the column for the corresponding label is set to one.
        all_labels = to_categorical(all_labels)

    return (all_data, all_labels)


def get_encoder(input_img):

    # Convolutional Encoder
    conv_1 = Conv2D(32,kernel_size=3, activation='relu', padding='same',
        input_shape=(img_columns, img_rows, 1))(input_img)
    pool_1 = MaxPooling2D((2, 2))(conv_1)
    conv_2 = Conv2D(32,kernel_size=3, activation='relu')(pool_1)
    pool_2 = MaxPooling2D((2, 2))(conv_2)
    drop_1 = Dropout(0.4)(pool_2)
    conv_3 = Conv2D(64, kernel_size=5, activation='relu')(drop_1)
    pool_3 = MaxPooling2D((2, 2))(conv_3)
    drop_2 = Dropout(0.4)(pool_3)
    norm = LayerNormalization()(drop_2)

    # Produces an array of size equal to constants.domain.
    code = Flatten()(norm)

    return code


def get_decoder(encoded):
    dense = Dense(units=7*7*32, activation='relu', input_shape=(64, ))(encoded)
    reshape = Reshape((7, 7, 32))(dense)
    trans_1 = Conv2DTranspose(64, kernel_size=3, strides=2,
        padding='same', activation='relu')(reshape)
    drop_1 = Dropout(0.4)(trans_1)
    trans_2 = Conv2DTranspose(32, kernel_size=3, strides=2,
        padding='same', activation='relu')(drop_1)
    drop_2 = Dropout(0.4)(trans_2)
    output_img = Conv2D(1, kernel_size=3, strides=1,
        activation='sigmoid', padding='same', name='autoencoder')(drop_2)

    # Produces an image of same size and channels as originals.
    return output_img


def get_classifier(encoded):
    dense_1 = Dense(constants.domain*2, activation='relu')(encoded)
    drop = Dropout(0.4)(dense_1)
    classification = Dense(10, activation='softmax', name='classification')(drop)

    return classification


def train_networks(training_percentage, filename):

    EPOCHS = constants.model_epochs
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

        input_img = Input(shape=(img_columns, img_rows, 1))
        encoded = get_encoder(input_img)
        classified = get_classifier(encoded)
        decoded = get_decoder(encoded)
        model = Model(inputs=input_img, outputs=[classified, decoded])

        model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
                    optimizer='adam',
                    metrics='accuracy')

        model.summary()

        stats = model.fit(training_data,
                (training_labels, training_data),
                batch_size=100,
                epochs=EPOCHS,
                validation_data= (testing_data,
                    {'classification': testing_labels, 'autoencoder': testing_data}),
                verbose=2)

        model.save(constants.model_filename(filename, n))
        n += 1

    return stats.history


def store_images(original, produced, directory, prefix, stage, idx, label):
    original_filename = constants.original_image_filename(directory,
        prefix, stage, idx, label)
    produced_filename = constants.produced_image_filename(directory,
        prefix, stage, idx, label)

    pixels = original.reshape(28,28) * 255
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(original_filename)
    pixels = produced.reshape(28,28) * 255
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(produced_filename)


def store_memories(labels, produced, features, directory, prefix, stage, msize):
    (idx, label) = labels
    produced_filename = constants.produced_memory_filename(directory,
        prefix, msize, stage, idx, label)

    if np.isnan(np.sum(features)):
        pixels = np.full((28,28), 255)
    else:
        pixels = produced.reshape(28,28) * 255
    pixels = pixels.round().astype(np.uint8)
    png.from_array(pixels, 'L;8').save(produced_filename)


def obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
            training_percentage, am_filling_percentage):
    (data, labels) = get_data()

    total = len(data)
    step = int(total/constants.training_stages)

    # Amount of data used for training the networks
    trdata = int(total*training_percentage)

    # Amount of data used for testing memories
    tedata = step

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

        # Drop the autoencoder and the last layers of the full connected neural network part.
        classifier = Model(model.input, model.output[0])
        model = Model(classifier.input, classifier.layers[-4].output)
        model.summary()

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


def remember(prefix):

    for i in range(constants.training_stages):
        testing_data_filename = prefix + constants.data_name + constants.testing_suffix
        testing_data_filename = constants.data_filename(testing_data_filename, i)
        testing_features_filename = prefix + constants.features_name + constants.testing_suffix
        testing_features_filename = constants.data_filename(testing_features_filename, i)
        testing_labels_filename = prefix + constants.labels_name + constants.testing_suffix
        testing_labels_filename = constants.data_filename(testing_labels_filename, i)
        memories_filename = prefix + constants.memories_name
        memories_filename = constants.data_filename(memories_filename, i)
        labels_filename = prefix + constants.labels_name + constants.memory_suffix
        labels_filename = constants.data_filename(labels_filename, i)
        model_filename = constants.model_filename(prefix + constants.model_name, i)

        testing_data = np.load(testing_data_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)
        memories = np.load(memories_filename)
        labels = np.load(labels_filename)
        model = tf.keras.models.load_model(model_filename)

        # Drop the classifier.
        autoencoder = Model(model.input, model.output[1])
        autoencoder.summary()

        # Drop the encoder
        input_mem = Input(shape=(constants.domain, ))
        decoded = get_decoder(input_mem)
        decoder = Model(inputs=input_mem, outputs=decoded)
        decoder.summary()

        for dlayer, alayer in zip(decoder.layers[1:], autoencoder.layers[11:]):
            dlayer.set_weights(alayer.get_weights())

        produced_images = decoder.predict(testing_features)
        n = len(testing_labels)

        Parallel(n_jobs=constants.n_jobs, verbose=5)( \
            delayed(store_images)(original, produced, constants.testing_directory, prefix, i, j, label) \
                for (j, original, produced, label) in \
                    zip(range(n), testing_data, produced_images, testing_labels))

        total = len(memories)
        steps = len(constants.memory_fills)
        step_size = int(total/steps)

        for j in range(steps):
            print('Decoding memory size ' + str(j) + ' and stage ' + str(i))
            start = j*step_size
            end = start + step_size
            mem_data = memories[start:end]
            mem_labels = labels[start:end]
            produced_images = decoder.predict(mem_data)

            Parallel(n_jobs=constants.n_jobs, verbose=5)( \
                delayed(store_memories)(label, produced, features, constants.memories_directory, prefix, i, j) \
                    for (produced, features, label) in zip(produced_images, mem_data, mem_labels))
