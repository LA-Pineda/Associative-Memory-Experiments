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
import png
import tensorflow as tf
import constants

img_rows = 28
img_columns = 28

# Load MNIST data, as part of TensorFlow.
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

data = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis= 0)
data = data.astype('float32') / 255

counts_cols = np.zeros((labels.size,img_columns))
counts_rows = np.zeros((labels.size,img_rows))
for i in range(labels.size):
    image = data[i]
    counts_cols[i] = image.sum(axis=0)
    counts_rows[i] = image.sum(axis=1)

means_cols = counts_cols.mean(axis=0)
stdev_cols = counts_cols.std(axis=0)

means_rows = counts_rows.mean(axis=0)
stdev_rows = counts_rows.std(axis=0)

pixels = 255*means_cols.reshape((means_cols.size, 1)).dot(means_rows.reshape((1,means_rows.size)))
pixels = pixels.round().astype(np.uint8)
 
image_filename = constants.run_path + '/average_image.png'
png.from_array(pixels, 'L;8').save(image_filename)
 
labels = labels.reshape((labels.size, 1))

counts_cols = np.concatenate((labels, counts_cols), axis=1)
counts_rows = np.concatenate((labels, counts_rows), axis=1)

cols_filename = constants.csv_filename('counts_cols')
rows_filename = constants.csv_filename('counts_rows')

np.savetxt(cols_filename, counts_cols, delimiter=',')
np.savetxt(rows_filename, counts_rows, delimiter=',')


