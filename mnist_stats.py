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


