# Directory where all results are stored.
run_path = './runs'
idx_digits = 3

def filename(s, idx = None, e = ''):
    """ Returns a file name in run_path directory with a given extension and an index
    """
    if idx is None:
        return run_path + '/' + s + e
    else:
        return run_path + '/' + s + '-' + str(idx).zfill(3) + e


def csv_filename(s, i = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, i, '.csv')


def data_filename(s, i = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, i, '.npy')


def picture_filename(s, i = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, i, '.png')


# Models (neural networks).
full_model_filename = filename('full_model')
features_model_dense_filename = filename('features_dense-model')
features_model_conv2d_filename = filename('features_conv2d-model')

# Features and labels.
train_features_dense_filename = filename('train_dense-features.npy')
train_features_conv2d_filename = filename('train_conv2d-features.npy')
test_features_dense_filename = filename('test_dense-features.npy')
test_features_conv2d_filename = filename('test_conv2d-features.npy')

train_labels_filename = filename('train-labels.npy')
test_labels_filename = filename('test-labels.npy')

dense_domain = 512
conv2d_domain = 512

dense_tag = 'dense'
conv2d_tag = 'conv2d'

n_memory_tests = 1
n_labels = 10
labels_per_memory = [0, 1, 2]

all_labels = list(range(n_labels))
  
precision_idx = 0
recall_idx = 1
n_measures = 2
mean_idx = 10
std_idx = 11

no_response_idx = 2
no_correct_response_idx = 3
no_correct_chosen_idx = 4
correct_response_idx = 5
mean_responses_idx = 6

n_behaviours = 7

memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
