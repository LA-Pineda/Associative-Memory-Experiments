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
full_model_filename = 'full_model'

def model_filename(n):
    return filename(full_model_filename, n)


# Features and labels.
features_fn_prefix = 'features'
labels_fn_prefix = 'labels'

training_stages = 10
nn_training_percent = 60.0/70.0
am_training_percent = 60.0/70.0

domain = 64

n_jobs = 4
n_labels = 10
labels_per_memory = [0, 1, 2]

all_labels = list(range(n_labels))
  
precision_idx = 0
recall_idx = 1
n_measures = 2
def mean_idx(m):
    return m

def std_idx(m):
    return m+1

no_response_idx = 2
no_correct_response_idx = 3
no_correct_chosen_idx = 4
correct_response_idx = 5
mean_responses_idx = 6

n_behaviours = 7

memory_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

