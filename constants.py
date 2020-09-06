# Directory where all results are stored.
run_path = './runs'
idx_digits = 3

def filename(s, idx = None, extension = ''):
    """ Returns a file name in run_path directory with a given extension and an index
    """
    if idx is None:
        return run_path + '/' + s + extension
    else:
        return run_path + '/' + s + '-' + str(idx).zfill(3) + extension


def csv_filename(s, idx = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, '.csv')


def data_filename(s, idx = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, '.npy')


def picture_filename(s, idx = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, '.png')


def model_filename(s, idx = None):
    return filename(s, idx)



# Categories prefixes.
encoder_prefix = 'encoder'
stats_encoder_prefix = 'encoder_stats'
data_prefix = 'data'
features_prefix = 'features'
labels_prefix = 'labels'
memories_prefix = 'memories'

full_prefix = 'full-'
partial_prefix = 'partial-'

# Categories suffixes.
training_suffix = '-training'
filling_suffix = '-filling'
testing_suffix = '-testing'

training_stages = 10
encoders_epochs = 10
nn_training_percent = 40.0/70.0
am_filling_percent = 20.0/70.0
# am_filling_percent = 0.1

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
ideal_memory_size = 38


