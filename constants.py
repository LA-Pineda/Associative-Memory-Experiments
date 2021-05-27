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

import os

# Directory where all results are stored.
run_path = './runs'
idx_digits = 3


def occlusion_suffix(occlusion):
    return '' if occlusion is None else '-occ_' + str(int(round(occlusion*100))).zfill(3)


def bars_type_suffix(bars_type):
    return '' if bars_type is None else '-bar_' + str(bars_type).zfill(3)


def tolerance_suffix(tolerance):
    return '' if not tolerance else '-tol_' + str(tolerance).zfill(3)


def filename(s, idx = None, occlusion = None, bars_type = None, tolerance = 0, extension = ''):
    """ Returns a file name in run_path directory with a given extension and an index
    """
    # Create target directory & all intermediate directories if don't exists
    try:
        os.makedirs(run_path)
        print("Directory " , run_path ,  " created ")
    except FileExistsError:
        pass

    if idx is None:
        return run_path + '/' + s + extension
    else:
        return run_path + '/' + s + '-' + str(idx).zfill(3) \
            + occlusion_suffix(occlusion) \
            + bars_type_suffix(bars_type) + tolerance_suffix(tolerance) + extension



def json_filename(s):
    """ Returns a file name for a JSON file in run_path directory
    """
    return filename(s,  extension = '.json')


def csv_filename(s, idx = None, occlusion = None, bars_type = None, tolerance = 0):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, occlusion, bars_type, tolerance, '.csv')


def data_filename(s, idx = None):
    """ Returns a file name for csv(i) in run_path directory
    """
    return filename(s, idx, extension='.npy')


def picture_filename(s, idx = None, occlusion = None, bars_type = None, tolerance = 0):
    """ Returns a file name for a graph.
    """
    return filename(s, idx, occlusion, bars_type, tolerance, '.svg')


def model_filename(s, idx = None):
    return filename(s, idx)


def image_filename(dir, stage, idx, label, suffix = ''):
    image_path = run_path + '/images/' + dir + '/' + 'stage_' + str(stage) + '/'

    try:
        os.makedirs(image_path)
    except FileExistsError:
        pass

    image_path += str(label) + '_' + str(idx).zfill(5)  + suffix + '.png'
    return image_path


testing_path = 'test'
memories_path = 'memories'


def testing_directory(i, occlusion = None, bars_type = None):
    return testing_path + '-' + str(i).zfill(3) \
        + occlusion_suffix(occlusion) + bars_type_suffix(bars_type)


def memories_directory(i, occlusion = None, bars_type = None, tolerance = 0):
    return memories_path + '-' + str(i).zfill(3) \
        + occlusion_suffix(occlusion) \
        + bars_type_suffix(bars_type) + tolerance_suffix(tolerance)


def memory_filename(dir, msize, stage, idx, label):
    # Remove '-'
    image_path = run_path + '/images/' + dir + '/' + 'stage_' + str(stage) + '/'
    image_path += 'msize_' + str(msize) + '/'

    try:
        os.makedirs(image_path)
        print("Directory " , image_path ,  " created ")
    except FileExistsError:
        pass

    image_path += str(label) + '_' + str(idx).zfill(5) + '.png'
    return image_path


original_suffix = '-original'


def original_image_filename(dir, stage, idx, label):
    return image_filename(dir, stage, idx, label, original_suffix)


def produced_image_filename(dir, stage, idx, label):
    return image_filename(dir, stage, idx, label)


def produced_memory_filename(dir, msize, stage, idx, label):
    return memory_filename(dir, msize, stage, idx, label)


features_prefix = 'features'
experiment_defaul_suffix = ''
experiment_suffix = ['', '', '', '', '',
    '-top_hidden', '-bottom_hidden', '-left_hidden', '-right_hidden',
    '-ver_bars', '-hor_bars']


def features_name(i = -1, occlusion = None, bars_type = None):
    if i  < 0:
        return features_prefix
    else:
        return features_prefix + experiment_suffix[i] \
            + occlusion_suffix(occlusion) + bars_type_suffix(bars_type)


memories_prefix = 'memories'

def memories_name(i = -1, occlusion = None, bars_type = None, tolerance = 0):
    mem_name = memories_prefix
    if i  >= 0:
        mem_name += experiment_suffix[i] + occlusion_suffix(occlusion) \
            + bars_type_suffix(bars_type) + tolerance_suffix(tolerance)
    return mem_name

# Categories prefixes.
model_name = 'model'
stats_model_name = 'model_stats'
data_name = 'data'
labels_name = 'labels'

# Categories suffixes.
training_suffix = '-training'
filling_suffix = '-filling'
testing_suffix = '-testing'
memory_suffix = '-memories'

training_stages = 10

am_testing_percent = (100 / training_stages) / 100
nn_training_percent = 0.57  # 0.10 + 0.57 = 0.67
am_filling_percent = 0.33   # 0.67 + 0.33 = 1.0

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
memory_fills = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
partial_ideal_memory_size = 32
full_ideal_memory_size = 64

CHARACTERIZE = -2
TRAIN_NN = -1
GET_FEATURES = 0
EXP_1 = 1
EXP_2 = 2
EXP_3 = 3
EXP_4 = 4
EXP_5 = 5
EXP_6 = 6
EXP_7 = 7
EXP_8 = 8
EXP_9 = 9
EXP_10 = 10


MIN_EXPERIMENT = 1
MAX_EXPERIMENT = 10

bar_patterns = [[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0],
            [1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]]
N_BARS = len(bar_patterns)
