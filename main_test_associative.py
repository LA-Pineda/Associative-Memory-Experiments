import sys
import gc
import argparse

import numpy as np
from joblib import Parallel, delayed
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random

import constants
import convnet
from associative import AssociativeMemory, AssociativeMemoryError


def print_error(*s):
    print('Error:', *s, file = sys.stderr)


def plot_pre_graph (pre_mean, rec_mean, ent_mean, pre_std, rec_std, ent_std, tag):
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])
    Z = [[0,0],[0,0]]
    step = 0.1
    levels = np.arange(0.0, 90 + step, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    plt.clf()

    main_step = len(constants.memory_sizes)
    plt.plot(np.arange(0, 100, main_step), pre_mean, 'r-o', label='Precision')
    plt.plot(np.arange(0, 100, main_step), rec_mean, 'b-s', label='Recall')
    plt.xlim(-0.1, 91)
    plt.ylim(0, 102)
    plt.xticks(np.arange(0, 100, 10), constants.memory_sizes)

    plt.xlabel('Range Quantization Levels')
    plt.ylabel('Percentage [%]')
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(main_average_entropy, decimals=1)]

    cbar = plt.colorbar(CS3, orientation='horizontal')
    cbar.set_ticks(np.arange(0, 100, 10))
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label('Entropy')

    plt.savefig(constants.picture_filename(tag + '-graph_l4_MEAN-{0}'.format(action)), dpi=500)


def plot_behs_graph(no_response, no_correct, no_chosen, correct, tag):

    plt.clf()
    main_step = len(constants.memory_sizes)
    xlocs = np.arange(0, 100, main_step)
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(xlocs, correct, width)
    p2 = plt.bar(xlocs, no_chosen, width, bottom=correct)
    p3 = plt.bar(xlocs, no_correct, width, bottom=no_chosen)
    p4 = plt.bar(xlocs, no_response, width, bottom=no_correct)

    plt.xlim(-0.1, 91)
    plt.ylim(0, 102)
    plt.xticks(np.arange(0, 100, 10), constants.memory_sizes)

    plt.xlabel('Range Quantization Levels')
    plt.ylabel('Labels')

    plt.legend((p4[0], p3[0], p2[0], p1[0]),\
        ('No responses', 'No correct response', 'Correct not chosen', 'Correct chosen'))
    plt.legend(loc=4)
    plt.grid(axis='y')

    plt.savefig(constants.picture_filename(tag + '-graph_behaviours_MEAN-{0}'.format(action)), dpi=500)


def get_label(memories, entropies = None):

    # Random selection
    if entropies is None:
        i = random.randrange(len(memories))
        return memories[i]
    else:
        i = memories[0] 
        entropy = entropies[i]

        for j in memories[1:]:
            if entropy > entropies[j]:
                i = j
                entropy = entropies[i]
    
    return i


def get_ams_results(midx, msize, domain, opm, trf, tef, trl, tel):
    print('Testing memory size:', msize)
    behaviour = np.zeros((5, ))
    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    trf_rounded = np.round(trf * (msize - 1) / max_value).astype(np.int16)
    tef_rounded = np.round(tef * (msize - 1) / max_value).astype(np.int16)

    nobjs = constants.n_labels
    nmems = int(nobjs/opm)

    measures = np.zeros((constants.n_measures, nobjs), dtype=np.float64)
    entropy = np.zeros((nmems, ), dtype=np.float64)

    # Create the required associative memories.
    ams = dict.fromkeys(range(nmems))
    for j in ams:
        ams[j] = AssociativeMemory(domain, msize)

    # Registration
    for features, label in zip(trf_rounded, trl):
        i = int(label/opm)
        ams[i].register(features)

    # Calculate entropies
    for j in ams:
        entropy[j] = ams[j].entropy

    # Recognition
    labels = []
    predictions = []

    remove_extra = False
    n = 0;
    response_size = 0

    for features, label in zip(tef_rounded, tel):
        correct = int(label/opm)
        labels.append(correct)
        n += 1

        memories = []
        for k in ams:
                recognized = ams[k].recognize(features)
                if recognized:
                    memories.append(k)

        response_size += len(memories)
        if len(memories) == 0:
            predictions.append(len(constants.all_labels))
            remove_extra = True

            # Register empty case
            behaviour[0] += 1
        else:
            # l = get_label(memories, entropy)
            l = get_label(memories)

            if correct in memories:
                # l = label
                if l == label:
                    # Register entropy worked.
                    behaviour[3] += 1
                else:
                    # Register entropy did not work.
                    behaviour[2] += 1
            else:
                # Register not in remembered.
                behaviour[1] += 1
            
            predictions.append(l)

    behaviour[4] = response_size * 1.0/n
    labels = np.array(labels)
    predictions = np.array(predictions)
    cm = confusion_matrix(labels, predictions)

    if remove_extra:
        (m, n) = cm.shape
        cm = np.delete(cm, m-1, 0)  
        cm = np.delete(cm, n-1, 1)

    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)

    measures[constants.precision_idx, :] = precision
    measures[constants.recall_idx, :] = recall
    
    return (midx, measures, entropy, behaviour)
    

def test_memories(training_features, training_labes, \
    testing_features, testing_labels, domain, tag, experiment):

    average_entropy = []
    stdev_entropy = []

    average_precision = []
    stdev_precision = [] 
    average_recall = []
    stdev_recall = []

    no_response = []
    no_correct_response = []
    no_correct_chosen = []
    correct_chosen = []
    total_responses = []

    for i in range(constants.n_memory_tests):
        gc.collect()

        measures_per_size = np.zeros((len(constants.memory_sizes), constants.n_labels, constants.n_measures), dtype=np.float64)
        
        lpm = constants.labels_per_memory[experiment]
        n_memories = int(constants.n_labels/lpm)

        # An entropy value per memory size and memory.
        entropies = np.zeros((len(constants.memory_sizes), n_memories), dtype=np.float64)
        behaviours = np.zeros((len(constants.memory_sizes), 5))

        print('Train the different co-domain memories -- NinM: ',experiment,' run: ',i)
        list_measures_entropies = Parallel(n_jobs=4, verbose=50)(
            delayed(get_ams_results)(midx, msize, domain, lpm, training_features, testing_features, training_labels, testing_labels) for midx, msize in enumerate(constants.memory_sizes))

        for j, measures, entropy, behaviour in list_measures_entropies:
            measures_per_size[j, :, :] = measures.T
            entropies[j, :] = entropy
            behaviours[j, :] = behaviour


        ##########################################################################################

        # Calculate precision and recall

        precision = np.zeros((len(constants.memory_sizes), constants.n_labels+2), dtype=np.float64)
        recall = np.zeros((len(constants.memory_sizes), constants.n_labels+2), dtype=np.float64)

        for j, s in enumerate(constants.memory_sizes):
            precision[j, 0:10] = measures_per_size[j, : , constants.precision_idx]
            precision[j, constants.mean_idx] = measures_per_size[j, : , constants.precision_idx].mean()
            precision[j, constants.std_idx] = measures_per_size[j, : , constants.precision_idx].std()
            recall[j, 0:10] = measures_per_size[j, : , constants.recall_idx]
            recall[j, constants.mean_idx] = measures_per_size[j, : , constants.recall_idx].mean()
            recall[j, constants.std_idx] = measures_per_size[j, : , constants.recall_idx].std()
        

        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        average_entropy.append( entropies.mean(axis=1) )
        stdev_entropy.append( entropies.std(axis=1) )

        # Average precision as percentage
        average_precision.append( precision[:, constants.mean_idx] * 100 )
        stdev_precision.append( precision[:, constants.std_idx] * 100 )

        # Average recall as percentage
        average_recall.append( recall[:, constants.mean_idx] * 100 )
        stdev_recall.append( recall[:, constants.std_idx] * 100 )

        no_response.append(behaviours[:, constants.no_response_idx])
        no_correct_response.append(behaviours[:, constants.no_correct_responde_idx])
        no_correct_chosen.append(behaviours[:, constants.no_correct_chosen_idx])
        correct_chosen.append(behaviours[:, constants.correct_chosen_idx])
        total_responses.append(behaviours[:, constants.total_responses_idx])

 
    average_precision = np.array(average_precision)
    stdev_precision = np.array(stdev_precision)
    main_average_precision =[]
    main_stdev_precision = []

    average_recall=np.array(average_recall)
    stdev_recall = np.array(stdev_recall)
    main_average_recall = []
    main_stdev_recall = []

    average_entropy=np.array(average_entropy)
    stdev_entropy=np.array(stdev_entropy)
    main_average_entropy=[]
    main_stdev_entropy=[]

    no_response = np.array(no_response)
    no_correct_response = np.array(no_correct_response)
    no_correct_chosen = np.array(no_correct_chosen)
    correct_chosen = np.array(correct_chosen)
    total_responses = np.array(total_responses)

    main_no_response = []
    main_no_correct_response = []
    main_no_correct_chosen = []
    main_correct_chosen = []
    main_total_responses = []


    for i in range(len(constants.memory_sizes)):
        main_average_precision.append( average_precision[:,i].mean() )
        main_average_recall.append( average_recall[:,i].mean() )
        main_average_entropy.append( average_entropy[:,i].mean() )

        main_stdev_precision.append( stdev_precision[:,i].mean() )
        main_stdev_recall.append( stdev_recall[:,i].mean() )
        main_stdev_entropy.append( stdev_entropy[:,i].mean() )

        main_no_response.append(no_response[:, i].mean())
        main_no_correct_response.append(no_correct_response[:, i].mean())
        main_no_correct_chosen.append(no_correct_chosen[:, i].mean())
        main_correct_chosen.append(correct_chosen[:, i].mean())
        main_total_responses.append(total_responses[:, i].mean())

    main_behaviours = [main_no_response, main_no_correct_response, \
        main_no_correct_chosen, main_correct_chosen, main_total_responses]

    np.savetxt(constants.csv_filename(tag + '-main_average_precision--{0}'.format(action)), \
        main_average_precision, delimiter=',')
    np.savetxt(constants.csv_filename(tag + '-main_average_recall--{0}'.format(action)), \
        main_average_recall, delimiter=',')
    np.savetxt(constants.csv_filename(tag + '-main_average_entropy--{0}'.format(action)), \
        main_average_entropy, delimiter=',')

    np.savetxt(constants.csv_filename(tag + '-main_stdev_precision--{0}'.format(action)), \
        main_stdev_precision, delimiter=',')
    np.savetxt(constants.csv_filename(tag + '-main_stdev_recall--{0}'.format(action)), \
        main_stdev_recall, delimiter=',')
    np.savetxt(constants.csv_filename(tag + '-main_stdev_entropy--{0}'.format(action)), \
        main_stdev_entropy, delimiter=',')

    np.savetxt(constants.csv_filename(tag + '-main_behaviours--{0}'.format(action)), \
        main_behaviours, delimiter=',')

    plot_pre_graph(main_average_precision, main_average_recall, main_average_entropy,\
        main_stdev_precision, main_stdev_recall, main_stdev_entropy, tag)

    plot_behs_graph(main_behaviours)

    print('Test complete')


##############################################################################
# Main section

TRAIN_NN = -1
GET_FEATURES = 0
FIRST_EXP = 1
SECOND_EXP = 2


def main(action):
    if action == TRAIN_NN:
        # Trains a neural network with those sections of data
        convnet.train_network()
    elif action == GET_FEATURES:
        # Generates features for the data sections using the previously generate neural network
        convnet.obtain_features()
    else:
        training_labels = np.load(constants.train_labels_filename)
        testing_labels = np.load(constants.test_labels_filename)

        training_features = np.load(constants.train_features_dense_filename)
        testing_features = np.load(constants.test_features_dense_filename)

        # The domain size, equal to the size of the output layer of the network.
        domain = constants.dense_domain
        tag = constants.dense_tag

        test_memories(training_features, training_labels, \
            testing_features, testing_labels, domain, tag, action)

        training_features = np.load(constants.train_features_conv2d_filename)
        testing_features = np.load(constants.test_features_conv2d_filename)
        domain = constants.conv2d_domain
        tag = constants.conv2d_tag

        test_memories(training_features, training_labels, \
            testing_features, testing_labels, domain, tag, action)
      




if __name__== "__main__" :

    parser = argparse.ArgumentParser(description='Associative Memory Experimenter.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', action='store_const', const=TRAIN_NN, dest='action',
                        help='train the neural network')
    group.add_argument('-f', action='store_const', const=GET_FEATURES, dest='action',
                        help='get data features using the neural network')
    group.add_argument('-e', nargs='?', dest='n', type=int, 
                        help='run the experiment with that number')

    args = parser.parse_args()
    action = args.action
    n = args.n
    
    if action is None:
        # An experiment was chosen
        if (n < FIRST_EXP) or (n > SECOND_EXP):
            print_error("There are only three experiments available, numbered 1, 2, and 3.")
            exit(1)
        main(n)
    else:
        main(action)

    
    

