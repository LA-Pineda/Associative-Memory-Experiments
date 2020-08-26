import sys
import gc
import argparse

import numpy as np
from joblib import Parallel, delayed
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

import constants
import convnet
from associative import AssociativeMemory, AssociativeMemoryError


def print_error(*s):
    print('Error:', *s, file = sys.stderr)


def plot_pre_graph (pre_mean, rec_mean, ent_mean, pre_std, rec_std, ent_std, tag=''):
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])
    Z = [[0,0],[0,0]]
    step = 0.1
    levels = np.arange(0.0, 90 + step, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    plt.clf()

    main_step = len(constants.memory_sizes)
    plt.errorbar(np.arange(0, 100, main_step), pre_mean, fmt='r-o', yerr=pre_std, label='Precision')
    plt.errorbar(np.arange(0, 100, main_step), rec_mean, fmt='b-s', yerr=rec_std, label='Recall')
    plt.xlim(0, 90)
    plt.ylim(0, 102)
    plt.xticks(np.arange(0, 100, 10), constants.memory_sizes)

    plt.xlabel('Range Quantization Levels')
    plt.ylabel('Percentage [%]')
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cbar = plt.colorbar(CS3, orientation='horizontal')
    cbar.set_ticks(np.arange(0, 100, 10))
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label('Entropy')

    plt.savefig(constants.picture_filename(tag + 'graph_l4_MEAN-{0}'.format(action)), dpi=500)


def plot_size_graph (response_size):
    plt.clf()

    main_step = len(constants.memory_sizes)
    plt.plot(np.arange(0, 100, main_step), response_size, 'g-D', label='Average number of responses')
    plt.xlim(0, 90)
    plt.ylim(0, 10)
    plt.xticks(np.arange(0, 100, 10), constants.memory_sizes)

    plt.xlabel('Range Quantization Levels')
    plt.ylabel('Size')
    plt.legend(loc=1)
    plt.grid(True)

    plt.savefig(constants.picture_filename('graph_size_MEAN-{0}'.format(action)), dpi=500)


def plot_behs_graph(no_response, no_correct, no_chosen, correct):

    for i in range(len(no_response)):
        total = (no_response[i] + no_correct[i] + no_chosen[i] + correct[i])/100.0
        no_response[i] /= total
        no_correct[i] /= total
        no_chosen[i] /= total
        correct[i] /= total

    plt.clf()
    main_step = len(constants.memory_sizes)
    xlocs = np.arange(0, 100, main_step)
    width = 5       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(xlocs, correct, width, label='Correct response chosen')
    cumm = np.array(correct)
    p2 = plt.bar(xlocs, no_chosen,  width, bottom=cumm, label='Correct response not chosen')
    cumm += np.array(no_chosen)
    p3 = plt.bar(xlocs, no_correct, width, bottom=cumm, label='No correct response')
    cumm += np.array(no_correct)
    p4 = plt.bar(xlocs, no_response, width, bottom=cumm, label='No responses')

    plt.xlim(-5, 95)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 100, 10), constants.memory_sizes)

    plt.xlabel('Range Quantization Levels')
    plt.ylabel('Labels')

    plt.legend(loc=0)
    plt.grid(axis='y')

    plt.savefig(constants.picture_filename('graph_behaviours_MEAN-{0}'.format(action)), dpi=500)


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


def get_ams_results(midx, msize, domain, lpm, trf, tef, trl, tel):
    print('Testing memory size:', msize)


    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    trf_rounded = np.round(trf * (msize - 1) / max_value).astype(np.int16)
    tef_rounded = np.round(tef * (msize - 1) / max_value).astype(np.int16)

    n_labels = constants.n_labels
    nmems = int(n_labels/lpm)

    measures = np.zeros((constants.n_measures, nmems), dtype=np.float64)
    entropy = np.zeros((nmems, ), dtype=np.float64)
    behaviour = np.zeros((constants.n_behaviours, ))

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((nmems, 2, 2))
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)

    # Create the required associative memories.
    ams = dict.fromkeys(range(nmems))
    for j in ams:
        ams[j] = AssociativeMemory(domain, msize)

    # Registration
    for features, label in zip(trf_rounded, trl):
        i = int(label/lpm)
        ams[i].register(features)

    # Calculate entropies
    for j in ams:
        entropy[j] = ams[j].entropy

    # Recognition
    response_size = 0
    n = len(tef_rounded)

    for features, label in zip(tef_rounded, tel):
        correct = int(label/lpm)

        memories = []
        for k in ams:
            recognized = ams[k].recognize(features)

            # For calculation of per memory precision and recall
            if (k == correct) and recognized:
                cms[k][TP] += 1
            elif k == correct:
                cms[k][FN] += 1
            elif recognized:
                cms[k][FP] += 1
            else:
                cms[k][TN] += 1

            # For calculation of behaviours, including overall precision and recall.
            if recognized:
                memories.append(k)
 
        response_size += len(memories)
        if len(memories) == 0:
            # Register empty case
            behaviour[constants.no_response_idx] += 1
        elif not (correct in memories):
            behaviour[constants.no_correct_response_idx] += 1
        else:
            l = get_label(memories, entropy)
            if l != correct:
                behaviour[constants.no_correct_chosen_idx] += 1
            else:
                behaviour[constants.correct_response_idx] += 1

    behaviour[constants.mean_responses_idx] = response_size /float(len(tef_rounded))
    all_responses = len(tef_rounded) - behaviour[constants.no_response_idx]
    all_precision = (behaviour[constants.correct_response_idx])/float(all_responses)
    all_recall = (behaviour[constants.correct_response_idx])/float(len(tef_rounded))

    behaviour[constants.precision_idx] = all_precision
    behaviour[constants.recall_idx] = all_recall

    for i in range(nmems):
        measures[constants.precision_idx,i] = cms[i][TP] /(cms[i][TP] + cms[i][FP])
        measures[constants.recall_idx,i] = cms[i][TP] /(cms[i][TP] + cms[i][FN])
   
    return (midx, measures, entropy, behaviour)
    

def test_memories(domain, experiment):

    average_entropy = []
    stdev_entropy = []

    average_precision = []
    stdev_precision = [] 
    average_recall = []
    stdev_recall = []

    all_precision = []
    all_recall = []

    no_response = []
    no_correct_response = []
    no_correct_chosen = []
    correct_chosen = []
    total_responses = []

    labels_x_memory = constants.labels_per_memory[experiment]
    n_memories = int(constants.n_labels/labels_x_memory)

    for i in range(constants.training_stages):
        gc.collect()

        feat_filename = constants.data_filename(constants.features_fn_prefix, i)
        labl_filename = constants.data_filename(constants.labels_fn_prefix, i)

        data = np.load(feat_filename)
        labels = np.load(labl_filename)
        j = int(len(data)*constants.am_training_percent)

        training_features = data[:j]
        training_labels = labels[:j]
        testing_features = data[j:]
        testing_labels = labels[j:]

        measures_per_size = np.zeros((len(constants.memory_sizes), \
            n_memories, constants.n_measures), dtype=np.float64)

        # An entropy value per memory size and memory.
        entropies = np.zeros((len(constants.memory_sizes), n_memories), dtype=np.float64)
        behaviours = np.zeros((len(constants.memory_sizes), constants.n_behaviours))

        print('Train the different co-domain memories -- NxM: ',experiment,' run: ',i)
        # Processes running in parallel.
        list_measures_entropies = Parallel(n_jobs=constants.n_jobs, verbose=50)(
            delayed(get_ams_results)(midx, msize, domain, labels_x_memory, \
                training_features, testing_features, training_labels, testing_labels) \
                    for midx, msize in enumerate(constants.memory_sizes))

        for j, measures, entropy, behaviour in list_measures_entropies:
            measures_per_size[j, :, :] = measures.T
            entropies[j, :] = entropy
            behaviours[j, :] = behaviour


        ##########################################################################################

        # Calculate precision and recall

        precision = np.zeros((len(constants.memory_sizes), n_memories+2), dtype=np.float64)
        recall = np.zeros((len(constants.memory_sizes), n_memories+2), dtype=np.float64)

        for j, s in enumerate(constants.memory_sizes):
            precision[j, 0:n_memories] = measures_per_size[j, : , constants.precision_idx]
            precision[j, constants.mean_idx(n_memories)] = measures_per_size[j, : , constants.precision_idx].mean()
            precision[j, constants.std_idx(n_memories)] = measures_per_size[j, : , constants.precision_idx].std()
            recall[j, 0:n_memories] = measures_per_size[j, : , constants.recall_idx]
            recall[j, constants.mean_idx(n_memories)] = measures_per_size[j, : , constants.recall_idx].mean()
            recall[j, constants.std_idx(n_memories)] = measures_per_size[j, : , constants.recall_idx].std()
        

        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        average_entropy.append( entropies.mean(axis=1) )
        stdev_entropy.append( entropies.std(axis=1) )

        # Average precision as percentage
        average_precision.append( precision[:, constants.mean_idx(n_memories)] * 100 )
        stdev_precision.append( precision[:, constants.std_idx(n_memories)] * 100 )

        # Average recall as percentage
        average_recall.append( recall[:, constants.mean_idx(n_memories)] * 100 )
        stdev_recall.append( recall[:, constants.std_idx(n_memories)] * 100 )

        all_precision.append(behaviours[:, constants.precision_idx] * 100)
        all_recall.append(behaviours[:, constants.recall_idx] * 100)

        no_response.append(behaviours[:, constants.no_response_idx])
        no_correct_response.append(behaviours[:, constants.no_correct_response_idx])
        no_correct_chosen.append(behaviours[:, constants.no_correct_chosen_idx])
        correct_chosen.append(behaviours[:, constants.correct_response_idx])
        total_responses.append(behaviours[:, constants.mean_responses_idx])

 
    average_precision = np.array(average_precision)
    stdev_precision = np.array(stdev_precision)
    main_average_precision =[]
    main_stdev_precision = []

    average_recall=np.array(average_recall)
    stdev_recall = np.array(stdev_recall)
    main_average_recall = []
    main_stdev_recall = []

    all_precision = np.array(all_precision)
    main_all_average_precision = []
    main_all_stdev_precision = []

    all_recall = np.array(all_recall)
    main_all_average_recall = []
    main_all_stdev_recall = []

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

        main_all_average_precision.append(all_precision[:, i].mean())
        main_all_stdev_precision.append(all_precision[:, i].std())
        main_all_average_recall.append(all_recall[:, i].mean())
        main_all_stdev_recall.append(all_recall[:, i].std())

        main_no_response.append(no_response[:, i].mean())
        main_no_correct_response.append(no_correct_response[:, i].mean())
        main_no_correct_chosen.append(no_correct_chosen[:, i].mean())
        main_correct_chosen.append(correct_chosen[:, i].mean())
        main_total_responses.append(total_responses[:, i].mean())

    main_behaviours = [main_no_response, main_no_correct_response, \
        main_no_correct_chosen, main_correct_chosen, main_total_responses]

    np.savetxt(constants.csv_filename('main_average_precision--{0}'.format(action)), \
        main_average_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_average_precision--{0}'.format(action)), \
        main_all_average_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_recall--{0}'.format(action)), \
        main_average_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_average_recall--{0}'.format(action)), \
        main_all_average_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_entropy--{0}'.format(action)), \
        main_average_entropy, delimiter=',')

    np.savetxt(constants.csv_filename('main_stdev_precision--{0}'.format(action)), \
        main_stdev_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_stdev_precision--{0}'.format(action)), \
        main_all_stdev_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_recall--{0}'.format(action)), \
        main_stdev_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_stdev_recall--{0}'.format(action)), \
        main_all_stdev_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_entropy--{0}'.format(action)), \
        main_stdev_entropy, delimiter=',')

    np.savetxt(constants.csv_filename('main_behaviours--{0}'.format(action)), \
        main_behaviours, delimiter=',')

    plot_pre_graph(main_average_precision, main_average_recall, main_average_entropy,\
        main_stdev_precision, main_stdev_recall, main_stdev_entropy)

    plot_pre_graph(main_all_average_precision, main_all_average_recall, \
        main_average_entropy, main_all_stdev_precision, main_all_stdev_recall,\
            main_stdev_entropy, 'overall')

    plot_size_graph(main_total_responses)

    plot_behs_graph(main_no_response, main_no_correct_response, main_no_correct_chosen,\
        main_correct_chosen)

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
        convnet.obtain_features(constants.features_fn_prefix, constants.labels_fn_prefix, 3)
    else:
        # The domain size, equal to the size of the output layer of the network.
        domain = constants.domain
        test_memories(domain, action)


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

    
    

