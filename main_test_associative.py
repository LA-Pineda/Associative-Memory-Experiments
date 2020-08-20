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

average_entropy = []
average_precision = []
average_recall = []


def print_error(*s):
    print('Error:', *s, file = sys.stderr)


def get_label(memories, entropies):

    # return random.randrange(len(memories))

    i = memories[0] 
    entropy = entropies[i]

    for j in memories[1:]:
        if entropy > entropies[j]:
            i = j
            entropy = entropies[i]
    
    return i


def get_ams_results(midx, msize, domain, opm, trf, tef, trl, tel):
    # Round the values
    max_value = trf.max()
    other_value = tef.max()
    max_value = max_value if max_value > other_value else other_value

    trf_rounded = np.round(trf * (msize - 1) / max_value).astype(np.int16)
    tef_rounded = np.round(tef * (msize - 1) / max_value).astype(np.int16)

    nobjs = constants.n_objects
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
    cms = np.zeros((nmems, 2, 2))
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)
 
    for features, label in zip(tef_rounded, tel):
        correct = int(label/opm)
        for k in ams:
            recognized = ams[k].recognize(features)
            if (k == correct) and recognized:
                cms[k][TP] += 1
            elif k == correct:
                cms[k][FN] += 1
            elif recognized:
                cms[k][FP] += 1
            else:
                cms[k][TN] += 1
 
    measures = np.zeros((constants.n_measures, nmems))

    for i in range(nmems):
        measures[constants.precision_idx,i] = cms[i][TP] /(cms[i][TP] + cms[i][FP])
        measures[constants.recall_idx,i] = cms[i][TP] /(cms[i][TP] + cms[i][FN])
   
    return (midx, measures, entropy)
    

def test_memories(experiment):

    for i in range(constants.n_memory_tests):
        gc.collect()

        # Load the features.
        training_features = np.load(constants.train_features_filename)
        testing_features = np.load(constants.test_features_filename)

        training_labels = np.load(constants.train_labels_filename)
        testing_labels = np.load(constants.test_labels_filename)
      
        # The domain size, equal to the size of the output layer of the network.
        domain = constants.domain

        tables = np.zeros((len(constants.memory_sizes), constants.n_objects, constants.n_measures), dtype=np.float64)
        
        opm = constants.objects_per_memory[experiment]
        n_memories = int(constants.n_objects/opm)

        # An entropy value per memory size and memory.
        entropies = np.zeros((len(constants.memory_sizes), n_memories), dtype=np.float64)

        print('Train the different co-domain memories -- NinM: ',experiment,' run: ',i)
        list_tables_entropies = Parallel(n_jobs=4, verbose=50)(
            delayed(get_ams_results)(midx, msize, domain, opm, training_features, testing_features, training_labels, testing_labels) for midx, msize in enumerate(constants.memory_sizes))

        for j, table, entropy in list_tables_entropies:
            tables[j, :, :] = table.T
            entropies[j, :] = entropy


        ##########################################################################################

        # Calculate precision and recall

        precision = np.zeros((len(constants.memory_sizes), 11, 1), dtype=np.float64)
        recall = np.zeros((len(constants.memory_sizes), 11, 1), dtype=np.float64)

        for j, s in enumerate(constants.memory_sizes):
            precision[j, 0:10, 0] = tables[j, : , constants.precision_idx]
            precision[j, 10, 0] = tables[j, : , constants.precision_idx].mean()
            recall[j, 0:10, 0] = tables[j, : , constants.recall_idx]
            recall[j, 10, 0] = tables[j, : , constants.recall_idx].mean()
        

        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        average_entropy.append( entropies.mean(axis=1) )

        # Average precision as percentage
        average_precision.append( precision[:, 10, :] * 100 )

        # Average recall as percentage
        average_recall.append( recall[:, 10, :] * 100 )
        
        np.save(constants.data_filename('average_precision'), average_precision)
        np.save(constants.data_filename('average_recall'), average_recall)
        np.save(constants.data_filename('average_entropy'), average_entropy)
        
        print('avg precision: ',average_precision[i])
        print('avg recall: ',average_recall[i])
        print('avg entropy: ',average_entropy[i])


        # Plot of precision and recall with entropies

        print('Plot of precision and recall with entropies-----{0}'.format(i))

        # Setting up a colormap that's a simple transition
        cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])

        # Using contourf to provide my colorbar info, then clearing the figure
        Z = [[0,0],[0,0]]
        step = 0.1
        levels = np.arange(0.0, 90 + step, step)
        CS3 = plt.contourf(Z, levels, cmap=cmap)

        plt.clf()


        plt.plot(np.arange(0, 100, len(constants.memory_sizes)), average_precision[i], 'r-o', label='Precision')
        plt.plot(np.arange(0, 100, len(constants.memory_sizes)), average_recall[i], 'b-s', label='Recall')
        plt.xlim(-0.1, 91)
        plt.ylim(0, 102)
        plt.xticks(np.arange(0, 100, 10), constants.memory_sizes)

        plt.xlabel('Range Quantization Levels')
        plt.ylabel('Percentage [%]')
        plt.legend(loc=4)
        plt.grid(True)

        entropy_labels = [str(e) for e in np.around(average_entropy[i], decimals=1)]

        cbar = plt.colorbar(CS3, orientation='horizontal')
        cbar.set_ticks(np.arange(0, 100, 10))
        cbar.ax.set_xticklabels(entropy_labels)
        cbar.set_label('Entropy')

        plt.savefig(constants.picture_filename('graph_l4_{0}_{1}'.format(experiment,i)), dpi=500)
        print('Iteration {0} complete'.format(i))
        #Uncomment the following line for plot at runtime
        plt.show()

        return average_precision, average_recall, average_entropy


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
        average_precision, average_recall, average_entropy = test_memories(action)

        ######################
        # Plot the final graph

        average_precision=np.array(average_precision)
        main_average_precision=[]

        average_recall=np.array(average_recall)
        main_average_recall=[]

        average_entropy=np.array(average_entropy)
        main_average_entropy=[]

        for i in range(10):
            main_average_precision.append( average_precision[:,i].mean() )
            main_average_recall.append( average_recall[:,i].mean() )
            main_average_entropy.append( average_entropy[:,i].mean() )
            
        print('main avg precision: ',main_average_precision)
        print('main avg recall: ',main_average_recall)
        print('main avg entropy: ',main_average_entropy)

        np.savetxt(constants.csv_filename('main_average_precision--{0}'.format(action)), main_average_precision, delimiter=',')
        np.savetxt(constants.csv_filename('main_average_recall--{0}'.format(action)), main_average_recall, delimiter=',')
        np.savetxt(constants.csv_filename('main_average_entropy--{0}'.format(action)), main_average_entropy, delimiter=',')

        cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])
        Z = [[0,0],[0,0]]
        step = 0.1
        levels = np.arange(0.0, 90 + step, step)
        CS3 = plt.contourf(Z, levels, cmap=cmap)

        plt.clf()

        plt.plot(np.arange(0, 100, 10), main_average_precision, 'r-o', label='Precision')
        plt.plot(np.arange(0, 100, 10), main_average_recall, 'b-s', label='Recall')
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

        plt.savefig(constants.picture_filename('graph_l4_MEAN-{0}'.format(action)), dpi=500)
        print('Test complete')



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

    
    

