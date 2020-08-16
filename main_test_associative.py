import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import cm
import matplotlib as mpl

import constants
import convnet
from associative import AssociativeMemory, AssociativeMemoryError


average_entropy = []
average_precision = []
average_recall = []


def print_error(*s):
    print('Error:', *s, file = sys.stderr)


def file_name_by_extension(s, e, i = None):
    """ Returns a file name in run_path directory with a given extension
    """

    if i is None:
        return run_path + '/' + s + e
    else:
        return run_path + '/' + s + '-' + str(i) + e


def data_file_name(s, i = None):
    """ Returns a file name for data(i) in run_path directory
    """

    return file_name_by_extesion(s, '.npy', i)


def picture_file_name(s, i = None):
    """ Returns a file name for picture(i) in run_path directory
    """

    return file_name_by_extesion(s, '.png', i)


def csv_file_name(s, i = None):
    """ Returns a file name for csv(i) in run_path directory
    """

    return file_name_by_extesion(s, '.csv', i)


def get_ams_results1(i, s, domain, train_X, test_X, trY, teY):
        table = np.zeros((10, 5), dtype=np.float64)
        entropy = np.zeros((10, ), dtype=np.float64)
        ams = dict.fromkeys(range(10))
        #print(str(ams))
        for j in ams:
            # Create the memories with domain 's'
            ams[j] = AssociativeMemory(domain, s)
        # Round the values
        if train_X.max()>test_X.max():
            valMax=train_X.max()
        else:
            valMax=test_X.max()
        train_X_around = np.round(train_X * (s - 1) / valMax).astype(np.int16)
        test_X_around = np.round(test_X * (s - 1) / valMax).astype(np.int16)
        # Abstraction
        for x, y in zip(train_X_around, trY):
            ams[y].abstract(x, input_range=s)
        # Calculate entropies
        for j in ams:
            #print(j)
            entropy[j] = ams[j].entropy
        # Reduction
        for x, y in zip(test_X_around, teY):
            table[y, 0] += 1
            for k in ams:
                try:
                    ams[k].reduce(x, input_range=s)
                    if k == y:
                        table[y, 1] += 1
                    else:
                        table[y, 2] += 1
                    # confusion_mat[k, y] += 1
                except AssociativeMemoryError:
                    if k != y:
                        table[y, 3] += 1
                    else:
                        table[y, 4] += 1
        return (i, table, entropy)

    
def get_ams_results2(i, s, domain, train_X, test_X, trY, teY):
        table = np.zeros((10, 5), dtype=np.float64)
        entropy = np.zeros((5, ), dtype=np.float64)
        ams = dict.fromkeys(range(5))
        #print(str(ams))
        for j in ams:
            # Create the memories with domain 's'
            ams[j] = AssociativeMemory(domain, s)
        # Round the values
        if train_X.max()>test_X.max():
            valMax=train_X.max()
        else:
            valMax=test_X.max()
        train_X_around = np.round(train_X * (s - 1) / valMax).astype(np.int16)
        test_X_around = np.round(test_X * (s - 1) / valMax).astype(np.int16)
        # Abstraction
        for x, y in zip(train_X_around, trY):
            yy=y%5
            ams[yy].abstract(x, input_range=s)
        # Calculate entropies
        for j in ams:
            #print(j)
            entropy[j] = ams[j].entropy
        # Reduction
        for x, y in zip(test_X_around, teY):
            yy=y%5
            table[y, 0] += 1
            for k in ams:
                try:
                    ams[k].reduce(x, input_range=s)
                    if k == yy:
                        table[y, 1] += 1
                    else:
                        table[y, 2] += 1
                    # confusion_mat[k, y] += 1
                except AssociativeMemoryError:
                    if k != yy:
                        table[y, 3] += 1
                    else:
                        table[y, 4] += 1
        return (i, table, entropy)
    

def test_memories(data, labels, experiment):

    for i in range(N_RUNS):
        # Load the features.
        training_features = np.load(data_file_name('train_features_l4'))
        testing_features = np.load(data_file_name('test_features_l4.npy'))

        trX, teX, trY, teY = get_data_and_labels(i, N_RUNS, data, labels)
        
        # The ranges of all the memories that will be trained.
        memory_sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

        # The domain size, equal to the size of the output layer of the network.
        domain = 625

        # Maximum value of the features in the train set
        max_val = training_features.max()

        # Train the different co-domain memories

        # For each memory size, and each digit, it stores:
        # 0.- Total count
        # 1.- Able to reduce and it is the same digit
        # 2.- Able to reduce and it is not the same digit 
        # 3.- Not able to reduce and it is not the same digit
        # 4.- Not able to reduce and it is the same digit
        tables = np.zeros((len(memory_sizes), 10, 5), dtype=np.float64)
        entropies = np.zeros((len(memory_sizes), int(10/experiment)), dtype=np.float64)

        print('Train the different co-domain memories -- NinM: ',experiment,' -----',i)
        if sel == 1:
            list_tables_entropies = Parallel(n_jobs=8, verbose=50)(
                delayed(get_ams_results1)(j, s, domain, training_features, testing_features, trY, teY) for j, s in enumerate(memory_sizes))
        elif sel == 2:
            list_tables_entropies = Parallel(n_jobs=8, verbose=50)(
                delayed(get_ams_results2)(j, s, domain, training_features, testing_features, trY, teY) for j, s in enumerate(memory_sizes))

        for j, table, entropy in list_tables_entropies:
            tables[j, :, :] = table
            entropies[j, :] = entropy


        ##########################################################################################

        # Calculate precision and recall

        print('Calculate precision and recall')
        precision = np.zeros((len(memory_sizes), 11, 1), dtype=np.float64)
        recall = np.zeros((len(memory_sizes), 11, 1), dtype=np.float64)

        for j, s in enumerate(memory_sizes):
            # Proportion of correct reductions among all reductions.
            prec_aux = tables[j, :, 1] / (tables[j, :, 1] + tables[j, :, 2])

            # Proportion of correct reductions among all cases.
            recall_aux = tables[j, :, 1] / tables[j, :, 0]


            precision[j, 0:10, 0] = prec_aux[:]
            precision[j, 10, 0] = prec_aux.mean()
            recall[j, 0:10, 0] = recall_aux[:]
            recall[j, 10, 0] = recall_aux.mean()
        

        ###################################################################3##
        # Measures by memory size

        # Average entropy among al digits.
        average_entropy.append( entropies.mean(axis=1) )

        # Average precision as percentage
        average_precision.append( precision[:, 10, :] * 100 )

        # Average recall as percentage
        average_recall.append( recall[:, 10, :] * 100 )
        
        np.save(data_file_name('average_precision'), average_precision)
        np.save(data_file_name('average_recall'), average_recall)
        np.save(data_file_name('average_entropy'), average_entropy)
        
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


        plt.plot(np.arange(0, 100, 10), average_precision[i], 'r-o', label='Precision')
        plt.plot(np.arange(0, 100, 10), average_recall[i], 'b-s', label='Recall')
        plt.xlim(-0.1, 91)
        plt.ylim(0, 102)
        plt.xticks(np.arange(0, 100, 10), memory_sizes)

        plt.xlabel('Range Quantization Levels')
        plt.ylabel('Percentage [%]')
        plt.legend(loc=4)
        plt.grid(True)

        entropy_labels = [str(e) for e in np.around(average_entropy[i], decimals=1)]

        cbar = plt.colorbar(CS3, orientation='horizontal')
        cbar.set_ticks(np.arange(0, 100, 10))
        cbar.ax.set_xticklabels(entropy_labels)
        cbar.set_label('Entropy')

        plt.savefig(picture_file_name('graph_l4_{0}_{1}'.format(experiment,i)), dpi=500)
        print('Iteration {0} complete'.format(i))
        #Uncomment the following line for plot at runtime
        #plt.show()


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
        test_memories()

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

        np.savetxt(csv_file_name('main_average_precision--{0}'.format(action)), main_average_precision, delimiter=',')
        np.savetxt(csv_file_name('main_average_recall--{0}'.format(action)), main_average_recall, delimiter=',')
        np.savetxt(csv_file_name('main_average_entropy--{0}'.format(action)), main_average_entropy, delimiter=',')

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
        plt.xticks(np.arange(0, 100, 10), memory_sizes)

        plt.xlabel('Range Quantization Levels')
        plt.ylabel('Percentage [%]')
        plt.legend(loc=4)
        plt.grid(True)

        entropy_labels = [str(e) for e in np.around(main_average_entropy, decimals=1)]

        cbar = plt.colorbar(CS3, orientation='horizontal')
        cbar.set_ticks(np.arange(0, 100, 10))
        cbar.ax.set_xticklabels(entropy_labels)
        cbar.set_label('Entropy')

        plt.savefig(picture_file_name('graph_l4_MEAN-{0}'.format(action)), dpi=500)
        print('Test complete')



if __name__== "__main__" :

    parser = argparse.ArgumentParser(description='Associative Memory Experimenter.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', action='store_const', const=TRAIN_NN, dest='action',
                        help='train the neural network')
    group.add_argument('-f', action='store_const', const=GET_FEATURES, dest='action',
                        help='get data features using the neural network')
    group.add_argument('-e', dest='n', type=int, default=argparse.SUPPRESS,
                        help='run the experiment with that number')

    args = parser.parse_args()
    action = args.action

  
    if (action == None):
        # An experiment was chosen
        if (n < FIRST_EXP) or (n > SECOND_EXP):
            print_error("There are only three experiments available, numbered 1, 2, and 3.")
            exit(1)
        else:
            action = args.n
    
    main(action)

    

