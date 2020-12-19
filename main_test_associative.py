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

import sys
import gc
import argparse
import gettext

import numpy as np
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import json

import constants
import convnet
from associative import AssociativeMemory

# Translation
gettext.install('ame', localedir=None, codeset=None, names=None)

def print_error(*s):
    print('Error:', *s, file = sys.stderr)


def plot_pre_graph (pre_mean, rec_mean, ent_mean, pre_std, rec_std, ent_std, \
    tag = '', action=None, xlabels = constants.memory_sizes, xtitle = None, \
        ytitle = None):

    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])
    Z = [[0,0],[0,0]]
    step = 0.1
    levels = np.arange(0.0, 90 + step, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    plt.clf()

    main_step = 100.0/len(xlabels)
    plt.errorbar(np.arange(0, 100, main_step), pre_mean, fmt='r-o', yerr=pre_std, label=_('Precision'))
    plt.errorbar(np.arange(0, 100, main_step), rec_mean, fmt='b-s', yerr=rec_std, label=_('Recall'))
    plt.xlim(0, 90)
    plt.ylim(0, 102)
    plt.xticks(np.arange(0, 100, main_step), xlabels)

    if xtitle is None:
        xtitle = _('Range Quantization Levels')
    if ytitle is None: 
        ytitle = _('Percentage')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.legend(loc=4)
    plt.grid(True)

    entropy_labels = [str(e) for e in np.around(ent_mean, decimals=1)]

    cbar = plt.colorbar(CS3, orientation='horizontal')
    cbar.set_ticks(np.arange(0, 100, main_step))
    cbar.ax.set_xticklabels(entropy_labels)
    cbar.set_label(_('Entropy'))

    plt.savefig(constants.picture_filename(tag + 'graph_prse_MEAN-{0}'.format(action) + _('-english')), dpi=500)


def plot_size_graph (response_size, size_stdev, action=None):
    plt.clf()

    main_step = len(constants.memory_sizes)
    plt.errorbar(np.arange(0, 100, main_step), response_size, fmt='g-D', yerr=size_stdev, label=_('Average number of responses'))
    plt.xlim(0, 90)
    plt.ylim(0, 10)
    plt.xticks(np.arange(0, 100, 10), constants.memory_sizes)

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Size'))
    plt.legend(loc=1)
    plt.grid(True)

    plt.savefig(constants.picture_filename('graph_size_MEAN-{0}'.format(action) + _('-english')), dpi=500)


def plot_behs_graph(no_response, no_correct, no_chosen, correct, action=None):

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

    plt.bar(xlocs, correct, width, label=_('Correct response chosen'))
    cumm = np.array(correct)
    plt.bar(xlocs, no_chosen,  width, bottom=cumm, label=_('Correct response not chosen'))
    cumm += np.array(no_chosen)
    plt.bar(xlocs, no_correct, width, bottom=cumm, label=_('No correct response'))
    cumm += np.array(no_correct)
    plt.bar(xlocs, no_response, width, bottom=cumm, label=_('No responses'))

    plt.xlim(-5, 95)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 100, 10), constants.memory_sizes)

    plt.xlabel(_('Range Quantization Levels'))
    plt.ylabel(_('Labels'))

    plt.legend(loc=0)
    plt.grid(axis='y')
    plt.savefig(constants.picture_filename('graph_behaviours_MEAN-{0}'.format(action) + _('-english')), dpi=500)


def plot_features_graph(domain, means, stdevs):
    ymin = np.PINF
    ymax = np.NINF
    for i in constants.all_labels:
        n = (means[i] - stdevs[i]).min()
        x = (means[i] + stdevs[i]).max()
        ymin = ymin if ymin < n else n
        ymax = ymax if ymax > x else x

    main_step = 100.0 / domain
    xrange = np.arange(0, 100, main_step)
    fmts = ['r-h', 'b-*', 'g-s', 'y-x', 'm-d', 'c-h', 'r-*', 'b--s', 'g--x', 'y--d']

    for i in constants.all_labels:
        plt.clf()
        plt.figure(figsize=(12,5))

        plt.errorbar(xrange, means[i], fmt=fmts[i], yerr=stdevs[i], label=str(i))
        plt.xlim(0, 100)
        plt.ylim(ymin, ymax)
        plt.xticks(xrange, labels='')

        plt.xlabel(_('Features'))
        plt.ylabel(_('Values'))
        plt.legend(loc='right')
        plt.grid(True)

        filename = constants.features_name + '-' + str(i) + _('-english')
        plt.savefig(constants.picture_filename(filename), dpi=500)



def get_label(memories, entropies = None):

    # Random selection
    if entropies is None:
        i = random.atddrange(len(memories))
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

    min_value = trf.min()
    other_value = tef.min()
    min_value = min_value if min_value < other_value else other_value

    trf_rounded = np.round((trf-min_value) * (msize - 1) / (max_value-min_value)).astype(np.int16)
    tef_rounded = np.round((tef-min_value) * (msize - 1) / (max_value-min_value)).astype(np.int16)

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

    suffix = constants.filling_suffix

    for i in range(constants.training_stages):
        gc.collect()

        training_features_filename = constants.features_name + suffix        
        training_features_filename = constants.data_filename(training_features_filename, i)
        training_labels_filename = constants.labels_name + suffix        
        training_labels_filename = constants.data_filename(training_labels_filename, i)

        suffix = constants.testing_suffix
        testing_features_filename = constants.features_name + suffix        
        testing_features_filename = constants.data_filename(testing_features_filename, i)
        testing_labels_filename = constants.labels_name + suffix        
        testing_labels_filename = constants.data_filename(testing_labels_filename, i)

        training_features = np.load(training_features_filename)
        training_labels = np.load(training_labels_filename)
        testing_features = np.load(testing_features_filename)
        testing_labels = np.load(testing_labels_filename)

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
    main_total_responses_stdev = []


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
        main_total_responses_stdev.append(total_responses[:, i].std())

    main_behaviours = [main_no_response, main_no_correct_response, \
        main_no_correct_chosen, main_correct_chosen, main_total_responses]

    np.savetxt(constants.csv_filename('main_average_precision--{0}'.format(experiment)), \
        main_average_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_average_precision--{0}'.format(experiment)), \
        main_all_average_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_recall--{0}'.format(experiment)), \
        main_average_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_average_recall--{0}'.format(experiment)), \
        main_all_average_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_entropy--{0}'.format(experiment)), \
        main_average_entropy, delimiter=',')

    np.savetxt(constants.csv_filename('main_stdev_precision--{0}'.format(experiment)), \
        main_stdev_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_stdev_precision--{0}'.format(experiment)), \
        main_all_stdev_precision, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_recall--{0}'.format(experiment)), \
        main_stdev_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_all_stdev_recall--{0}'.format(experiment)), \
        main_all_stdev_recall, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_entropy--{0}'.format(experiment)), \
        main_stdev_entropy, delimiter=',')

    np.savetxt(constants.csv_filename('main_behaviours--{0}'.format(experiment)), \
        main_behaviours, delimiter=',')

    plot_pre_graph(main_average_precision, main_average_recall, main_average_entropy,\
        main_stdev_precision, main_stdev_recall, main_stdev_entropy, action=experiment)

    plot_pre_graph(main_all_average_precision, main_all_average_recall, \
        main_average_entropy, main_all_stdev_precision, main_all_stdev_recall,\
            main_stdev_entropy, 'overall', action=experiment)

    plot_size_graph(main_total_responses, main_total_responses_stdev, action=experiment)

    plot_behs_graph(main_no_response, main_no_correct_response, main_no_correct_chosen,\
        main_correct_chosen, action=experiment)

    print('Test complete')


def get_recalls(ams, msize, domain, min, max, trf, trl, tef, tel, idx):

    trf_rounded = np.round((trf - min) * (msize - 1) / (max - min)).astype(np.int16)
    tef_rounded = np.round((tef - min) * (msize - 1) / (max - min)).astype(np.int16)

    n_mems = constants.n_labels
    measures = np.zeros((constants.n_measures, n_mems), dtype=np.float64)
    entropy = np.zeros((n_mems, ), dtype=np.float64)

    # Confusion matrix for calculating precision and recall per memory.
    cms = np.zeros((n_mems, 2, 2))
    TP = (0,0)
    FP = (0,1)
    FN = (1,0)
    TN = (1,1)

    # Registration
    for features, label in zip(trf_rounded, trl):
        ams[label].register(features)

    # Calculate entropies
    for j in ams:
        entropy[j] = ams[j].entropy

    all_recalls = []
    total_recalls = 0
    # Recover memories
    for n, features, label in zip(range(len(tef_rounded)), tef_rounded, tel):
        memories = []
        recalls ={}

        for k in ams:
            recall = ams[k].recall(features)
            recognized = not (ams[k].is_undefined(recall))

            # For calculation of per memory precision and recall
            if (k == label) and recognized:
                cms[k][TP] += 1
            elif k == label:
                cms[k][FN] += 1
            elif recognized:
                cms[k][FP] += 1
            else:
                cms[k][TN] += 1

            if recognized:
                memories.append(k)
                recalls[k] = recall

        if (len(memories) == 0):
            # Register empty case
            undefined = np.full(domain, ams[0].undefined)
            all_recalls.append((n, label, undefined))
        else:
            l = get_label(memories, entropy)
            features = recalls[l]*(max-min)*1.0/(msize-1) + min
            all_recalls.append((n, label, features))
            total_recalls += 1

    for i in range(n_mems):
        positives = cms[i][TP] + cms[i][FP]
        measures[constants.precision_idx,i] = cms[i][TP] / positives if positives else 1.0
        measures[constants.recall_idx,i] = cms[i][TP] /(cms[i][TP] + cms[i][FN])    

    return all_recalls, measures, entropy, total_recalls
    

def get_means(d):
    n = len(d.keys())
    means = np.zeros((n, ))
    for k in d:
        rows = np.array(d[k])
        mean = rows.mean()
        means[k] = mean

    return means


def get_stdev(d):
    n = len(d.keys())
    stdevs = np.zeros((n, ))
    for k in d:
        rows = np.array(d[k])
        std = rows.std()
        stdevs[k] = std

    return stdevs    
    

def test_recalling_fold(n_memories, mem_size, domain, experiment, fold):
    # Create the required associative memories.
    ams = dict.fromkeys(range(n_memories))
    for j in ams:
        ams[j] = AssociativeMemory(domain, mem_size)

    suffix = constants.filling_suffix
    training_features_filename = constants.features_name + suffix        
    training_features_filename = constants.data_filename(training_features_filename, fold)
    training_labels_filename = constants.labels_name + suffix        
    training_labels_filename = constants.data_filename(training_labels_filename, fold)

    suffix = constants.testing_suffix
    testing_features_filename = constants.features_name + suffix        
    testing_features_filename = constants.data_filename(testing_features_filename, fold)
    testing_labels_filename = constants.labels_name + suffix        
    testing_labels_filename = constants.data_filename(testing_labels_filename, fold)

    training_features = np.load(training_features_filename)
    training_labels = np.load(training_labels_filename)
    testing_features = np.load(testing_features_filename)
    testing_labels = np.load(testing_labels_filename)

    training_max = training_features.max()
    testing_max = testing_features.max()
    training_min = training_features.min()
    testing_min = testing_features.min()

    maximum = training_max if training_max > testing_max else testing_max
    minimum = training_min if training_min < testing_min else testing_min

    total = len(training_features)
    percents = np.array(constants.memory_fills)
    steps = np.round(total*percents/100.0).astype(int)

    stage_recalls = []
    stage_entropies = {}
    stage_mprecision = {}
    stage_mrecall = {}
    total_recalls = []

    i = 0
    k = 0
    for j in range(len(steps)):
        k += steps[j]
        features = training_features[i:k]
        labels = training_labels[i:k]

        recalls, measures, entropies, total_recall = get_recalls(ams, mem_size, domain, minimum, maximum, \
            features, labels, testing_features, testing_labels, fold)

        # A list of tuples (position, label, features)
        stage_recalls += recalls

        # An array with entropies per memory
        stage_entropies[j] = entropies

        # An array with precision per memory
        stage_mprecision[j] = measures[constants.precision_idx,:]

        # An array with recall per memory
        stage_mrecall[j] = measures[constants.recall_idx,:]

        # Total number of recalls per step
        total_recalls.append(total_recall)

        i = k

    print(total_recalls)
    return  fold, stage_recalls, stage_entropies, stage_mprecision, stage_mrecall, np.array(total_recalls)


def test_recalling(domain, mem_size, experiment):
    n_memories = constants.n_labels

    all_recalls = {}
    all_entropies = {}
    all_mprecision = {}
    all_mrecall = {}
    total_recalls = np.zeros((constants.training_stages, len(constants.memory_fills)))

    xlabels = constants.memory_fills
    list_results = Parallel(n_jobs=constants.n_jobs, verbose=50)(
        delayed(test_recalling_fold)(n_memories, mem_size, domain, experiment, fold) \
            for fold in range(constants.training_stages))

    for fold, stage_recalls, stage_entropies, stage_mprecision, stage_mrecall, total_recall in list_results:
        all_recalls[fold] = stage_recalls
        for msize in stage_entropies:
            all_entropies[msize] = all_entropies[msize] + [stage_entropies[msize]] \
                if msize in all_entropies.keys() else [stage_entropies[msize]]
            all_mprecision[msize] = all_mprecision[msize] + [stage_mprecision[msize]] \
                if msize in all_mprecision.keys() else [stage_mprecision[msize]]
            all_mrecall[msize] = all_mrecall[msize] + [stage_mrecall[msize]] \
                if msize in all_mrecall.keys() else [stage_mrecall[msize]]
            total_recalls[fold] = total_recall

    for fold in all_recalls:
        list_tups = all_recalls[fold]
        tags = []
        memories = []
        for (idx, label, features) in list_tups:
            tags.append((idx, label))
            memories.append(np.array(features))
        
        tags = np.array(tags)
        memories = np.array(memories)
        memories_filename = constants.data_filename(constants.memories_name, fold)
        np.save(memories_filename, memories)
        tags_filename = constants.labels_name + constants.memory_suffix
        tags_filename = constants.data_filename(tags_filename, fold)
        np.save(tags_filename, tags)

    main_avrge_entropies = get_means(all_entropies)
    main_stdev_entropies = get_stdev(all_entropies)
    main_avrge_mprecision = get_means(all_mprecision)
    main_stdev_mprecision = get_stdev(all_mprecision)
    main_avrge_mrecall = get_means(all_mrecall)
    main_stdev_mrecall = get_stdev(all_mrecall)
    
    np.savetxt(constants.csv_filename('main_average_precision',experiment), \
        main_avrge_mprecision, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_recall',experiment), \
        main_avrge_mrecall, delimiter=',')
    np.savetxt(constants.csv_filename('main_average_entropy',experiment), \
        main_avrge_entropies, delimiter=',')

    np.savetxt(constants.csv_filename('main_stdev_precision',experiment), \
        main_stdev_mprecision, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_recall',experiment), \
        main_stdev_mrecall, delimiter=',')
    np.savetxt(constants.csv_filename('main_stdev_entropy',experiment), \
        main_stdev_entropies, delimiter=',')
    np.savetxt(constants.csv_filename('main_total_recalls',experiment), \
        total_recalls, delimiter=',')


    plot_pre_graph(main_avrge_mprecision*100, main_avrge_mrecall*100, main_avrge_entropies,\
        main_stdev_mprecision*100, main_stdev_mrecall*100, main_stdev_entropies, 'recall-', \
            xlabels = xlabels, xtitle = _('Percentage of memory corpus'), action=experiment)

    print('Test complete')


def get_all_data(prefix, domain):
    data = None

    for stage in range(constants.training_stages):
        filename = constants.data_filename(prefix, stage)
        if data is None:
            data = np.load(filename)
        else:
            newdata = np.load(filename)
            data = np.concatenate((data, newdata), axis=0)

    return data

def characterize_features(domain):
    features_prefix = constants.features_name
    ff_filename = features_prefix + constants.filling_suffix
    tf_filename = features_prefix + constants.testing_suffix

    labels_prefix = constants.labels_name
    fl_filename = labels_prefix + constants.filling_suffix
    tl_filename = labels_prefix + constants.testing_suffix

    features = np.concatenate((get_all_data(ff_filename, domain),
        get_all_data(tf_filename, domain)), axis=0)
    
    labels = np.concatenate((get_all_data(fl_filename, 1),
        get_all_data(tl_filename, 1)), axis=0)

    d = {}
    for i in constants.all_labels:
        d[i] = []

    for (i, feats) in zip(labels, features):
        d[i].append(feats)

    means = {}
    stdevs = {}
    for i in constants.all_labels:
        d[i] = np.array(d[i])
        means[i] = np.mean(d[i], axis=0)
        stdevs[i] = np.std(d[i], axis=0)

    plot_features_graph(domain, means, stdevs)
    
    

    
##############################################################################
# Main section

CHARACTERIZE = -2
TRAIN_NN = -1
GET_FEATURES = 0
FIRST_EXP = 1
SECOND_EXP = 2
THIRD_EXP = 3
FOURTH_EXP = 4

MIN_EXPERIMENT = 1
MAX_EXPERIMENT = 4

def main(action):

    if (action == TRAIN_NN):
        # Trains a neural network with those sections of data
        training_percentage = constants.nn_training_percent
        model_prefix = constants.model_name
        stats_prefix = constants.stats_model_name

        history = convnet.train_networks(training_percentage, model_prefix)

        stats = {}
        stats['history'] = []
        for h in history:
            stats['history'].append(h.history)

        with open(constants.json_filename(stats_prefix), 'w') as outfile:
            json.dump(stats, outfile)
    elif (action == GET_FEATURES):
        # Generates features for the data sections using the previously generate neural network
        training_percentage = constants.nn_training_percent
        am_filling_percentage = constants.am_filling_percent
        model_prefix = constants.model_name
        features_prefix = constants.features_name
        labels_prefix = constants.labels_name
        data_prefix = constants.data_name

        convnet.obtain_features(model_prefix, features_prefix, labels_prefix, data_prefix,
            training_percentage, am_filling_percentage)
    elif action == CHARACTERIZE:
        # The domain size, equal to the size of the output layer of the network.
        characterize_features(constants.domain)
    elif (action == FIRST_EXP) or (action == SECOND_EXP):
        # The domain size, equal to the size of the output layer of the network.
        test_memories(constants.domain, action)
    elif (action == THIRD_EXP):
        test_recalling(constants.domain, constants.partial_ideal_memory_size, action)
    elif (action == FOURTH_EXP):
        convnet.remember(action)


if __name__== "__main__" :

    parser = argparse.ArgumentParser(description='Associative Memory Experimenter.')
    parser.add_argument('-l', nargs='?', dest='lang', choices=['en', 'es'], default='en',
                        help='choose between English (en) or Spanish (es) labels for graphs.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n', action='store_const', const=TRAIN_NN, dest='action',
                        help='train the neural networks, separating NN and AM training data (Separate Data NN).')
    group.add_argument('-f', action='store_const', const=GET_FEATURES, dest='action',
                        help='get data features using the separate data neural networks.')
    group.add_argument('-c', action='store_const', const=CHARACTERIZE, dest='action',
                        help='characterize the features from partial data neural networks by class.')
    group.add_argument('-e', nargs='?', dest='n', type=int, 
                        help='run the experiment with that number, using separate data neural networks.')

    args = parser.parse_args()
    action = args.action
    lang = args.lang
    n = args.n
    
    if lang == 'es':
        es = gettext.translation('ame', localedir='locale', languages=['es'])
        es.install()
 
    if action is None:
        # An experiment was chosen
        if (n < MIN_EXPERIMENT) or (n > MAX_EXPERIMENT):
            print_error("There are only {1} experiments available, numbered consecutively from from {0}."
                .format(MIN_EXPERIMENT, MAX_EXPERIMENT))
            exit(1)
        else:
            main(n)
    else:
        main(action)

    
    
