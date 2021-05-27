# Copyright [2021] Luis Alberto Pineda Cortés,
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
import json
import numpy as np
from matplotlib import pyplot as plt

# Keys for data
LOSS = 'loss'
C_LOSS = 'classification_loss'
A_LOSS = 'autoencoder_loss'
C_ACCURACY = 'classification_accuracy'
A_ACCURACY = 'autoencoder_accuracy'
VAL = 'val_'



def plot(a_measure, b_measure, a_label, b_label, nn, epoch):
    fig = plt.figure()
    x = np.arange(0,epoch)
    plt.errorbar(x, a_measure[:epoch], fmt='b-o', label=a_label)
    plt.errorbar(x, b_measure[:epoch], fmt='r--s', label=b_label)
    plt.legend(loc=0)
    plt.suptitle(f'Neural net No. {nn}')
    plt.show()
    

def compare_loss(bigger_loss, smaller_loss, epoch):
    if (len(bigger_loss) < epoch) or (len(smaller_loss) < epoch):
        print('Sequences are sorter')
        sys.exit(1)
    holds = 0.0
    for i in range(epoch):
        if bigger_loss[i] < smaller_loss[i]:
            holds += 1.0
    return holds/float(epoch)


def compare_accuracy(smaller_acc, bigger_acc, epoch):
    if (len(smaller_acc) < epoch) or (len(bigger_acc) < epoch):
        print('Sequences are sorter')
        sys.exit(1)
    holds = 0.0
    for i in range(epoch):
        if smaller_acc[i] > bigger_acc[i]:
            holds += 1.0 
    return holds/float(epoch)


def training_stats(data, epoch):
    """ Analyse neural nets training data. 
        
        Training stats data is a list of dictionaries with the full
        set of keys declared above.
    """

    a = {LOSS: [], C_LOSS: [], A_LOSS: [], C_ACCURACY: [], A_ACCURACY: []}

    n = 0
    for d in data:
        a[LOSS].append(compare_loss(d[LOSS], d[VAL+LOSS], epoch))
        plot(d[LOSS], d[VAL+LOSS], LOSS, VAL+LOSS,n,epoch)
        a[C_LOSS].append(compare_loss(d[C_LOSS], d[VAL+C_LOSS], epoch))
        plot(d[C_LOSS], d[VAL+C_LOSS], C_LOSS, VAL+C_LOSS,n,epoch)
        a[A_LOSS].append(compare_loss(d[A_LOSS], d[VAL+A_LOSS], epoch))
        plot(d[A_LOSS], d[VAL+A_LOSS], A_LOSS, VAL+A_LOSS,n,epoch)
        a[C_ACCURACY].append(compare_accuracy(d[C_ACCURACY], d[VAL+C_ACCURACY], epoch))
        plot(d[C_ACCURACY], d[VAL+C_ACCURACY], C_ACCURACY, VAL+C_ACCURACY,n,epoch)
        a[A_ACCURACY].append(compare_accuracy(d[A_ACCURACY], d[VAL+A_ACCURACY], epoch))
        plot(d[A_ACCURACY], d[VAL+A_ACCURACY], A_ACCURACY, VAL+A_ACCURACY,n,epoch)
        n += 1
    return a

if __name__== "__main__" :
    if len(sys.argv) != 2:
        print('You only need to provide an epoch number.')
        sys.exit(1)

    epoch = int(sys.argv[1])
    
    history = {}
    with open('runs/model_stats.json') as json_file:
        history = json.load(json_file)
    history = history['history']

    # Now, history contains a list with the statistics from the neural nets.
    # Odd elements have statistics from training and validation, while
    # even elements have statistics from testing.

    training = []
    testing = []

    odd = True
    for s in history:
        if odd:
            training.append(s)
        else:
            testing.append(s)
        odd = not odd

    ts = training_stats(training, epoch)
    print(ts)
    print(testing)
    # testing_stats(testing)