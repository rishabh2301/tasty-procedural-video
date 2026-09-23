#!/usr/bin/python

import os
import numpy as np
import multiprocessing as mp
import queue
from utils.dataset import Dataset
from utils.network import Forwarder
from utils.grammar import PathGrammar
from utils.length_model import PoissonModel
from utils.viterbi import Viterbi
import json

### helper function for parallelized Viterbi decoding ##########################    
def decode(queue, log_probs, decoder, index2label):
    while not queue.empty():
        try:
            video = queue.get(timeout = 3)
            score, labels, segments = decoder.decode( log_probs[video] )
            # save result
            print("video : ",video)
            print("score :",score)
            #print("labels :",labels)
            #print("segments :",segments)
            with open('results_3/' + video, 'w') as f:
                f.write( '### Recognized sequence: ###\n' )
                f.write( ' '.join( [index2label[s.label] for s in segments] ) + '\n' )
                f.write( '### Score: ###\n' + str(score) + '\n')
                f.write( '### Frame level recognition: ###\n')
                f.write( ' '.join( [index2label[l] for l in labels] ) + '\n' )
        except queue.Empty:
            pass


### read label2index mapping and index2label mapping ###########################
label2index = dict()
index2label = dict()
for i in range(201):
        label2index[str(int(i))] = int(i)
        index2label[int(i)] = str(int(i))

### read test data #############################################################
base_path = os.environ.get("TASTY_DATA_ROOT", "/mnt/data/tasty_data/")
video_list = [lines.rstrip('\n') for lines in open( os.environ.get("TASTY_PROJECT_ROOT", "/home/rishabhs/NeuralNetwork-Viterbi/") + 'TEST_SET.txt')]
dataset = Dataset(base_path, video_list, shuffle = False)

# load prior, length model, grammar, and network
load_iteration = 5000
log_prior = np.log( np.loadtxt('results/prior.iter-' + str(load_iteration) + '.txt') )
grammar = PathGrammar(os.environ.get("TASTY_PROJECT_ROOT", "/home/rishabhs/NeuralNetwork-Viterbi/") + "recipe_labels_cluster.json")
length_model = PoissonModel('results/lengths.iter-' + str(load_iteration) + '.txt', max_length = 2000)
forwarder = Forwarder(dataset.input_dimension, dataset.n_classes)
forwarder.load_model('results/network.iter-' + str(load_iteration) + '.net')

# parallelization
n_threads = 24

# Viterbi decoder
viterbi_decoder = Viterbi(grammar, length_model, frame_sampling = 30, max_hypotheses = np.inf)
# forward each video
log_probs = dict()
queue = mp.Queue()
for i, data in enumerate(dataset):
    sequence, _ = data
    video = list(dataset.features.keys())[i]
    queue.put(video)
    log_probs[video] = forwarder.forward(sequence) - log_prior
    log_probs[video] = log_probs[video] - np.max(log_probs[video])
# Viterbi decoding
procs = []
for i in range(n_threads):
    p = mp.Process(target = decode, args = (queue, log_probs, viterbi_decoder, index2label) )
    procs.append(p)
    p.start()
for p in procs:
    p.join()

