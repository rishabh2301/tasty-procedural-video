#!/usr/bin/python3

import os
import numpy as np
from utils.dataset import Dataset
from utils.network import Trainer, Forwarder
from utils.viterbi import Viterbi


### read training data #########################################################
print('read data...')
base_path = os.environ.get("TASTY_DATA_ROOT", "/mnt/data/tasty_data/")
video_list = [lines.rstrip('\n') for lines in open( os.environ.get("TASTY_PROJECT_ROOT", "/home/rishabhs/NeuralNetwork-Viterbi/") + 'TRAIN_SET.txt')]
dataset = Dataset(base_path, video_list, shuffle = True)
print('done')

"""### generate path grammar for inference ########################################
paths = set()
for _, transcript in dataset:
    paths.add( ' '.join([index2label[index] for index in transcript]) )
with open('results/grammar.txt', 'w') as f:"""

### actual nn-viterbi training #################################################
decoder = Viterbi(None, None, frame_sampling = 30, max_hypotheses = np.inf) # (None, None): transcript-grammar and length-model are set for each training sequence separately, see trainer.train(...)
trainer = Trainer(decoder, dataset.input_dimension, dataset.n_classes, buffer_size = len(dataset), buffered_frame_ratio = 25)
learning_rate = 0.01

# train for 10000 iterations
for i in range(10000):
    sequence, transcript = dataset.get()
    #print("In train.py line 40 sequence is {}".format(sequence))
    #print("In train.py line 41 transcript is {}".format(transcript_p))
    loss = trainer.train(sequence, transcript, batch_size = 52, learning_rate = learning_rate)
    # print some progress information
    if (i+1) % 10 == 0:
        print('Iteration %d, loss: %f' % (i+1, loss))
    # save model every 1000 iterations
    if (i+1) % 100 == 0:
        print('save snapshot ' + str(i+1))
        network_file = 'results/network.iter-' + str(i+1) + '.net'
        length_file = 'results/lengths.iter-' + str(i+1) + '.txt'
        prior_file = 'results/prior.iter-' + str(i+1) + '.txt'
        trainer.save_model(network_file, length_file, prior_file)
    # adjust learning rate after 2500 iterations
    if (i+1) == 500:
        learning_rate = learning_rate * 0.1
