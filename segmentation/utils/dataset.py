#!/usr/bin/python3

import numpy as np
import random
import json
# self.features[video]: the feature array of the given video (dimension x frames)
# self.transcrip[video]: the transcript (as label indices) for each video
# self.input_dimension: dimension of video features
# self.n_classes: number of classes
class Dataset(object):

    def __init__(self, base_path, video_list, shuffle = False):
        self.features = dict()
        self.transcript = dict()
        self.shuffle = shuffle
        self.idx = 0
        cluster_labels = json.load(open('/home/rishabhs/NeuralNetwork-Viterbi/' + "recipe_labels_cluster.json","r"))
        # read features for each video        
        for video in video_list:
            self.features[video] = np.load(base_path +"ALL_RECIPES/" + video + '/resnet50.npy').T
            # transcript
            self.transcript[video] = cluster_labels[video]
            print("[*] {}".format(video))

        # selectors for random shuffling
        self.selectors = list(self.features.keys())
        if self.shuffle:
            random.shuffle(self.selectors)
        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0]
        #print("Input dim in dataset.py is ",self.input_dimension)
        self.n_classes = 201

    def videos(self):
        return self.features.keys()

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.selectors)
            raise StopIteration
        else:
            video = self.selectors[self.idx]
            self.idx += 1
            #print("line 52",self.transcript[video])
            return self.features[video], self.transcript[video]
    def get(self):
        try:
            return self.__next__()
        except StopIteration:
            return self.get()

