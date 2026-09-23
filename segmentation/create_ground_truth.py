
import xml.etree.ElementTree as ET
import os
import json
import xml.etree.cElementTree as etree
import sys
import os.path
import numpy as np

def read_label_frames(recipe_labels, dirnames_ii, annotation_steps, number_of_frames):
    frame_labels = np.ones(number_of_frames)*200
    for xi in range (0, len(recipe_labels)) :
        if annotation_steps[xi] == '-1,-1':
            continue
        else:
            annots = annotation_steps[xi].split(',')
            start_frame = int(annots[0] )
            end_frame = int(annots[1])
            if start_frame == 1:
                start_frame_actual = 1
            else:
                start_frame_actual = start_frame*5 - 4
            end_frame_actual = end_frame*5
            frame_labels[start_frame_actual-1:end_frame_actual] = recipe_labels[xi]

    return frame_labels


if __name__ == "__main__":


    split_path = '/mnt/data/tasty_data/'
    base_path = '/home/rishabhs/NeuralNetwork-Viterbi/'
    test_lines  = [test_zero_lines.rstrip('\n')   for test_zero_lines   in open( split_path + 'ALL_RECIPES.txt')]

    all_recipes = [all_recipes.rstrip('\n')   for all_recipes  in open( split_path + 'ALL_RECIPES.txt')]

    for kki in range( len(test_lines) ):
        src_xml   = split_path + 'ALL_RECIPES/' + test_lines[kki] + '/recipe.xml'
        xmlDoc_f = open(src_xml, 'r')
        xmlDocData_f = xmlDoc_f.read()
        xmlDoc_f.close()
        xmlDocTree = etree.XML(xmlDocData_f)

        tree = ET.parse(src_xml)
        recipe_root = tree.getroot()

        recipe_labels = json.load(open(base_path + 'recipe_cluster_labels.json','r'))

        feat = np.load(split_path + 'ALL_RECIPES/' + test_lines[kki] +'/resnet50.npy')

        number_of_frames = feat.shape[0]

        src_annotation = split_path +'ALL_RECIPES/' +   test_lines[kki]   +  '/csvalignment.dat'
        annotation_steps =  [line.rstrip('\n') for line in open(src_annotation)]

        recipe_annotations = read_label_frames(recipe_labels[test_lines[kki]], test_lines[kki], annotation_steps, number_of_frames)

        with open(base_path + 'data_tasty/groundTruth/' + test_lines[kki] +'.txt',"w") as f:
            for labl in recipe_annotations:
                     f.write(str(int(labl))+'\n')

        print(' [*] ', kki, ' ', test_lines[kki])









