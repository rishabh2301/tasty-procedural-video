
import xml.etree.ElementTree as ET
import os
import json
import xml.etree.cElementTree as etree
import sys
import os.path
import numpy as np
import json


def read_transcript(recipe_labels, annotation_steps):
    
    res = []

    for xi in range (0, len(recipe_labels)) :
        if annotation_steps[xi] == '-1,-1':
            res.append(int(200))
        else:
            res.append(recipe_labels[xi])

    return res



if __name__ == "__main__":

    base_path = os.environ.get("TASTY_PROJECT_ROOT", "/home/rishabhs/NeuralNetwork-Viterbi/")
    split_path = os.environ.get("TASTY_DATA_ROOT", "/mnt/data/tasty_data/")

    all_recipes = [all_recipes.rstrip('\n')   for all_recipes  in open( base_path + 'ALL_RECIPES.txt')]

    dict_trans = dict()

    recipe_labels = json.load(open(base_path + 'recipe_cluster_labels.json','r'))

    for kki in range(len(all_recipes)):

        src_xml   = split_path + 'ALL_RECIPES/' + all_recipes[kki] + '/recipe.xml'
        xmlDoc_f = open(src_xml, 'r')
        xmlDocData_f = xmlDoc_f.read()
        xmlDoc_f.close()
        xmlDocTree = etree.XML(xmlDocData_f)

        tree = ET.parse(src_xml)
        recipe_root = tree.getroot()


        src_annotation = split_path +'ALL_RECIPES/' +   all_recipes[kki]   +  '/csvalignment.dat'
        annotation_steps =  [line.rstrip('\n') for line in open(src_annotation)]


        dict_trans[all_recipes[kki]] = read_transcript(recipe_labels[all_recipes[kki]],annotation_steps)

        print(' [*] ', kki, ' ', all_recipes[kki])

    with open(base_path + 'recipe_transcript_final.json','w') as f:
        json.dump(dict_trans,f)

    c = 0
    for vid, val in dict_trans.items():
        c+=len(val)
    print("total captions ",c)

    print("DONE")










