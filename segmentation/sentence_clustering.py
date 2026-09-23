import os
import collections
from sklearn.cluster import KMeans
import numpy as np
import json
import time
import xml.etree.ElementTree as ET
import xml.etree.cElementTree as etree


def cluster_sentences(sentences, nb_of_clusters=200):

        kmeans = KMeans(n_clusters=nb_of_clusters)
        kmeans.fit(sentences)
        clusters = collections.defaultdict(list) 
        print(kmeans.labels_)       
        for i, label in enumerate(kmeans.labels_):
                clusters[str(label)].append(i)
        print(dict(clusters))
        return dict(clusters), kmeans.labels_


def read_steps(recipe_root,annotation_steps,emb):
    feat = []
    final_list = []
    recipe_steps = recipe_root.findall('steps_updated')
    assert( recipe_steps != None )

    recipe_steps_all = recipe_steps[0].findall('steps_updated')
    assert( recipe_steps_all != None )

    step_list = []
    for cat in recipe_steps_all:
        curr_step = cat.text.lower().strip()
        step_list.append(curr_step)

    for xi in range (0,len(step_list)):
        if annotation_steps[xi] == '-1,-1':
            continue
        else:
            feat.append(emb[xi,:])
            final_list.append(step_list[xi])

    return final_list, feat



if __name__ == "__main__":
        
        total_time = 0
        recipe_number_of_caps = dict()
        captions = []
        recipe_path = os.environ.get("TASTY_DATA_ROOT", "/mnt/data/tasty_data/")
        all_recipes = [lines.rstrip('\n') for lines in open( recipe_path + 'ALL_RECIPES.txt')]
        tot_emb = []
        writer  = open("ignore_recipes.txt","w")
        for i in range(len(all_recipes)):
                stt = time.time()
                src = recipe_path + 'ALL_RECIPES/' + all_recipes[i]
                src_xml   = recipe_path + 'ALL_RECIPES/'  + all_recipes[i]   + '/recipe.xml'
                xmlDoc_f = open(src_xml, 'r')
                src_annotation = recipe_path +'ALL_RECIPES/' +   all_recipes[i]   +  '/csvalignment.dat'
                annotation_steps =  [line.rstrip('\n') for line in open(src_annotation)]
                xmlDocData_f = xmlDoc_f.read()
                xmlDoc_f.close()
                xmlDocTree = etree.XML(xmlDocData_f)
                tree = ET.parse(src_xml)
                recipe_root = tree.getroot()
                emb = np.load(src + '/steps_embeddings.npy')
                recipe_steps, em = read_steps(recipe_root,annotation_steps,emb)
                if not em:
                    writer.write(all_recipes[i]+'\n')
                    continue
                embd = np.vstack(em)
                captions+=recipe_steps
                print("[*] {} ".format(all_recipes[i]))
                recipe_number_of_caps[all_recipes[i]] = embd.shape[0]
                tot_emb.append(embd)
        all_emb = np.vstack(tot_emb)        
        nclusters= 200
        s = time.time()
        print("len captions",len(captions))
        print("len emb",all_emb.shape)
        clusters, labels = cluster_sentences(all_emb, nclusters)
        
        print("labels are",np.unique(labels))

        with open(recipe_path + "cluster_labels_Refined.json",'w') as f:
            json.dump(clusters,f)

        print("Cluster dict saved !")
        print("Time taken to cluster {}".format(time.time()-s))
        recipe_cluster_labels = dict()
        recipe_clusters_dict = dict()

        for cl,la in clusters.items():
                recipe_clusters_dict[cl] = [captions[i] for i in la]

        with open(recipe_path + "sentences_clusters_Refined.json",'w') as f:
                json.dump(recipe_clusters_dict,f)

        prev = 0
        for recipe, num in recipe_number_of_caps.items():
                print("[*] {}".format(recipe))
                st = time.time()
                recipe_cluster_labels[recipe] = labels[prev:prev+recipe_number_of_caps[recipe]].tolist()
                print("Labels",recipe_cluster_labels[recipe])
                print("time taken {}".format(time.time()-st))
                prev += recipe_number_of_caps[recipe]
        with open(recipe_path + "recipe_cluster_labels_Refined.json",'w') as f:
                json.dump(recipe_cluster_labels,f)

        print("DONE")
