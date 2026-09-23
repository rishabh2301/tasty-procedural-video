# -*- coding: utf-8 -*-
"""
Created on Thu May 14 2019
    
@author: rishabhs
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import sys

scriptpath = "/home/r/rishabhs/tasty_dataset/i3d_edit.py"
sys.path.append(os.path.abspath(scriptpath))

import numpy as np
import tensorflow as tf
import i3d_edit as i3d
import argparse
import cv2
from glob import glob
import numpy as np
import torch 
from torch.autograd import Variable


config = tf.ConfigProto()
config.gpu_options.allow_growth = True


data_dir = 'ALL_RECIPES'


_IMAGE_SIZE = 224
_SAMPLE_VIDEO_FRAMES = 17 

_CHECKPOINT_PATHS = {
    'rgb': 'i3d/data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'i3d/data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'i3d/data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'i3d/data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'i3d/data/label_map.txt'
_LABEL_MAP_PATH_600 = 'i3d/data/label_map_600.txt'


  # Endpoints of the model in order, from the output of the i3d.InceptionI3d,
  # all the endpoints up to a designated `final_endpoint` are returned in a dictionary as the
  # second return value.
VALID_ENDPOINTS = (         #just for reference from original i3d.py
      'Conv3d_1a_7x7',
      'MaxPool3d_2a_3x3',
      'Conv3d_2b_1x1',
      'Conv3d_2c_3x3',
      'MaxPool3d_3a_3x3',
      'Mixed_3b',
      'Mixed_3c',
      'MaxPool3d_4a_3x3',
      'Mixed_4b',
      'Mixed_4c',
      'Mixed_4d',
      'Mixed_4e',
      'Mixed_4f',
      'MaxPool3d_5a_2x2',
      'Mixed_5b',
      'Mixed_5c',
      'Logits',
      'Predictions')




def run():
  
  imagenet_pretrained = True

  NUM_CLASSES = 400

  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))


  with tf.variable_scope('RGB',reuse=tf.AUTO_REUSE):
    rgb_model = i3d.InceptionI3d(NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)


    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    model_logits = rgb_logits

  
    recipe_names_fid = open("ALL_RECIPES.txt", "r")
    recipe_names = recipe_names_fid.readlines()
    recipe_names_fid.close()
    recipe_count = 0
    file = open('missingframes.txt',"w")

    for indu in range(0,len(recipe_names)):
        curr_recipe = recipe_names[indu].replace('\n', '')
        recipe_count +=1
        print(curr_recipe)
        print('recipe count is {}'.format(recipe_count))
        file_glob = os.path.join(data_dir,curr_recipe,'frames','*')
        image_list = sorted(glob(file_glob))
        if (len(image_list)==0):
          file.write(curr_recipe+"\n")
          print('missing')
        else:
          f = os.path.join(image_list[0][0:-17],'i3dFeatures400')
          if not os.path.exists(f):
            os.mkdir(f)
          images = cv2.imread(image_list[0])
          images = cv2.resize(images,(224,224))
          images = torch.from_numpy(images).cuda()
          images = images.unsqueeze(0).cuda()
          for x_path in image_list[1:]:
            image = cv2.imread(x_path)
            image = cv2.resize(image,(224,224))
            image = torch.from_numpy(image).cuda()
            image = image.unsqueeze(0).cuda()
            images = torch.cat((images,image),dim=0).cuda()
          t, h, w, c = images.size()
          totalframes = t
          print('original shape is :',images.size())
          first_pad = images[1:9,:,:,:].cuda()
          first_pad = torch.flip(first_pad,[0]).cuda()
          images = torch.cat((first_pad,images),dim=0).cuda()
          t, h, w, c = images.size()
          print('shape after first_pad :',images.size())
          second_pad = images[t-9:t-1,:,:,:].cuda()
          second_pad = torch.flip(second_pad,[0]).cuda()
          images = torch.cat((images,second_pad),dim=0).cuda()
          images = images.data.cpu().numpy()
          t, h, w, c = images.shape
          print('shape after second_pad :',images.shape)
        

          j=0
          i=8
          count = 0
          with tf.Session(config=config) as sess:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            while count<totalframes:
              feed_dict = {}
              feed_dict[rgb_input] = images[np.newaxis,j:i+9,:]
              out_logits = sess.run([model_logits],feed_dict=feed_dict)
              out_logits = out_logits[0]
              features_dir = os.path.join(image_list[0][0:-17],'i3dFeatures400',str(count))
              np.save(features_dir, out_logits.reshape((1,400)))
              i = i+1
              j = j+1
              count = count+1
              if(count%100==0):
                print('saved {} features'.format(count))
            print('saving {} i3d400 features for recipe : '.format(count),curr_recipe)
            images = None

         


if __name__ == '__main__':
    run() 
