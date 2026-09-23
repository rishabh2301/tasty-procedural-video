# -*- coding: utf-8 -*-
"""
Created on Thu May 9 2019

@author: rishabhs
"""

import os, torch
from glob import glob
import numpy as np
from torch.autograd import Variable
from PIL import Image  
from torchvision import models, transforms
import torch.nn as nn
import time

data_dir = 'ALL_RECIPES'

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
 
#ResNet Standards 
def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([transforms.Resize((224, 224)),
        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])             
    
    img = Image.open(img_path)
    img = transform(img)
    
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.data.numpy()
    y= y.reshape((1,2048))

    np.save(saved_path,y)
    return y
    
if __name__ == '__main__':

    recipe_names_fid = open("noresnet.txt", "r")
    recipe_names = recipe_names_fid.readlines()
    recipe_names_fid.close()

    resnet50 = models.resnet50(pretrained=True)
    modules=list(resnet50.children())[:-1] # delete the last fc layer. 
    resnet50.eval()
    resnet50_feature_extractor=nn.Sequential(*modules)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    use_gpu = torch.cuda.is_available()
    

    for i in range(0,len(recipe_names)): 
        start = time.time()
        recipe_names[i] = recipe_names[i].replace('\n','')
        curr_recipe = recipe_names[i]
        file_glob = os.path.join(data_dir,curr_recipe,'frames','*')
        image_list = sorted(glob(file_glob))
        features_dir = os.path.join(data_dir,curr_recipe,'FeaturesResnet50')
        if not os.path.exists(features_dir):
            os.mkdir(features_dir)
        
        count = 0
        fx_path = os.path.join(features_dir,str(count))
        feat = extractor(image_list[0], fx_path, resnet50_feature_extractor, use_gpu)
        count+=1

        for x_path in image_list[1:]:
            fx_path = os.path.join(features_dir,str(count))
            featr = extractor(x_path, fx_path, resnet50_feature_extractor, use_gpu)
            feat  = np.concatenate((feat,featr),axis=0)
            count+=1
            if (count%100==0):
                print('saved {} features'.format(count))

        np.save(os.path.join(data_dir,curr_recipe,'resnet50'),feat)

        end = time.time()
        print('[*] ',i,' ',recipe_names[i],' time taken = {}'.format(end-start))

