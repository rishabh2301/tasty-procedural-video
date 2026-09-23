#!/usr/bin/python
import numpy as np
from glob import glob
from data_process import *
import cv2
import imageio
import os
import pickle

n_lstm_steps = 80

word_counts,unk_required = build_vocab(0)
word2id,id2word = word_to_ids(word_counts,unk_required=unk_required)


def fetch_train_data(batch_size):
    VIDEO_DIR = '/home/r/rishabhs/tasty_dataset/ALL_RECIPES/'
    home_dir = '/home/r/rishabhs/tasty_dataset/s2vt_pytorch'
    pickle_path = '/home/r/rishabhs/tasty_dataset/data/tasty_S2VT.pkl'
    """Function to fetch a batch of video features, captions and caption masks
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos
                curr_masks: Mask for the pad locations in curr_caps"""



    picklefile = open(pickle_path,'rb')

    picklefile = pickle.load(picklefile)

    train = picklefile['training']

    train_names = list()
    for  vid, val in train.items():
        train_names.append(vid)

    #print('***APPENDED TRAINING NAMES***')

    cur_vid = []
    batch = np.random.choice(train_names, batch_size)
    captions = list()
    for i in range(0,batch_size):
        #print('[*] {} RECIPE : {}'.format(i,train[batch[i]]['recipe_name']))
        captions.append(batch[i])
        #print('CAPTION : {}'.format(batch[i]))
        features_dir = VIDEO_DIR + train[batch[i]]['recipe_name'] + '/resnet50.npy'
        features = np.load(features_dir)
        number_of_frames = len(train[batch[i]]['frame_indices'])
        start = train[batch[i]]['frame_indices'][0]
        end = train[batch[i]]['frame_indices'][number_of_frames-1]
        if(number_of_frames < n_lstm_steps):
            frames = features[start:end+1]
            padding = np.zeros((n_lstm_steps-number_of_frames,2048))
            frames = np.concatenate((frames,padding),axis=0)
            #print('dimension 1 is ',np.shape(frames))
        else:
            idx = np.linspace(start,end,n_lstm_steps).astype(int)
            frames = features[idx,:]
            #print('dimension 2 is ',np.shape(frames))

        #print('dimension {}'.format(np.shape(frames)))
        cur_vid.append(frames)

    cur_vid = np.array(cur_vid)
    #print('SHAPE OF CUR VID {}'.format(np.shape(cur_vid)))
    #print('caption length ',len(captions))

    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)

    return cur_vid, captions, cap_mask
# vid_size = [batch_size, 80, 2048]     caption_size = [batch_size, 80]

def fetch_val_data(batch_size):
    VIDEO_DIR = '/home/r/rishabhs/tasty_dataset/ALL_RECIPES/'
    home_dir = '/home/r/rishabhs/tasty_dataset/s2vt_pytorch'
    pickle_path = '/home/r/rishabhs/tasty_dataset/data/tasty_S2VT.pkl'
    """Function to fetch a batch of video features, captions and caption masks
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos
                curr_masks: Mask for the pad locations in curr_caps"""



    picklefile = open(pickle_path,'rb')

    picklefile = pickle.load(picklefile)

    val = picklefile['validation']

    val_names = list()
    for  vid, va in val.items():
        val_names.append(vid)

    #print('***APPENDED VALIDATION NAMES***')

    cur_vid = []
    batch = np.random.choice(val_names, batch_size)
    captions = list()
    for i in range(0,batch_size):
        #print('[*] {} RECIPE : {}'.format(i,val[batch[i]]['recipe_name']))
        captions.append(batch[i])
        #print('CAPTION : {}'.format(batch[i]))
        features_dir = VIDEO_DIR + val[batch[i]]['recipe_name'] + '/resnet50.npy'
        features = np.load(features_dir)
        number_of_frames = len(val[batch[i]]['frame_indices'])
        start = val[batch[i]]['frame_indices'][0]
        end = val[batch[i]]['frame_indices'][number_of_frames-1]
        if(number_of_frames < n_lstm_steps):
            frames = features[start:end+1]
            padding = np.zeros((n_lstm_steps-number_of_frames,2048))
            frames = np.concatenate((frames,padding),axis=0)
            #print('dimension 1 is ',np.shape(frames))
        else:
            idx = np.linspace(start,end,n_lstm_steps).astype(int)
            frames = features[idx,:]
        
            #print('dimension 2 is ',np.shape(frames))

        #print('dimension {}'.format(np.shape(frames)))
        #frames = frames[np.newaxis,]
        """if(i==0):
            cur_vid = frames
        else:
            cur_vid = np.concatenate((cur_vid,frames),axis=0)"""
        cur_vid.append(frames)
    cur_vid = np.array(cur_vid)


    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)

    return cur_vid, captions, cap_mask
# vid_size = [batch_size, 80, 2048]     caption_size = [batch_size, 80]

def fetch_val_data_orderly(idx,batch_size):
    VIDEO_DIR = '/home/r/rishabhs/tasty_dataset/ALL_RECIPES/'
    home_dir = '/home/r/rishabhs/tasty_dataset/s2vt_pytorch'
    pickle_path = '/home/r/rishabhs/tasty_dataset/data/tasty_S2VT.pkl'
    """Function to fetch a batch of video features, captions and caption masks
        Input:
                batch_size: Size of batch to load
        Output:
                curr_vids: Features of the randomly selected batch of video_files
                curr_caps: Ground truth (padded) captions for the selected videos
                curr_masks: Mask for the pad locations in curr_caps"""



    picklefile = open(pickle_path,'rb')

    picklefile = pickle.load(picklefile)

    test = picklefile['testing']

    test_names = list()
    for  vid, va in test.items():
        test_names.append(vid)

    #print('***APPENDED VALIDATION NAMES ORDERLY***')

    cur_vid = []
    batch = test_names[idx:idx+batch_size]
    captions = list()
    for i in range(0,batch_size):
        #print('[*] {} RECIPE : {}'.format(i,val[batch[i]]['recipe_name']))
        captions.append(batch[i])
        #print('CAPTION : {}'.format(batch[i]))
        features_dir = VIDEO_DIR + test[batch[i]]['recipe_name'] + '/resnet50.npy'
        features = np.load(features_dir)
        number_of_frames = len(test[batch[i]]['frame_indices'])
        start = test[batch[i]]['frame_indices'][0]
        end = test[batch[i]]['frame_indices'][number_of_frames-1]
        if(number_of_frames < n_lstm_steps):
            frames = features[start:end+1]
            padding = np.zeros((n_lstm_steps-number_of_frames,2048))
            frames = np.concatenate((frames,padding),axis=0)
            #print('dimension 1 is ',np.shape(frames))
        else:
            idx = np.linspace(start,end,n_lstm_steps).astype(int)
            frames = features[idx,:]
        
            #print('dimension 2 is ',np.shape(frames))

        #print('dimension {}'.format(np.shape(frames)))
        #frames = frames[np.newaxis,]
        """if(i==0):
            cur_vid = frames
        else:
            cur_vid = np.concatenate((cur_vid,frames),axis=0)"""
        cur_vid.append(frames)
    cur_vid = np.array(cur_vid)


    captions, cap_mask = convert_caption(captions, word2id, n_lstm_steps)

    return cur_vid, captions, cap_mask, batch
# vid_size = [batch_size, 80, 2048]     caption_size = [batch_size, 80]



def print_in_english(caption_idx):
    """Function to take a list of captions with words mapped to ids and
        print the captions after mapping word indices back to words."""
    captions_english = [[id2word[word] for word in caption] for caption in caption_idx]
    for i,caption in enumerate(captions_english):
        if '<EOS>' in caption:
            caption = caption[0:caption.index('<EOS>')]
        print(str(i+1) + ' ' + ' '.join(caption))
        print('..................................................')

"""def playVideo(video_urls):
    video = imageio.get_reader(YOUTUBE_CLIPS_DIR + video_urls[0] + '.avi','ffmpeg')
    for frame in video:
        fr = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',fr)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()"""
def write_in_english(caption_idx, typestr):
    captions_english = [[id2word[word] for word in caption] for caption in caption_idx]
    with open("/home/r/rishabhs/tasty_dataset/S2VT_tf/caption_log.txt", 'a+') as f:
        f.write(typestr+'\n')
        for i,caption in enumerate(captions_english):
            if '<EOS>' in caption:
                caption = caption[0:caption.index('<EOS>')]
            f.write(str('[ ')+str(i+1)+str(' ]') + ' ' + ' '.join(caption))
            f.write('\n')

def save_val_result(vid, captions, val_result):
    cur_dict = {}
    cur_caps = []
    captions_english = [[id2word[word] for word in caption] for caption in captions]
    for cap in captions_english:
        if '<EOS>' in cap:
            cap = cap[0:cap.index('<EOS>')]
            cap = [' '.join(cap)]
            cur_caps.append(cap)
    for i, id in enumerate(vid):
        cur_dict[id] = cur_caps[i]
    val_result = dict(val_result, **cur_dict)
    return val_result
