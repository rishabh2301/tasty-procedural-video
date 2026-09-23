from module import *
import torch
from torch import nn
from utils import *


EPOCH = 51
sample_size = 30800
BATCH_SIZE = 300
nIter = int(sample_size/BATCH_SIZE)
LEARNING_RATE = 0.0001
vovab_size = len(word2id)
n_steps = 80

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# save training log
def write_txt(epoch, iteration, loss):
    with open("/home/r/rishabhs/tasty_dataset/s2vt_pytorch/training_log.txt", 'a+') as f:
        f.write("Epoch:[ %d ]\t Iteration:[ %d ]\t loss:[ %f ]\n" % (epoch, iteration, loss))

def write_cap(epoch, iteration, loss, captions, caption):
    with open("/home/r/rishabhs/tasty_dataset/s2vt_pytorch/caption_log.txt", 'a+') as f:
        f.write("Epoch:[ %d ]\t Iteration:[ %d ]\t loss:[ %f ]\n" % (epoch, iteration, loss))
        write_in_english(captions,'GT Caption')
        write_in_english(caption,'Label Caption')



if __name__ == "__main__":
    #pkl_file = None
    
    #if pkl_file:
    #    s2vt.load_state_dict(torch.load("/data/video-captioning/Data/s2vt_params.pkl"))
    s2vt = S2VT(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    s2vt = s2vt.cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(s2vt.parameters(), lr=LEARNING_RATE)
    

    for epoch in range(EPOCH):
        print("EPOCH: %d " % (epoch))
        for i in range(nIter):
            print("iteration: %d" % (i))
            video, caption, cap_mask = fetch_train_data(BATCH_SIZE)
            video, caption, cap_mask = torch.FloatTensor(video).cuda(), torch.LongTensor(caption).cuda(), torch.FloatTensor(cap_mask).cuda()
            cap_out = s2vt(video, caption)
            cap_labels = caption[:, 1:].contiguous().view(-1)       # size [batch_size, 79]
            cap_mask = cap_mask[:, 1:].contiguous().view(-1)        # size [batch_size, 79]

            logit_loss = loss_func(cap_out, cap_labels)
            masked_loss = logit_loss*cap_mask
            loss = torch.sum(masked_loss)/torch.sum(cap_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                out_logits = cap_out.view(BATCH_SIZE,n_steps-1,vovab_size)
                values, output_captions = torch.max(out_logits, 2)
                output_captions = output_captions.data.cpu().numpy()
                print_in_english(output_captions[0:3])
                #captions = [[row[i] for row in captions] for i in range(len(captions[0]))]
                print('\nGT Caption:\n')
                #print_in_english(captions)
                #print('............................\nLABEL Caption:\n')
                gt_caption = caption.data.cpu().numpy()
                print_in_english(gt_caption[0:3])

                print("Epoch: %d  iteration: %d , loss: %f" % (epoch, i, loss))
                write_txt(epoch, i, loss)
                #write_cap(epoch, i, loss, output_captions, gt_caption)

            if (i%100==0 and epoch%10==0):
                pickle_file = 's2vt_epoch{}_iteration{}.pkl'.format(epoch,i)
                torch.save(s2vt.state_dict(), "/home/r/rishabhs/tasty_dataset/s2vt_pytorch/" + pickle_file)
                print("Epoch: %d iter: %d save successed!" % (epoch, i))
