import os
from module import *
from utils import *
import json

vovab_size = len(word_counts)
BATCH_SIZE = 1

if __name__ == "__main__":
    s2vt = S2VT(vocab_size=vovab_size, batch_size=BATCH_SIZE)
    s2vt = s2vt.cuda()
    s2vt.load_state_dict(torch.load(os.environ.get("S2VT_ROOT", "/home/r/rishabhs/tasty_dataset/") + "S2VT_pytorch2/s2vt_params.pkl"))
    s2vt.eval()
    val_result = {}
    for idx in range(0,3284,BATCH_SIZE):
        video, caption, cap_mask, vid = fetch_val_data_orderly(idx, batch_size=BATCH_SIZE)
        video = torch.FloatTensor(video).cuda()

        cap_out = s2vt(video)

        captions = []
        for tensor in cap_out:
            captions.append(tensor.tolist())

        captions = [[row[i] for row in captions] for i in range(len(captions[0]))]

        print('............................\nPredicted Caption:\n')
        print_in_english(captions)
        print('............................\nGround Truth Caption:\n')
        print_in_english(caption)
        val_result = save_val_result(vid, captions, val_result)
    with open(os.environ.get("S2VT_ROOT", "/home/r/rishabhs/tasty_dataset/") + "S2VT_pytorch2/result.json", "a+") as f:
        json.dump(val_result, f)
