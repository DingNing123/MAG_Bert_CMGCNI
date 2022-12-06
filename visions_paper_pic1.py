#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import json
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from mustards import MUStartDataset
from torch.utils.data import DataLoader
from PIL import Image
from features import load_sample_feature
from features import load_box_coordinate
from configs import *

def plt_ax_box_picture(ax, video_id, title='(a) Satirical sample'):
    '''
    plt a subplot from video_id display 5 boxes
    '''
    img = Image.open(frame_path + video_id +'/00001.jpg')
    ax.imshow(img)
    save_path = feature_sample_saved_path + video_id + '.pkl'
    audio_feature, video_feature, graph, cross_graph, vision_full_feat, cls_names_10 = load_sample_feature(save_path)
    boxs = load_box_coordinate(video_id)
    index = 0
    for cls_name0, bbox in zip(cls_names_10,boxs):
        index += 1
        if index == 6:
            break
        ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=2, alpha=0.5)
                    )

        ax.text(bbox[0], bbox[1] - 2,
                        '%s' % (cls_name0),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=10, color='white')
        ax.set_title(title)

def get_video_id(data_loader, label=1):    
    '''
    从数据集选择一个制定标记的样本id
    '''
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            print(batch.keys())
            labels = batch['labels']
            video_ids = batch['video_ids']
            print(labels[:10])
            sarcasm = (labels==label).nonzero()[0]
            print(sarcasm)
            video_id = video_ids[sarcasm]
            print(video_id)
            print('--'*20)
            break
    return video_id

def id_to_utterance(sarcasm_video_id):
    '''
    from video_id return utterance
    '''
    data = json.load(open(json_file))
    # print(data[sarcasm_video_id].keys())
    sarcasm_text = data[sarcasm_video_id]['utterance']
    return sarcasm_text

# 因为现在不是蜜蜂季节，你可以用我的肾上腺素。
# In[6]:
if __name__=="__main__": 
    
    test_dataset = MUStartDataset(mode='test', feature_path=feature_file)
    test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)

    sarcasm_video_id = get_video_id(test_dataloader,label=1)
    non_sarcasm_video_id = get_video_id(test_dataloader,label=0)

    sarcasm_text = id_to_utterance(sarcasm_video_id)
    non_sarcasm_text = id_to_utterance(non_sarcasm_video_id)
    
    # 从PPT中绘图
    print(sarcasm_text)
    print(non_sarcasm_text)
    
    fig, (ax1, ax2)= plt.subplots(1, 2,  figsize = (10,5))
    video_id = sarcasm_video_id
    plt_ax_box_picture(ax1, video_id, title='(a) Satirical sample' )
    video_id = non_sarcasm_video_id
    plt_ax_box_picture(ax2, video_id, title='(b) Non satirical sample ' )
    fig.tight_layout()
    fp = results_path + sarcasm_video_id + '_' + non_sarcasm_video_id + '_pic_1_5box.png'
    plt.savefig(fp)
    print('save to ' + fp)
    plt.show()
