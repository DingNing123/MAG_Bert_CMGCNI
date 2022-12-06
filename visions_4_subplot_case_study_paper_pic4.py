'''
绘制一个案例的可视化图，一个组合图，包含四个子图
'''
import logging
import json
import torch
import librosa

import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel
from sklearn import metrics
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers import BertTokenizer


from loggers import get_stderr_file_logger
from heatmap import heatmap,annotate_heatmap
from layers.dynamic_rnn import DynamicLSTM
from cmgcni import CMGCNI
from train import train, evaluate_acc_f1
from cmgcni import get_cmgcni_optimizer
from cmgcni import MultimodalConfig
from mustards import MUStartDataset

from PIL import Image
from heatmap import heatmap,annotate_heatmap
from features import load_sample_feature
from features import load_box_coordinate
from configs import *

def plt_ax_box_picture(ax, video_id):
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
        ax.set_title('(a) The bounding box in the sample')

def visualize_a_sample(video_id, scores, column, graph ):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (22,22), 
                                          gridspec_kw = dict(height_ratios = [1,1], 
                                                             width_ratios=[1,1]))
        
    scores_text = scores[:,:10]
    column_text = column[:10]
    row = [video_id]
    
    plt_ax_box_picture(ax1, video_id)
    
    
    im, cbar = heatmap(scores_text, row, column_text, ax=ax2, cbar_kw = {'shrink':0.8},
               cmap="YlGn", cbarlabel="scores")
    texts = annotate_heatmap(im, valfmt="{x:.3f} ")
    ax2.set_title('(b) The Bert token part of cross-modal attention')

    scores_box = scores[:,-10:]
    column_box = column[-10:]
    im, cbar = heatmap(scores_box, row, column_box, ax=ax3, cbar_kw = {'shrink':0.8},
               cmap="YlGn", cbarlabel="scores")
    texts = annotate_heatmap(im, valfmt="{x:.3f} ")
    ax3.set_title('(c) The Box Bert token part of cross-modal attention')

    im, cbar = heatmap(graph, column, column, ax=ax4, cbar_kw = {'shrink':0.8},
               cmap="YlGn", cbarlabel="graph weight")
    # texts = annotate_heatmap(im, valfmt="{x:.1f} ")
    ax4.set_title('(d) Adjacency matrix for cross-modal graphs')

    # ax1.set_box_aspect(8/10)
    ax2.set_box_aspect(1/10)
    ax3.set_box_aspect(1/10)
    ax4.set_box_aspect(10/10)
    

    fig.tight_layout()
    plt.savefig('a_sample_4_plot.png')
    plt.show()
    
def batch_to_bert_tokens_attention_graph(batch, index):
    '''
    from batch get video_id, graph, attention_scores, bert_tokens with box_tokens
    param:
        batch:  batch 128 of dataloader 
        index: index of one batch 0..127
    return :
        
    '''
    bert_indices = batch['bert_indices'][index]
    bert_tokens = tokenizer.convert_ids_to_tokens(bert_indices, skip_special_tokens=False)
    box_indices = batch['box_pad_indices'][index]
    box_tokens = []
    for i in range(10):
        box_token = tokenizer.convert_ids_to_tokens(box_indices[i], skip_special_tokens=True)
        box_tokens.append(box_token)
    box_tokens = [' '.join(box_token) for box_token in box_tokens]
    bert_tokens.extend(box_tokens)

    video_id = batch['video_ids'][index]

    inputs = batch
    outputs, alpha = model(inputs, outAttention = True)
    outputs = torch.argmax(outputs,-1)
    scores = alpha[index]
    graph = batch['big_graphs'][index]
    return video_id, graph, scores, bert_tokens

def run_visualization_a_sample(data_loader):
    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader):
            # print(i_batch)
            # print(batch.keys())
            index = 0
            video_id, graph, scores, bert_tokens = batch_to_bert_tokens_attention_graph(batch, index)
            column = bert_tokens
            visualize_a_sample(video_id, scores, column, graph)
            print('--'*20)
            break


if __name__=="__main__": 
    logger = get_stderr_file_logger(log_file)
    multimodal_config = MultimodalConfig(
        beta_shift=beta_shift, dropout_prob=dropout_prob
    )

    # train_dataset = MUStartDataset(mode='train')
    # valid_dataset = MUStartDataset(mode='valid')
    test_dataset = MUStartDataset(mode='test')

    logger.error('train_dataset len : {}'.format(len(train_dataset)))

    # train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=True)
    # valid_dataloader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)
    test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)

    cmgcni_model = CMGCNI(multimodal_config = multimodal_config )

    cmgcni_model_path = '../tools/cmgcni_model.pth'
    model = cmgcni_model
    model_save_path = cmgcni_model_path
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    Bert_model_path = '../tools/bert-base-uncased/'
    tokenizer = BertTokenizer.from_pretrained(Bert_model_path)
    data_loader = test_dataloader
    
    run_visualization_a_sample(data_loader)
    
    
    