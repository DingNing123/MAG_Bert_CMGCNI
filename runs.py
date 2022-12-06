'''
runs.py
设计各种跑模型的函数
'''
import logging


import json
import torch
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
from cmgcni import train_cmgcni_model
from configs import *

logger = get_stderr_file_logger(log_file)
logger.error('log in '+log_file)

def run_five_times_result(train_one_times):
    p, r, f, accs = [], [], [], []
    for i in range(5):
        logger.error('times: {}'.format(i))
        test_acc, test_f1,test_precision,test_recall = train_one_times(times=i)
        p.append(test_precision)
        r.append(test_recall)
        f.append(test_f1)
        accs.append(test_acc)
    data = {
        'p':p,
        'r':r,
        'f':f,
        'acc':accs,
    }
    
    df = pd.DataFrame(data)
    # mean and std append last two rows
    df2 = df.append(df.describe())
    
    df2.to_csv('6.times_5_{}.csv'.format(train_model_name),index=True)
    
    
if __name__=="__main__": 
    
    
    run_five_times_result(train_cmgcni_model)