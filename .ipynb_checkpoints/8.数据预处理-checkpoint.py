#!/usr/bin/env python
# coding: utf-8

# In[13]:


import logging
import json
import spacy
import pickle
import numpy as np
import re
import os
import pandas as pd
import detectron2.utils.comm as comm
import cv2
import torch
import librosa
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from glob import glob
from PIL import Image
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from transformers import BertTokenizer
from transformers import ViTFeatureExtractor,ViTModel

from loggers import get_stderr_file_logger
from utils.functions import csv_header, csv_add_one_row, get_file_size
from bua.caffe import add_bottom_up_attention_config
from extract_utils import get_image_blob
from bua.caffe.modeling.layers.nms import nms


# # 函数
# ## 生成10个目标盒子

# In[3]:


def clean_text_remove_punctuation(text):
    '''
    清理文本中的标点符号,并且转换为小写，合并连续的空格为一个
    '''
    punctuation = '!,;:?"、，；.'
    
    text1 = re.sub(r'[{}]+'.format(punctuation),' ',text)
    text2 = re.sub(r'[\']',' ',text1)
    text2 = text2.strip().lower()
    text2 = ' '.join(text2.split())
    return text2

def add_split(mode_index, mode ='train'):
    for idx, ID in enumerate(list(data.keys())):
            video_id = ID
            clip_id = 0 
            text = data[ID]['utterance']
            text = clean_text_remove_punctuation(text)
            
            label = 1.0 if data[ID]['sarcasm'] else -1.0
            annotation = 'Positive' if data[ID]['sarcasm'] else 'Negative'        # train valid test
            label_by = 0 
            if idx in mode_index:
                logger.info(video_id)
                row = {'video_id':video_id,
                        'clip_id':clip_id,
                        'text':text,
                        'label':label,
                        'annotation':annotation,
                        'mode':mode,
                        'label_by':label_by,
                        }
                # logger.info(str(row))
                csv_add_one_row(label_csv, fieldnames, row)

def get_label_csv():
    csv_header(label_csv, fieldnames)
    add_split(train_index, mode = 'train')
    add_split(test_index, mode = 'valid')
    add_split(test_index, mode = 'test')    
    logger.info('write to {} '.format(label_csv))
    
    
    


# In[1]:


def get_train_test_ids():
    with open(json_file) as f:
        data = json.load(f)

    train_ids = []
    test_ids = []
    train_index = []
    test_index = []
    for id,ID in enumerate(list(data.keys())):
        speaker = data[ID]['speaker']
        if speaker == 'HOWARD' or speaker == 'SHELDON':
            test_index.append(id)
            test_ids.append(ID)
        else:
            train_index.append(id)
            train_ids.append(ID)

    logger.info('train:{} test:{}'.format(len(train_ids), len(test_ids)))

    if debug:
        DATA_PIECES = 2
        train_index = train_index[:DATA_PIECES]
        test_index = test_index[:DATA_PIECES]
        train_ids = train_ids[:DATA_PIECES]
        test_ids = test_ids[:DATA_PIECES]

    logger.info('train:{} test:{}'.format(len(train_ids), len(test_ids)))
    logger.info(train_ids)
    logger.info(test_ids)
    logger.info(train_index)
    logger.info(test_index)
    return train_ids,test_ids,train_index,test_index,data


# In[5]:


def video_2_frames():
    videos = glob(video_path+'*')
    for video in videos:
        video = video.replace('\\','/')
        logger.info(video)
        video_id = video.split('/')[2].split('.')[0]
        if video_id in train_ids + test_ids:
            dirName = frame_path + video_id
            if os.path.exists(dirName):
                logger.info("Directory {} already exists".format(dirName))
                img_files = os.listdir(dirName)
                if len(img_files) == 0:
                    logger.info(dirName)
                    input_mp4 = video_path + video_id + ".mp4"
                    ffmpeg = '/Users/mac/anaconda3/envs/t18/bin/ffmpeg'
                    ffmpeg = 'C:\\ProgramData\\Anaconda3\\envs\\t18\\Library\\bin\\ffmpeg.exe'
                    cmd = "{} -i {} -vf fps=3 {}/%5d.jpg".format(ffmpeg,input_mp4,dirName)
                    logger.info(cmd)
                    os.system(cmd)
                
            
            else:
                os.mkdir(dirName)
                logger.info("Directory {} Created".format(dirName))

                input_mp4 = video_path + video_id + ".mp4"
                ffmpeg = '/Users/mac/anaconda3/envs/t18/bin/ffmpeg'
                ffmpeg = 'C:\\ProgramData\\Anaconda3\\envs\\t18\\Library\\bin\\ffmpeg.exe'
                cmd = "{} -i {} -vf fps=1 {}/%5d.jpg".format(ffmpeg,input_mp4,dirName)
                logger.info(cmd)
                os.system(cmd)

                


# In[7]:


def get_classes_attributes():
    data_path = 'evaluation'
    # Load classes
    classes = ['__background__']
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())
    logger.info(len(classes))
    logger.info(classes[:10])

    # Load attributes
    attributes = ['__no_attribute__']
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for att in f.readlines():
            attributes.append(att.split(',')[0].lower().strip())
    logger.info(len(attributes))
    logger.info(attributes[:10])
    return classes, attributes




def get_bua_model():
    config_file = 'configs/caffe/test-caffe-r152.yaml'
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cuda'

    add_bottom_up_attention_config(cfg, True)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(['MODEL.BUA.EXTRACT_FEATS',True])
    cfg.freeze()


    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    model.eval()
    return model,cfg





def model_inference(model, batched_inputs, mode):
    logger.info('caffe model device: {}'.format(model.device))
    if mode == "caffe":
        return model(batched_inputs)
    elif mode == "d2":
        images = model.preprocess_image(batched_inputs)
        features = model.backbone(images.tensor)
    
        if model.proposal_generator:
            proposals, _ = model.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(model.device) for x in batched_inputs]

        return model.roi_heads(images, features, proposals, None)
    else:
        raise Exception("detection model not supported: {}".format(mode))
        




def video_to_wav(video_file, audio_file):
    cmd = 'ffmpeg -v quiet  -i ' + video_file + ' -f wav -vn ' + audio_file
    os.system(cmd)


def audio_embedding(audio_file):
    audio_feature = None
    y, sr = librosa.load(audio_file)
    hop_length = 512 * 8 
    # hop_length smaller, seq_len larger
    f0 = librosa.feature.zero_crossing_rate(y, hop_length = hop_length).T
    mfcc = librosa.feature.mfcc(y=y,sr=sr,hop_length=hop_length).T
    cqt = librosa.feature.chroma_cqt(y=y,sr=sr,hop_length=hop_length).T
    audio_feature = np.concatenate([f0, mfcc, cqt], axis = -1)
    # logger.info(audio_feature.shape)
    return audio_feature

def padding_sequence(sequences):
    '''
    return 填充后的形状统一的特征,截断后的序列长度数组
    原始数据集长度差异太大了。
    '''
    features = None
    feature_dim = sequences[0].shape[-1]
    lens = [s.shape[0] for s in sequences]
    # final_length = int(np.mean(lens) + 1 * np.std(lens))
    logger.error('mean: {} std:{}'.format(np.mean(lens), np.std(lens)))
    
    final_length = final_seq_length
    
    features = np.zeros([len(sequences), final_length, feature_dim])
    sequence_lenth_array = []
    for i, s in enumerate(sequences):
        # features[i] = s + [0000]
        # feature = s
        # MAX_LEN = final_length 
        # 为了避免后期LSTM长度错误，记录截断后的长度
        length = s.shape[0]
        if length >= final_length:
            features[i] = s[:final_length, :]
            length = final_length
            # logger.info('截断')
        else:
            pad = np.zeros([final_length - length, feature_dim])
            features[i] = np.concatenate((s, pad), axis = 0)
            # logger.info('pad end', pad.shape)
        sequence_lenth_array.append(length)
    
    return features, sequence_lenth_array
    
def file_2_10_boxes_list(im_file):    
    im = cv2.imread(im_file)
    dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
    
    mode = "caffe"
    img_id = im_file
    with torch.set_grad_enabled(False):
        boxes, scores, features_pooled, attr_scores = model_inference(model,[dataset_dict],mode)
    
    dets = boxes[0].tensor.cpu() / dataset_dict['im_scale']
    scores = scores[0].cpu()
    feats = features_pooled[0].cpu()
    attr_scores = attr_scores[0].cpu()
    
    max_conf = torch.zeros((scores.shape[0])).to(scores.device)
    logger.info(scores.device)
    for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.3)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                        cls_scores[keep],
                                        max_conf[keep])

    keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    boxes = dets[keep_boxes].numpy()
    import numpy as np

    objects = np.argmax(scores[keep_boxes].numpy()[:,1:], axis=1)
    attr_thresh = 0.1
    attr = np.argmax(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
    attr_conf = np.max(attr_scores[keep_boxes].numpy()[:,1:], axis=1)
    
    box_list = list()
    
    for i in range(len(keep_boxes)):
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1

        if mode == "caffe":
            cls = classes[objects[i]+1]  # caffe +2
            if attr_conf[i] > attr_thresh:
                cls = attributes[attr[i]+1] + " " + cls   #  caffe +2
        elif mode == "d2":
            cls = classes[objects[i]+2]  # d2 +2
            if attr_conf[i] > attr_thresh:
                cls = attributes[attr[i]+2] + " " + cls   # d2 +2
        else:
            raise Exception("detection model not supported: {}".format(mode))

        
        logger.info(bbox)
        logger.info(cls)
        box_list.append([bbox,cls])
    
    return box_list

def generate_10_box_pics(video_id, frame0, box_list):
    box_path = './mmsd_raw_data/Processed/video/box/'

    im_dir = box_path + video_id
    import os
    isExist = os.path.exists(im_dir)
    if  isExist:
        logger.info('The directory {} existed,no operation warning! '.format(im_dir))
        
    else:
        os.makedirs(im_dir)
        logger.info("The new directory {} is created!".format(im_dir))
        
        # 生成box图片，写入txt文件
        img_path = frame0
        im = cv2.imread(img_path)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        lines = []
        for idx, box in enumerate(box_list):
            box , cls_name = box
            x1,y1,x2,y2 = box
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            patch_img = im[y1:y2,x1:x2]
            patch_name = im_dir +'/'+str(idx)+'.jpg'
            cv2.imwrite(patch_name,patch_img)
            lines.append(str(idx) + ' ' + cls_name + '\n')
        with open(im_dir+'.txt','w') as f:
            f.writelines(lines)

        logger.info(im_dir+'.txt')


def vit_feature(video_id):
    box_path = './mmsd_raw_data/Processed/video/box/'
    
    txt = box_path + video_id + '.txt'
    im_id = video_id
    with open(txt) as f:
        lines = f.readlines()
    # 为了能批量处理特征所以生成图片列表 
    features = []
    for line in lines:
        box_id = line.split()[0]
        cls_name = ' '.join(line.split()[1:])
        feature_id = cls_name + '_' + box_id
        box_img = box_path + im_id + '/'+box_id+ '.jpg'
        box_img_pil = Image.open(box_img)

        inputs = feature_extractor(box_img_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)

        pool = outputs.pooler_output

        features.append(pool)
    
    features = torch.cat(features,0)
    return features

def vision_10_boxes_embedding(video_id):
    '''
    return 10 * 768 and 10 box_text
    '''
    frame_id_path = frame_path + video_id
    frames = glob(frame_id_path + '/*')
    frames = sorted(frames)
    # 00001.jpg
    frame0 = frames[0]
    box_list = file_2_10_boxes_list(frame0)
    
    generate_10_box_pics(video_id, frame0, box_list)
    
    box_feature768 = vit_feature(video_id)
    
    
    cls_names = [box[1] for box in box_list]
    
    
    return box_feature768, cls_names


# ## 生成全局视觉特征768

# In[8]:


def vision_full_pic_embedding(video_id):
    '''
    return full pic vit model 768 
    '''
    frame_id_path = frame_path + video_id
    frames = glob(frame_id_path + '/*')
    full_feature768 = []
    for frame in sorted(frames):
        
        img = Image.open(frame)
        inputs = feature_extractor(img, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)

        pool = outputs.pooler_output
        full_feature768.append(pool)
        
        # break
    full_feature768 = torch.cat(full_feature768, 0 )
    
    
    return full_feature768


def func_padding_vision_full_feature():
    max_vision_len = 0
    vision_feature_dim = vision_full_feats[0].size(1)
    video_nums = len(vision_full_feats)
    for vision in vision_full_feats:
        if vision.size(0) > max_vision_len:
            max_vision_len = vision.size(0)
    
    logger.error('source max_vision_len: {}'.format(max_vision_len))
    max_vision_len =  final_seq_length

    padding_vision_full_feature = torch.zeros(video_nums, max_vision_len,vision_feature_dim)
    for index, vision in enumerate(vision_full_feats):
        frames_num = vision.size(0)
        padding_vision_full_feature[index][:frames_num] = vision
        
    return padding_vision_full_feature


# ## 生成图的函数

# In[10]:


def generate_graph(text):
    from transformers import BertTokenizer
    from collections import defaultdict
    import spacy

    bert_token = tokenizer.tokenize(text)
    document = nlp(text)
    spacy_token = [str(x) for x in document]
    spacy_len = len(spacy_token)
    bert_len = len(bert_token)
    
    ii = 0
    jj = 0
    s = ""
    pre = []
    split_link = []
    while ii < bert_len and jj < spacy_len:
        b = bert_token[ii].replace('##','')
        s += b
        pre.append(ii)
        spa_ = spacy_token[jj]
        logger.info('{} =? {}'.format(s, spa_))
        if s == spa_:
            split_link.append(pre)
            jj += 1
            s = ""
            pre = []

        ii += 1
        
    flag_use_spacy = True
    # 判断如果spack_token 反而拆分了单词的情况 
    if jj < spacy_len: 
        logger.info('spacy split a bert word,just simplify it,not use nlp  ')
        flag_use_spacy = False
    
    logger.info(text)
    logger.info('bert_token: {}'.format(bert_token))
    logger.info('spacy_token: {}'.format(spacy_token))
    logger.info(split_link)
    
    doc = nlp(text)
    mat = defaultdict(list,[])
    for t in doc:
        for child in t.children:
            # print(t, t.i, child.i)
            mat[child.i].append(t.i)
            mat[t.i].append(child.i)
    
    import numpy as np
    outter_graph = np.zeros((bert_len,bert_len)).astype('float32')
    
    if flag_use_spacy:
        for key,linked in mat.items():
                for x in split_link[key]:
                    for link in linked:
                        for y in split_link[link]:
                            outter_graph[x][y] = 1
    else:
        pass
        
                
    tokens = bert_token
    inner_graph = np.identity(bert_len).astype('float32')
    
    if flag_use_spacy:
        for link in split_link:
            for x in link:
                for y in link:
                    inner_graph[x][y] = 1
    else:
        pass
    
    outter_graph = np.pad(outter_graph, ((1,1),(1,1)), 'constant')
    inner_graph = np.pad(inner_graph, ((1,1),(1,1)), 'constant')
    inner_graph[0][0] = 1
    inner_graph[-1][-1] = 1
    graph1 = inner_graph + outter_graph
                
    logger.info(graph1.shape)
    return graph1


# In[11]:


def get_split(text):
    split_link = []
    bert_token = tokenizer.tokenize(text)
    document = nlp(text)
    spacy_token = [str(x) for x in document]
    bert_len = len(bert_token)
    spacy_len = len(spacy_token)
    ii = 0
    jj = 0
    pre = []
    s = ""
    while ii < bert_len and jj < spacy_len:
        bert_ = bert_token[ii].replace('##','')
        s += bert_
        pre.append(ii)
        spacy_ = spacy_token[jj]
        if s == spacy_:
            split_link.append(pre)
            pre = []
            s = ""
            jj += 1
        ii += 1

    return split_link

def load_sentic_word():
    path = './senticNet/senticnet_word.txt'
    sNet = {}
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            word, score = line.split('\t')
            sNet[word] = float(score)
    return sNet



def get_sentic_score(si, adj):
    '''
    计算情感的不一致性，一致设置为0 ，否则放大这种不协调，讽刺的冲突
    '''
    word_i = si.lemma_names()[0]
    word_j = adj.lemma_names()[0]
    if word_i not in senticNet or word_j not in senticNet or \
      word_i == word_j:
        res = 0
    else:
        res = abs(senticNet[word_i] - senticNet[word_j]) * \
          y ** (-1 * senticNet[word_i] * senticNet[word_j])
    
    # print(word_i, word_j, res)
    return res


# In[12]:


def generate_cross_graph(text, box_text):
    # text = 'haha . lol'
    # text = 'thanks for showing up for our appointment today .'
    
    bert_text_tokens = tokenizer.tokenize(text.lower())
    
    
    document = nlp(text.lower())
    spacy_text_tokens = [str(x).lower() for x in document]
    spacy_text_tokens
    box_len = len(box_text)
    box_len
    import numpy as np
    graph = np.zeros((len(bert_text_tokens), box_len)).astype('float32')
    
    split_link_ = get_split(text)
    split_link_
    split_link = dict()
    for idx, value in enumerate(split_link_):
        for v in value:
            split_link[bert_text_tokens[v]] = spacy_text_tokens[idx]
    logger.info(split_link)
    
    for i, token in enumerate(bert_text_tokens):
        cur = 0
        try:
            sp_token = split_link[token]
        except:
            continue
        si = wn.synsets(sp_token)
        if len(si) == 0 :
            continue
        si = si[0]
        for b in box_text:
            tokens_ = nlp(b)
            tokens_ = [str(x) for x in tokens_]
            logger.info(tokens_)
            if len(tokens_) >= 2:
                adj_j, obj_j = tokens_[0],tokens_[1]
            if len(tokens_) == 1:
                adj_j, obj_j = 'a', tokens_[0]
            
            logger.info(adj_j)
            logger.info(obj_j)
            logger.info(len(wn.synsets(adj_j)))
            if len(wn.synsets(adj_j)) == 0:
                sim = 0
            elif len(wn.synsets(obj_j)) == 0:
                sim = 0
            else:
                adj = wn.synsets(adj_j)[0]
                obj = wn.synsets(obj_j)[0]
                sim = wn.path_similarity(si, obj)
            if sim is None:
                graph[i][cur] = 0 + get_sentic_score(si, adj)
            else:
                graph[i][cur] = sim + get_sentic_score(si, adj)
            # print(i, cur, graph[i][cur])
            cur += 1
    # print(graph, graph.shape)

    return graph


# ## 填充函数

# In[13]:


def merge_graph(graphs, cross_graphs):
    bert_indices_max_len = 0
    bert_indices_len = []
    for graph, cross_graph in zip(graphs, cross_graphs):
        bert_indices_len.append(graph.shape[0])
        
    # bert_indices_max_len = max(bert_indices_len)
    bert_indices_max_len = final_seq_length
    big_graphs = []
    for graph, cross_graph in zip(graphs, cross_graphs):
        cross_graph = np.pad(cross_graph, ((1,1),(0,0)), 'constant')
        # print(graph.shape, cross_graph.shape)
        if graph.shape[0] < bert_indices_max_len:
            graph = np.pad(graph, ((0,bert_indices_max_len-graph.shape[0]),(0,bert_indices_max_len-graph.shape[0])), 'constant')
            
        # 将文本图的大小设置为20 
        graph = graph[:20,:20]
        cross_graph = cross_graph[:20,:10]
        logger.info('graph:{} cross_graph: {}'.format(graph.shape, cross_graph.shape))
        
        graph = np.pad(graph,((0,10),(0,10)),'constant')

        image_graph = cross_graph
        for i in range(image_graph.shape[0]-2):
            for j in range(image_graph.shape[1]):
                if not np.isnan(image_graph[i][j]):
                    
                    graph[i+1][j+bert_indices_max_len] = image_graph[i + 1][j] + 1 
                    graph[j+bert_indices_max_len][i+1] = image_graph[i + 1][j] + 1 
                else:
                    graph[i+1][j+bert_indices_max_len] =  1
                    graph[j+bert_indices_max_len][i+1] =  1

        for i in range(image_graph.shape[1]):
            graph[i+bert_indices_max_len][i+bert_indices_max_len] = 1 
        graph = np.expand_dims(graph, axis=0)
        big_graphs.append(graph)
            
    big_graphs = np.concatenate(big_graphs, axis=0)
    
    return big_graphs

def labels_to_np(labels):
    labels = np.array(labels)
    return labels



def padding_video_features(video_features):
    features_p = []
    for vf in video_features:
        vf = np.expand_dims(vf, axis=0)
        features_p.append(vf)
    features_p = np.concatenate(features_p, axis = 0)
    return features_p



def padding_text(texts):
    bert_indices = []
    for text in texts:
        bert_tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
        bert_index = tokenizer.convert_tokens_to_ids(bert_tokens)
        bert_indices.append(bert_index)
        # print(bert_index)
    lens = [len(x) for x in bert_indices]
    
    lens = np.array(lens)
    
    bert_indices_max_len = lens.max().item()
    bert_indices_mean_len = lens.mean().item()
    bert_indices_std_len = lens.std().item()
    # final_seq_length = int(bert_indices_mean_len)
    logger.error('max: {} mean: {} std: {}'.format(bert_indices_max_len,bert_indices_mean_len,bert_indices_std_len ))
    
    # bert_indices_max_len = max(lens)
    
    
    
    bert_indices_pad = [np.pad(x,(0, bert_indices_max_len - len(x)),'constant') for x in bert_indices]
    
    bert_indices_pad = [x[:final_seq_length] for x in bert_indices_pad]
    
    bert_indices_pad = np.array(bert_indices_pad)
    
    
    return bert_indices_pad


def padding_box_text(box_texts):
    box_bert_indices = []
    for box_text in box_texts:
        
        for text in box_text:
            bert_tokens = tokenizer.tokenize(text) 
            bert_index = tokenizer.convert_tokens_to_ids(bert_tokens)
            box_bert_indices.append(bert_index)
        
    lens = [len(x) for x in box_bert_indices]
    box_indices_max_len = max(lens)
    
    box_pad_indices = []
    for box_text in box_texts:
        new_box_indices = []
        box_indices = []
        for text in box_text:
            bert_tokens = tokenizer.tokenize(text) 
            bert_index = tokenizer.convert_tokens_to_ids(bert_tokens)
            box_indices.append(bert_index)
            
        for box_indice in box_indices:
            if len(box_indice) < box_indices_max_len:
                box_indice = box_indice + [0]*(box_indices_max_len - len(box_indice))
            new_box_indices.append(np.array(box_indice))

        while len(new_box_indices) < 10:
            new_box_indices.append([0]*box_indices_max_len)
        box_pad_indices.append(new_box_indices)
    
    box_pad_indices = np.array(box_pad_indices)
    
    logger.info(box_pad_indices.shape)
    return box_pad_indices


# ## 保存加载每条数据的特征的函数

# In[15]:


def save_sample_feature(audio_feature,video_feature,graph,cross_graph,vision_full_feat,cls_names_10,save_path):
    data  = audio_feature,video_feature,graph,cross_graph,vision_full_feat,cls_names_10
    with open(save_path,'wb') as f:
        pickle.dump(data,f)

def load_sample_feature(save_path):
    with open(save_path,'rb') as f:
        data = pickle.load(f) 
    return data

def generate_feature(save_path,video_id,text):
    audio_dir = audios_path + video_id
    audio_file = audio_dir + '/tmp.wav'
    video_file = video_path + video_id + '.mp4'
    if  os.path.exists(audio_dir):
        logger.info('{} existed, do nothing'.format(audio_dir))
        pass
    else:
        os.makedirs(audio_dir)
        logger.info('makedirs ' + audio_dir)
        video_to_wav(video_file, audio_file)
        
    audio_feature = audio_embedding(audio_file)
    
    # vision
    video_feature, cls_names_10 = vision_10_boxes_embedding(video_id)
    graph = generate_graph(text)
    cross_graph = generate_cross_graph(text, cls_names_10)
    vision_full_feat = vision_full_pic_embedding(video_id)
    
    # save feature to save time if interrupted 
    # audio_feature,video_feature,graph,cross_graph
    
    save_sample_feature(audio_feature, video_feature, graph,cross_graph, vision_full_feat, cls_names_10, save_path)
    return audio_feature,video_feature,graph,cross_graph,vision_full_feat,cls_names_10


# # 开始运行

# In[14]:


log_file = '8.log'
debug = True
debug = False
json_file = 'sarcasm_data.json'
label_csv = 'label_indep.csv'
fieldnames = ['video_id', 'clip_id','text', 'label', 'annotation','mode', 'label_by']
video_path = 'mmsd_raw_data/utterances_final/'
frame_path = 'mmsd_raw_data/Processed/video/Frames/'

logger = get_stderr_file_logger(log_file)

logging.info("This is an INFO message")
logging.warning("This is a WARNING message")

VIT_MODEL_PATH = "../tools/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(VIT_MODEL_PATH)
vit_model = ViTModel.from_pretrained(VIT_MODEL_PATH)
output_path = './featuresIndepVit768.pkl'
final_seq_length = 20
audios_path = 'mmsd_raw_data/Processed/audio/'
df = pd.read_csv(label_csv)
classes, attributes = get_classes_attributes()
model,cfg = get_bua_model()
logger.info(model.device)
MIN_BOXES = 10
MAX_BOXES = 10
CONF_THRESH = 0.4

y = 3
senticNet = load_sentic_word()
logger.info(list(senticNet.items())[:2])
Bert_model_path = '../tools/bert-base-uncased/'
tokenizer = BertTokenizer.from_pretrained(Bert_model_path)
nlp = spacy.load('en_core_web_sm')


# ## speaker_independent的划分的索引

# In[2]:


train_ids,test_ids,train_index,test_index,data = get_train_test_ids()


# ## 生成需要的标签文件 label_indep.csv

# In[4]:


get_label_csv()


# ## 视频拆分为帧

# In[6]:


video_2_frames()
img = Image.open(frame_path + '1_90/00001.jpg')
img


# ## 生成多模态特征数组

# In[15]:


video_ids = []
audio_features = []
modes = []
labels = []
video_features = []
box_texts =[]
graphs = []
cross_graphs = []
texts = []
vision_full_feats = []
feature_sample_saved_path = './feature_sample/'
for index in tqdm(range(len(df))):
    video_id, _, text, label, _, mode, _ = df.loc[index]
    
    save_path = feature_sample_saved_path + video_id + '.pkl'
    logger.info('start save {}'.format(save_path))
    
    if os.path.exists(save_path):
        logger.info('load from file {}'.format(save_path))
        audio_feature,video_feature,graph,cross_graph,vision_full_feat,cls_names_10 = load_sample_feature(save_path)
    else:
        audio_feature,video_feature,graph,cross_graph,vision_full_feat,cls_names_10 = generate_feature(save_path,video_id,text)
    logger.info(video_feature.shape)
    logger.info(vision_full_feat.shape)
 
    texts.append(text)
    video_ids.append(video_id)
    audio_features.append(audio_feature)
    video_features.append(video_feature)
    box_texts.append(cls_names_10)
    graphs.append(graph)
    cross_graphs.append(cross_graph)
    modes.append(mode)
    labels.append(label)
    vision_full_feats.append(vision_full_feat)
    
    if index == 1:
        pass
        # break
logger.info('processed {} samples'.format(len(box_texts)))


# ## 填充

# In[16]:


video_ids = np.array(video_ids)
bert_indices = padding_text(texts)
logger.error('bert_indices : {}'.format(bert_indices.shape))
big_graphs = merge_graph(graphs, cross_graphs)
logger.error('big_graphs : {}'.format(big_graphs.shape))
labels = labels_to_np(labels)
video_features_p = padding_video_features(video_features)
box_pad_indices = padding_box_text(box_texts)
audio_padding_fea, audio_length_array = padding_sequence(audio_features)
logger.error('audio_padding_fea : {}'.format(audio_padding_fea.shape))
padding_vision_full_feature = func_padding_vision_full_feature()
logger.error('padding_vision_full_feature : {}'.format(padding_vision_full_feature.shape))


# ## 整理数据 train valid test 

# In[17]:


inx_dict = { mode + '_index' : [
    i for i,v in enumerate(modes) if v==mode
] for mode in ['train', 'valid', 'test']}


final_data = {k:{} for k in ['train', 'valid', 'test']}
for mode in ['train', 'valid', 'test']:
    indexes = inx_dict[mode + '_index']
    final_data[mode]['audio_feature'] = audio_padding_fea[indexes]
    final_data[mode]['video_features_p'] = video_features_p[indexes]
    final_data[mode]['bert_indices'] = bert_indices[indexes]
    final_data[mode]['box_pad_indices'] = box_pad_indices[indexes]
    final_data[mode]['big_graphs'] = big_graphs[indexes]
    final_data[mode]['labels'] = labels[indexes]
    final_data[mode]['vision_full_feature'] = padding_vision_full_feature[indexes]
    final_data[mode]['video_ids'] = video_ids[indexes]
    

with open(output_path, 'wb') as f:
    pickle.dump(final_data, f)
logger.error('write to {}'.format(output_path))
get_file_size(output_path, "MB")

logger.info(video_features_p.shape)
logger.info(padding_vision_full_feature.shape)


# # 注意事项
# macOS 出现了找不到libffi.7这个库
# brew install 之后 
# 建立符号链接
# LD_LIBRARY_PATH 使用这个环境变量添加查找共享库
