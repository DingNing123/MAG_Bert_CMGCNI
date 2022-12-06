#!/usr/bin/env python
# coding: utf-8

# macOS 出现了找不到libffi.7这个库
# brew install 之后 
# 建立符号链接
# LD_LIBRARY_PATH 使用这个环境变量添加查找共享库

# # 生成speaker_independent的划分的索引文件

# In[1]:


# debug : 只测试两条数据以完善代码。
import json

debug = True
json_file = 'sarcasm_data.json'

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
        
print('total:', len(train_ids), len(test_ids))

if debug:
    DATA_PIECES = 2
    train_index = train_index[:DATA_PIECES]
    test_index = test_index[:DATA_PIECES]
    train_ids = train_ids[:DATA_PIECES]
    test_ids = test_ids[:DATA_PIECES]
    
print(len(train_ids), len(test_ids))
print(train_ids[:DATA_PIECES], test_ids[:DATA_PIECES])
print(train_index[:DATA_PIECES], test_index[:DATA_PIECES])



# # 生成需要的标签文件 label_indep.csv

# In[2]:


from utils.functions import csv_header, csv_add_one_row, get_file_size

def clean_text_remove_punctuation(text):
    '''
    清理文本中的标点符号,并且转换为大写
    '''
    punctuation = '!,;:?"、，；.'
    import re
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
                # print(idx, video_id, mode)
                row = {'video_id':video_id,
                        'clip_id':clip_id,
                        'text':text,
                        'label':label,
                        'annotation':annotation,
                        'mode':mode,
                        'label_by':label_by,
                        }
                csv_add_one_row(label_csv, fieldnames, row)

label_csv = 'label_indep.csv'
fieldnames = ['video_id', 'clip_id','text', 'label', 
              'annotation','mode', 'label_by']

csv_header(label_csv, fieldnames)
add_split(train_index, mode = 'train')
add_split(test_index, mode = 'valid')
add_split(test_index, mode = 'test')    


# In[3]:


get_ipython().system('cat -n  label_indep.csv | head')
get_ipython().system('wc {label_csv}')


# # 视频拆分为帧

# In[4]:


import os
from glob import glob

video_path = 'mmsd_raw_data/utterances_final/'
frame_path = 'mmsd_raw_data/Processed/video/Frames/'

videos = glob(video_path+'*')

for video in videos:
    video_id = video.split('/')[2].split('.')[0]
    if video_id in train_ids + test_ids:
        dirName = frame_path + video_id
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
            input_mp4 = video_path + video_id + ".mp4"
            ffmpeg = '/Users/mac/anaconda3/envs/t18/bin/ffmpeg'
            cmd = "{} -i {} -vf fps=1 {}/%5d.jpg".format(ffmpeg,input_mp4,dirName)
            print(cmd)
            os.system(cmd)
        else:
            print("Directory " , dirName ,  " already exists")
            


# In[5]:


get_ipython().system('tree {frame_path}')


# In[6]:


from PIL import Image
img = Image.open(frame_path + '1_90/00001.jpg')
img


# In[7]:


get_ipython().system("grep '1_90' {label_csv}")


# # 提取特征
!tree mmsd_raw_data/Processed/audio
!mv mmsd_raw_data/Processed/audio/1_70/ /tmp/2
# In[8]:


import pandas as pd
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from bua.caffe import add_bottom_up_attention_config
import cv2
from extract_utils import get_image_blob
from bua.caffe.modeling.layers.nms import nms
import torch

output_path = './featuresIndepResnet152.pkl'
audios_path = 'mmsd_raw_data/Processed/audio/'
data_path = 'evaluation'

df = pd.read_csv(label_csv)
# Load classes
classes = ['__background__']
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        classes.append(object.split(',')[0].lower().strip())
print(len(classes))
print(classes[:10])

# Load attributes
attributes = ['__no_attribute__']
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for att in f.readlines():
        attributes.append(att.split(',')[0].lower().strip())
print(len(attributes))
print(attributes[:10])


config_file = 'configs/caffe/test-caffe-r152.yaml'
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'

add_bottom_up_attention_config(cfg, True)
cfg.merge_from_file(config_file)
cfg.merge_from_list(['MODEL.BUA.EXTRACT_FEATS',True])
cfg.freeze()


model = DefaultTrainer.build_model(cfg)
DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    cfg.MODEL.WEIGHTS, resume=True
)
model.eval()


MIN_BOXES = 10
MAX_BOXES = 10
CONF_THRESH = 0.4

def model_inference(model, batched_inputs, mode):
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
        


import librosa
import numpy as np
from tqdm import tqdm

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
    # print(audio_feature.shape)
    return audio_feature

def padding_sequence(sequences):
    '''
    return 填充后的形状统一的特征,截断后的序列长度数组
    原始数据集长度差异太大了。
    '''
    features = None
    feature_dim = sequences[0].shape[-1]
    lens = [s.shape[0] for s in sequences]
    final_length = int(np.mean(lens) + 1 * np.std(lens))
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
            # print('截断')
        else:
            pad = np.zeros([final_length - length, feature_dim])
            features[i] = np.concatenate((s, pad), axis = 0)
            # print('pad end', pad.shape)
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

        # print('x1,y1,x2,y2 : ', bbox,'  cls:', cls)
        box_list.append([bbox,cls])
    
    return box_list

def generate_10_box_pics(video_id, frame0, box_list):
    box_path = './mmsd_raw_data/Processed/video/box/'

    im_dir = box_path + video_id
    import os
    isExist = os.path.exists(im_dir)
    if not isExist:
        os.makedirs(im_dir)
        print("The new directory {} is created!".format(im_dir))
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
    # print(len(lines), im_dir + '.txt')



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
#         print(pool.shape,feature_id) 1,768 dark sky_0
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
    # print(cls_names)
    
    return box_feature768, cls_names


# # 生成全局视觉特征768

# In[9]:


import torch
from transformers import ViTFeatureExtractor,ViTModel
from PIL import Image
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "vit-base-patch16-224-in21k"
)
vit_model = ViTModel.from_pretrained("vit-base-patch16-224-in21k")


def vision_full_pic_embedding(video_id):
    '''
    return full pic vit model 768 
    '''
    frame_id_path = frame_path + video_id
    frames = glob(frame_id_path + '/*')
    full_feature768 = []
    for frame in sorted(frames):
        # print(frame)
        img = Image.open(frame)
        inputs = feature_extractor(img, return_tensors="pt")
        with torch.no_grad():
            outputs = vit_model(**inputs)

        pool = outputs.pooler_output
        full_feature768.append(pool)
        
        # break
    full_feature768 = torch.cat(full_feature768, 0 )
    # print(full_feature768.size())
    
    return full_feature768

vision_full_feats = []
for index in (range(len(df))):
    video_id, _, text, label, _, mode, _ = df.loc[index]
    print(index)
    f = vision_full_pic_embedding(video_id)
    vision_full_feats.append(f)
    if index == 1 :
        pass
        # break

print(len(vision_full_feats)) 


# In[10]:


def func_padding_vision_full_feature():
    max_vision_len = 0
    vision_feature_dim = vision_full_feats[0].size(1)
    video_nums = len(vision_full_feats)
    for vision in vision_full_feats:
        if vision.size(0) > max_vision_len:
            max_vision_len = vision.size(0)

    padding_vision_full_feature = torch.zeros(video_nums, max_vision_len,vision_feature_dim)
    for index, vision in enumerate(vision_full_feats):
        frames_num = vision.size(0)
        padding_vision_full_feature[index][:frames_num] = vision
        
    return padding_vision_full_feature



# # generate_graph(text) generate_cross_graph

# In[11]:


def generate_graph(text):
    from transformers import BertTokenizer
    from collections import defaultdict
    import spacy
    nlp = spacy.load('en_core_web_sm')
    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased/")
    
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
        # print(s, ' =? ', spa_)
        if s == spa_:
            split_link.append(pre)
            jj += 1
            s = ""
            pre = []

        ii += 1
    # print(text)
    # print(bert_token)
    # print(spacy_token)
    # print(split_link)
    
    doc = nlp(text)
    mat = defaultdict(list,[])
    for t in doc:
        for child in t.children:
            # print(t, t.i, child.i)
            mat[child.i].append(t.i)
            mat[t.i].append(child.i)
    
    import numpy as np
    outter_graph = np.zeros((bert_len,bert_len)).astype('float32')
    
    for key,linked in mat.items():
        for x in split_link[key]:
            for link in linked:
                for y in split_link[link]:
                    outter_graph[x][y] = 1
                
    tokens = bert_token
    inner_graph = np.identity(bert_len).astype('float32')
    for link in split_link:
        for x in link:
            for y in link:
                inner_graph[x][y] = 1
    
    outter_graph = np.pad(outter_graph, ((1,1),(1,1)), 'constant')
    inner_graph = np.pad(inner_graph, ((1,1),(1,1)), 'constant')
    inner_graph[0][0] = 1
    inner_graph[-1][-1] = 1
    graph1 = inner_graph + outter_graph
                
    # print(graph1, graph1.shape)
    return graph1

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/')
import spacy
nlp = spacy.load('en_core_web_sm')

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
    print(spacy_token)
    print(bert_token)
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

from nltk.corpus import wordnet as wn
y = 3

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

senticNet = load_sentic_word()
list(senticNet.items())[:2]

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
    graph, graph.shape
    split_link_ = get_split(text)
    split_link_
    split_link = dict()
    for idx, value in enumerate(split_link_):
        for v in value:
            split_link[bert_text_tokens[v]] = spacy_text_tokens[idx]
    split_link
    
    for i, token in enumerate(bert_text_tokens):
        cur = 0
        sp_token = split_link[token]
        si = wn.synsets(sp_token)
        if len(si) == 0 :
            continue
        si = si[0]
        for b in box_text:
            tokens_ = nlp(b)
            tokens_ = [str(x) for x in tokens_]
    #         print(tokens_)
            if len(tokens_) == 2:
                adj_j, obj_j = tokens_
            if len(tokens_) == 1:
                adj_j, obj_j = 'a', tokens_[0]
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


# # 调试生成特征

# In[12]:


for index in (range(len(df))):
    if index == 1:
        video_id, _, text, label, _, mode, _ = df.loc[index]
        # graph = generate_graph(text)


# # 生成多模态特征数组

# In[13]:


audio_features = []
modes = []
labels = []
video_features = []
box_texts =[]
graphs = []
cross_graphs = []
texts = []
for index in tqdm(range(len(df))):
    video_id, _, text, label, _, mode, _ = df.loc[index]
    audio_dir = audios_path + video_id
    audio_file = audio_dir + '/tmp.wav'
    video_file = video_path + video_id + '.mp4'
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print('makedirs ' + audio_dir)
        video_to_wav(video_file, audio_file)
    audio_feature = audio_embedding(audio_file)
    
    # vision
    video_feature, cls_names_10 = vision_10_boxes_embedding(video_id)
    graph = generate_graph(text)
    cross_graph = generate_cross_graph(text, cls_names_10)
    
    texts.append(text)
    audio_features.append(audio_feature)
    video_features.append(video_feature)
    box_texts.append(cls_names_10)
    graphs.append(graph)
    cross_graphs.append(cross_graph)
    modes.append(mode)
    labels.append(label)
    
    if index == 1:
        pass
        # break
print(len(box_texts))


# # 填充函数

# In[14]:


import numpy as np
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased/")


def merge_graph(graphs, cross_graphs):
    bert_indices_max_len = 0
    bert_indices_len = []
    for graph, cross_graph in zip(graphs, cross_graphs):
        bert_indices_len.append(graph.shape[0])
    bert_indices_max_len = max(bert_indices_len)
    
    big_graphs = []
    for graph, cross_graph in zip(graphs, cross_graphs):
        cross_graph = np.pad(cross_graph, ((1,1),(0,0)), 'constant')
        # print(graph.shape, cross_graph.shape)
        if graph.shape[0] < bert_indices_max_len:
            graph = np.pad(graph, ((0,bert_indices_max_len-graph.shape[0]),(0,bert_indices_max_len-graph.shape[0])), 'constant')
        print(graph.shape, cross_graph.shape)
        
        graph = np.pad(graph,((0,10),(0,10)),'constant')
        print(graph.shape)
        image_graph = cross_graph
        for i in range(image_graph.shape[0]-2):
            for j in range(image_graph.shape[1]):
                if not np.isnan(image_graph[i][j]):
                    # print(i+1,j+bert_indices_max_len,'|',i+1,j)
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
    bert_indices_max_len = max(lens)
    
    bert_indices_pad = [np.pad(x,(0, bert_indices_max_len - len(x)),'constant') for x in bert_indices]
    bert_indices_pad = np.array(bert_indices_pad)
    
    # batch_bert_indices.append(numpy.pad(bert_indices,(0,bert_indices_max_len - len(bert_indices)),'constant'))
    print(bert_indices_pad.shape)
    return bert_indices_pad


def padding_box_text(box_texts):
    box_bert_indices = []
    for box_text in box_texts:
        # print(box_text)
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
    
    print(box_pad_indices.shape)
    return box_pad_indices


# # 填充

# In[15]:


big_graphs = merge_graph(graphs, cross_graphs)
labels = labels_to_np(labels)
video_features_p = padding_video_features(video_features)
bert_indices = padding_text(texts)
box_pad_indices = padding_box_text(box_texts)
audio_padding_fea, audio_length_array = padding_sequence(audio_features)
padding_vision_full_feature = func_padding_vision_full_feature()


# # 整理数据 train valid test 

# In[16]:


import pickle

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
    

with open(output_path, 'wb') as f:
    pickle.dump(final_data, f)
print('write to', output_path)
get_file_size(output_path, "MB")


# In[21]:


video_features_p.shape, padding_vision_full_feature.shape

