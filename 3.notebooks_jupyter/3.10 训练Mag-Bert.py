#!/usr/bin/env python
# coding: utf-8

# # 准备数据集

# In[1]:


'''
dataset.py
'''
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader

class MUStartDataset(Dataset):
    def __init__(self,mode = 'train',feature_path = './featuresIndepResnet152.pkl'):
        with open(feature_path,'rb') as f:
            import pickle
            data = pickle.load(f)
        self.feature_dict = data[mode]
        # [-1,1] -> [0,1]
        self.feature_dict['labels'] = ((self.feature_dict['labels'] + 1)/2).astype(np.int64)

    def __getitem__(self,index):
        feature ={}
        feature['audio_feature'] = self.feature_dict['audio_feature'][index]
        feature['video_features_p'] = self.feature_dict['video_features_p'][index]
        feature['bert_indices'] = self.feature_dict['bert_indices'][index]
        feature['box_pad_indices'] = self.feature_dict['box_pad_indices'][index]
        feature['big_graphs'] = self.feature_dict['big_graphs'][index]
        feature['labels'] = self.feature_dict['labels'][index]
        feature['index'] = index
        
        return feature
    def __len__(self):
        labels = self.feature_dict['labels']
        length = labels.shape[0]
        return length
    
    def get_sample_shape(self,index):
        shape_dict = {}
        shape_dict['audio_feature'] = self.feature_dict['audio_feature'][index].shape
        shape_dict['video_features_p'] = self.feature_dict['video_features_p'][index].shape
        shape_dict['bert_indices'] = self.feature_dict['bert_indices'][index].shape
        shape_dict['box_pad_indices'] = self.feature_dict['box_pad_indices'][index].shape
        shape_dict['big_graphs'] = self.feature_dict['big_graphs'][index].shape
        # shape_dict['labels'] = self.feature_dict['labels'][index].shape
        shape_dict['labels'] = type(self.feature_dict['labels'][index])
        return shape_dict
        

if __name__=="__main__":
    d = MUStartDataset('valid')
    dl = DataLoader(d, batch_size=2, num_workers=0, shuffle=False)
    batch_sample = iter(dl).next()
    print(batch_sample.keys())
    print(batch_sample['audio_feature'].size(2))
    print(batch_sample['video_features_p'].size(2))


# # 准备模型

# In[2]:


import torch
import torch.nn as nn
from transformers import BertModel
from layers.dynamic_rnn import DynamicLSTM
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics

class MAG(nn.Module):
    def __init__(self, hidden_size, beta_shift, dropout_prob):        
        super(MAG, self).__init__()
        print("Initializing MAG with beta_shift:{} hidden_prob:{}".format(beta_shift, dropout_prob))

        self.W_hv = nn.Linear(VISUAL_DIM + TEXT_DIM, TEXT_DIM)
        self.W_ha = nn.Linear(ACOUSTIC_DIM + TEXT_DIM, TEXT_DIM)
        self.W_v = nn.Linear(VISUAL_DIM, TEXT_DIM)
        self.W_a = nn.Linear(ACOUSTIC_DIM, TEXT_DIM)
        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        weight_v = F.relu(self.W_hv(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))
        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        DEVICE = visual.device
        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(DEVICE)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(DEVICE)
        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)
        acoustic_vis_embedding = alpha * h_m
        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        return embedding_output


from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers import BertTokenizer

class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
        
class MAG_BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.MAG = MAG(
            config.hidden_size,
            multimodal_config.beta_shift,
            multimodal_config.dropout_prob,
        )

        self.init_weights()
        
    def forward(
    self,
    input_ids,
    visual,
    acoustic,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    singleTask = False,
    ):
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        fused_embedding = self.MAG(embedding_output, visual, acoustic)
        
        encoder_outputs = self.encoder(
            fused_embedding,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs
        

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias :
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        
    def forward(self, text, adj):
        hidden = torch.matmul(text,self.weight)
        
        denom = torch.sum(adj,dim=2,keepdim=True) + 1
        output = torch.matmul(adj, hidden.float())/denom
        if self.bias is not None:
            output = output + self.bias

        return output

import torch.nn as nn

class AlignSubNet(nn.Module):
    def __init__(self, dst_len):
        """
        mode: the way of aligning avg_pool 这个模型并没有参数
        """
        super(AlignSubNet, self).__init__()
        self.dst_len = dst_len

    def get_seq_len(self):
        return self.dst_len
    
    def __avg_pool(self, text_x, audio_x, video_x):
        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len:
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x
        text_x = align(text_x)
        audio_x = align(audio_x)
        video_x = align(video_x)
        return text_x, audio_x, video_x
    
 
    def forward(self, text_x, audio_x, video_x):
        if text_x.size(1) == audio_x.size(1) == video_x.size(1):
            return text_x, audio_x, video_x
        return self.__avg_pool(text_x, audio_x, video_x)

    
class CMGCN(nn.Module):
    def __init__(self):
        super(CMGCN, self).__init__()
        print('create CMGCN model')
        self.bert = BertModel.from_pretrained('./bert-base-uncased/')
        self.text_lstm = DynamicLSTM(768,4,num_layers=1,batch_first=True,bidirectional=True)
        self.vit_fc = nn.Linear(768,2*4)
        self.gc1 = GraphConvolution(2*4, 2*4)
        self.gc2 = GraphConvolution(2*4, 2*4)
        self.fc = nn.Linear(2*4,2)
        
    def forward(self, inputs):
        bert_indices = inputs['bert_indices']
        graph = inputs['big_graphs']
        box_vit = inputs['video_features_p']
        bert_text_len = torch.sum(bert_indices != 0, dim = -1)
        outputs = self.bert(bert_indices)
        encoder_layer = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        text_out, (_, _) = self.text_lstm(encoder_layer, bert_text_len)
        # 与原始代码不同，这里因为进行了全局的特征填充，导致text_out可能无法达到填充长度，补充为0
        if text_out.shape[1] < encoder_layer.shape[1]:
            pad = torch.zeros((text_out.shape[0],encoder_layer.shape[1]-text_out.shape[1],text_out.shape[2]))
            text_out = torch.cat((text_out,pad),dim=1)

        box_vit = box_vit.float()
        box_vit = self.vit_fc(box_vit)
        features = torch.cat([text_out, box_vit], dim=1)

        graph = graph.float()
        x = F.relu(self.gc1(features, graph))
        x = F.relu(self.gc2(x,graph))
        
        alpha_mat = torch.matmul(features,x.transpose(1,2))
        alpha_mat = alpha_mat.sum(1, keepdim=True)
        alpha = F.softmax(alpha_mat, dim = 2)
        x = torch.matmul(alpha, x).squeeze(1)
        
        output = self.fc(x)
        return output
    
    
class CMGCNI(nn.Module):
    def __init__(self, multimodal_config):
        super(CMGCNI, self).__init__()
        print('create CMGCNI model')
        self.mag_bert = MAG_BertModel.from_pretrained('./bert-base-uncased/',multimodal_config=multimodal_config)
        self.text_lstm = DynamicLSTM(768,4,num_layers=1,batch_first=True,bidirectional=True)
        self.vit_fc = nn.Linear(768,2*4)
        self.gc1 = GraphConvolution(2*4, 2*4)
        self.gc2 = GraphConvolution(2*4, 2*4)
        self.fc = nn.Linear(2*4,2)
        
        
    def forward(self, inputs):
        bert_indices = inputs['bert_indices']
        graph = inputs['big_graphs']
        box_vit = inputs['video_features_p']
        bert_text_len = torch.sum(bert_indices != 0, dim = -1)
        # 2,24, audio_feature key 2 33 33 , 2,10 768 
        visual = box_vit
        acoustic = inputs['audio_feature']
        self.align_subnet = AlignSubNet(bert_indices.size(1))
        bert_indices, acoustic, visual= self.align_subnet(bert_indices,acoustic,visual)
        acoustic = acoustic.float()
        outputs = self.mag_bert(bert_indices, visual, acoustic)
        
        encoder_layer = outputs[0]
        pooled_output = outputs[1]
        
        text_out, (_, _) = self.text_lstm(encoder_layer, bert_text_len)
        # 与原始代码不同，这里因为进行了全局的特征填充，导致text_out可能无法达到填充长度，补充为0
        if text_out.shape[1] < encoder_layer.shape[1]:
            pad = torch.zeros((text_out.shape[0],encoder_layer.shape[1]-text_out.shape[1],text_out.shape[2]))
            text_out = torch.cat((text_out,pad),dim=1)

        box_vit = box_vit.float()
        box_vit = self.vit_fc(box_vit)
        features = torch.cat([text_out, box_vit], dim=1)

        graph = graph.float()
        x = F.relu(self.gc1(features, graph))
        x = F.relu(self.gc2(x,graph))
        
        alpha_mat = torch.matmul(features,x.transpose(1,2))
        alpha_mat = alpha_mat.sum(1, keepdim=True)
        alpha = F.softmax(alpha_mat, dim = 2)
        x = torch.matmul(alpha, x).squeeze(1)
        
        output = self.fc(x)
        return output

class MagBertForSequenceClassification(nn.Module):
    def __init__(self, multimodal_config):
        super(MagBertForSequenceClassification, self).__init__()
        print('create MagBertForSequenceClassification model')
        self.mag_bert = MAG_BertModel.from_pretrained('./bert-base-uncased/',multimodal_config=multimodal_config)
        self.dropout = nn.Dropout(0.1) # bert config 中的设置
        self.classifier = nn.Linear(768,2)
        
    def forward(self, inputs):
        bert_indices = inputs['bert_indices']
        box_vit = inputs['video_features_p']
        bert_text_len = torch.sum(bert_indices != 0, dim = -1)
        # 2,24, audio_feature key 2 33 33 , 2,10 768 
        visual = box_vit
        acoustic = inputs['audio_feature']
        self.align_subnet = AlignSubNet(bert_indices.size(1))
        bert_indices, acoustic, visual= self.align_subnet(bert_indices,acoustic,visual)

        acoustic = acoustic.float()
        
        outputs = self.mag_bert(bert_indices, visual, acoustic)
        
        pooled_output = outputs[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        output = logits
        return output
    
class EF_LSTM(nn.Module):
    def __init__(self):
        super(EF_LSTM, self).__init__()
        print('create EF_LSTM model')
        self.bert = BertModel.from_pretrained('./bert-base-uncased/')
        self.norm = nn.BatchNorm1d(TEXT_SEQ_LEN)
        self.lstm = nn.LSTM(ACOUSTIC_DIM+VISUAL_DIM+TEXT_DIM, 64, num_layers=2, dropout=0.3, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(64,64)
        self.out = nn.Linear(64,2)
        
    def forward(self, inputs):
        bert_indices = inputs['bert_indices']
        # graph = inputs['big_graphs']
        box_vit = inputs['video_features_p']
        bert_text_len = torch.sum(bert_indices != 0, dim = -1)
        # 2,24, audio_feature key 2 33 33 , 2,10 768 
        visual = box_vit
        acoustic = inputs['audio_feature']
        self.align_subnet = AlignSubNet(bert_indices.size(1))
        bert_indices, acoustic, visual= self.align_subnet(bert_indices,acoustic,visual)
        acoustic = acoustic.float()
        outputs = self.bert(bert_indices)
        encoder_layer = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        text_x = encoder_layer
        audio_x = acoustic
        video_x = visual
        x = torch.cat([text_x, audio_x, video_x], dim=-1)
        x = self.norm(x)
        _, final_states = self.lstm(x)
        x = self.dropout(final_states[0][-1].squeeze(dim=0))
        x = F.relu(self.linear(x), inplace=True)
        x = self.dropout(x)
        output = self.out(x)
        
        return output

from models.subNets.FeatureNets import SubNet,TextSubNet

class LF_DNN(nn.Module):
    def __init__(self):
        super(LF_DNN, self).__init__()
        print('create LF_DNN model')
        self.bert = BertModel.from_pretrained('./bert-base-uncased/')
        self.audio_in = ACOUSTIC_DIM
        self.video_in = VISUAL_DIM
        self.text_in = TEXT_DIM
        self.audio_hidden = 8
        self.video_hidden = 64
        self.text_hidden = 64
        self.text_out = 32
        self.post_fusion_dim = 32
        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = 0.2, 0.2, 0.2, 0.2
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden + self.audio_hidden,
                                             self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 2)
        
        
    def forward(self, inputs):
        bert_indices = inputs['bert_indices']
        box_vit = inputs['video_features_p']
        bert_text_len = torch.sum(bert_indices != 0, dim = -1)
        # 2,24, audio_feature key 2 33 33 , 2,10 768 
        visual = box_vit
        acoustic = inputs['audio_feature']
        self.align_subnet = AlignSubNet(bert_indices.size(1))
        bert_indices, acoustic, visual= self.align_subnet(bert_indices,acoustic,visual)
        acoustic = acoustic.float()
        outputs = self.bert(bert_indices)
        encoder_layer = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        text_x = encoder_layer
        audio_x = acoustic
        video_x = visual
        audio_x = torch.mean(audio_x,1,True)
        video_x = torch.mean(video_x,1,True)
        audio_x[audio_x != audio_x] = 0
        video_x[video_x != video_x] = 0
        audio_x = audio_x.squeeze(1)
        video_x = video_x.squeeze(1)
        
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        
        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)

        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output = self.post_fusion_layer_3(x)
               
        return output

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
    
    
class SELF_MM(nn.Module):
    def __init__(self):
        super(SELF_MM, self).__init__()
        print('create SELF_MM model')
        
        self.audio_in = ACOUSTIC_DIM
        self.video_in = VISUAL_DIM
        self.text_in = TEXT_DIM
        self.audio_hidden = 8
        self.video_hidden = 64
        self.text_hidden = 64
        self.text_out = 768
        self.post_text_dim = 32
        self.audio_out = 8
        self.post_audio_dim = 8 
        self.video_out = 32
        self.post_video_dim = 32 
        self.post_fusion_dim = 32
        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = 0.2, 0.2, 0.2, 0.2
        
        self.bert = BertModel.from_pretrained('./bert-base-uncased/')
        self.audio_model = AuViSubNet(self.audio_in, self.audio_hidden, self.audio_out, dropout = self.audio_prob)
        self.video_model = AuViSubNet(self.video_in, self.video_hidden, self.video_out, dropout = self.video_prob)

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_out + self.audio_out,
                                             self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)
        

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=self.text_prob)
        self.post_text_layer_1 = nn.Linear(self.text_out, self.post_text_dim)
        self.post_text_layer_2 = nn.Linear(self.post_text_dim, self.post_text_dim)
        self.post_text_layer_3 = nn.Linear(self.post_text_dim, 1)
        
        
        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=self.audio_prob)
        self.post_audio_layer_1 = nn.Linear(self.audio_out, self.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(self.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(self.post_audio_dim, 1)

        # the classify layer for video

        self.post_video_dropout = nn.Dropout(p=self.video_prob)
        self.post_video_layer_1 = nn.Linear(self.video_out, self.post_video_dim)
        self.post_video_layer_2 = nn.Linear(self.post_video_dim, self.post_video_dim)
        self.post_video_layer_3 = nn.Linear(self.post_video_dim, 1)
        
        
    def forward(self, inputs):
        bert_indices = inputs['bert_indices']
        box_vit = inputs['video_features_p']
        bert_text_len = torch.sum(bert_indices != 0, dim = -1)
        # 2,24, audio_feature key 2 33 33 , 2,10 768 
        visual = box_vit
        acoustic = inputs['audio_feature']
        self.align_subnet = AlignSubNet(bert_indices.size(1))
        bert_indices, acoustic, visual= self.align_subnet(bert_indices,acoustic,visual)
        acoustic = acoustic.float()
        outputs = self.bert(bert_indices)
        encoder_layer = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        text_x = encoder_layer
        audio_x = acoustic
        video_x = visual
        
        text = pooled_output
        audio = self.audio_model(audio_x, bert_text_len)
        video = self.video_model(video_x, bert_text_len)
        
        # fusion
        fusion_h = torch.cat([text, audio, video], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        # # text
        text_h = self.post_text_dropout(text)
        text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
        # audio
        audio_h = self.post_audio_dropout(audio)
        audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
        # vision
        video_h = self.post_video_dropout(video)
        video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)

        # classifier-fusion
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        # classifier-text
        x_t = F.relu(self.post_text_layer_2(text_h), inplace=False)
        output_text = self.post_text_layer_3(x_t)

        # classifier-audio
        x_a = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        output_audio = self.post_audio_layer_3(x_a)

        # classifier-vision
        x_v = F.relu(self.post_video_layer_2(video_h), inplace=False)
        output_video = self.post_video_layer_3(x_v)
        
        res = {
            'M': output_fusion, 
            'T': output_text,
            'A': output_audio,
            'V': output_video,
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
        }
        return res
    
    


# # 初始模型参数，

# In[3]:


def init_cmgcn_improve_params(cmgcni_model):
    for child in cmgcni_model.children():
        # print(type(child) != BertModel)
        if type(child) != MAG_BertModel:
            for p in child.parameters():
                if p.requires_grad :
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        import math
                        stdv = 1.0 / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print('init_cmgcn_improve_params()')

def init_cmgcn_params(cmgcn_model):
    for child in cmgcn_model.children():
        # print(type(child) != BertModel)
        if type(child) != BertModel:
            for p in child.parameters():
                if p.requires_grad :
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        import math
                        stdv = 1.0 / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print('init_cmgcn_params()')


def init_magbert_params(magbert_forseqcls):
    for child in magbert_forseqcls.children():
        if type(child) != MAG_BertModel:
            for p in child.parameters():
                if p.requires_grad :
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        import math
                        stdv = 1.0 / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print('init_magbert_params()')

def init_ef_lstm_params(ef_lstm_model):
    for child in ef_lstm_model.children():
        if type(child) != BertModel:
            for p in child.parameters():
                if p.requires_grad :
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        import math
                        stdv = 1.0 / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print('init_ef_lstm_params()')
    
def init_lf_dnn_params(lf_dnn_model):
    for child in lf_dnn_model.children():
        if type(child) != BertModel:
            for p in child.parameters():
                if p.requires_grad :
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        import math
                        stdv = 1.0 / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print('init_lf_dnn_params()')
    


def init_self_mm_params(self_mm_model):
    for child in self_mm_model.children():
        if type(child) != BertModel:
            for p in child.parameters():
                if p.requires_grad :
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        import math
                        stdv = 1.0 / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    print('init_self_mm_params()')


# # 准备optimizer函数，

# In[4]:


def get_cmgcn_improve_optimizer(cmgcni_model):
    optimizer = torch.optim.Adam([
        {'params':cmgcni_model.mag_bert.parameters(),'lr':2e-5},
        {'params':cmgcni_model.text_lstm.parameters(),},
        {'params':cmgcni_model.vit_fc.parameters(),},
        {'params':cmgcni_model.gc1.parameters(),},
        {'params':cmgcni_model.gc2.parameters(),},
        {'params':cmgcni_model.fc.parameters(),},
    ],lr=0.001,weight_decay=1e-5)
    # optimizer = torch.optim.Adam(cmgcn_model.parameters(),lr=1e-3,weight_decay=1e-5)
    return optimizer

def get_cmgcn_optimizer(cmgcn_model):
    optimizer = torch.optim.Adam([
        {'params':cmgcn_model.bert.parameters(),'lr':2e-5},
        {'params':cmgcn_model.text_lstm.parameters(),},
        {'params':cmgcn_model.vit_fc.parameters(),},
        {'params':cmgcn_model.gc1.parameters(),},
        {'params':cmgcn_model.gc2.parameters(),},
        {'params':cmgcn_model.fc.parameters(),},
    ],lr=0.001,weight_decay=1e-5)
    # optimizer = torch.optim.Adam(cmgcn_model.parameters(),lr=1e-3,weight_decay=1e-5)
    return optimizer

def get_magbert_optimizer(magbert_forseqcls):
    optimizer = torch.optim.Adam([
        {'params':magbert_forseqcls.mag_bert.parameters(),'lr':1e-5},
        {'params':magbert_forseqcls.classifier.parameters(),},
        {'params':magbert_forseqcls.dropout.parameters(),},
    ],lr=0.001,weight_decay=1e-5)
    return optimizer

def get_ef_lstm_optimizer(ef_lstm_model):
    optimizer = torch.optim.Adam([
        {'params':ef_lstm_model.bert.parameters(),'lr':1e-5},
        {'params':ef_lstm_model.norm.parameters(),},
        {'params':ef_lstm_model.lstm.parameters(),},
        {'params':ef_lstm_model.dropout.parameters(),},
        {'params':ef_lstm_model.linear.parameters(),},
        {'params':ef_lstm_model.out.parameters(),},
    ],lr=0.001,weight_decay=1e-5)
    return optimizer

def get_lf_dnn_optimizer(lf_dnn_model):
    optimizer = torch.optim.Adam([
        {'params':lf_dnn_model.bert.parameters(),'lr':1e-5},
        {'params':lf_dnn_model.audio_subnet.parameters(),},
        {'params':lf_dnn_model.video_subnet.parameters(),},
        {'params':lf_dnn_model.text_subnet.parameters(),},
        {'params':lf_dnn_model.post_fusion_dropout.parameters(),},
        {'params':lf_dnn_model.post_fusion_layer_1.parameters(),},
        {'params':lf_dnn_model.post_fusion_layer_2.parameters(),},
        {'params':lf_dnn_model.post_fusion_layer_3.parameters(),},
    ],lr=0.001,weight_decay=1e-5)
    return optimizer

def get_self_mm_optimizer(self_mm_model):
    optimizer = torch.optim.Adam([
        {'params':self_mm_model.bert.parameters(),'lr':1e-5},
        {'params':self_mm_model.audio_model.parameters(),},
        {'params':self_mm_model.video_model.parameters(),},
        
        {'params':self_mm_model.post_fusion_dropout.parameters(),},
        {'params':self_mm_model.post_fusion_layer_1.parameters(),},
        {'params':self_mm_model.post_fusion_layer_2.parameters(),},
        {'params':self_mm_model.post_fusion_layer_3.parameters(),},
        
        {'params':self_mm_model.post_text_dropout.parameters(),},
        {'params':self_mm_model.post_text_layer_1.parameters(),},
        {'params':self_mm_model.post_text_layer_2.parameters(),},
        {'params':self_mm_model.post_text_layer_3.parameters(),},
        
        {'params':self_mm_model.post_audio_dropout.parameters(),},
        {'params':self_mm_model.post_audio_layer_1.parameters(),},
        {'params':self_mm_model.post_audio_layer_2.parameters(),},
        {'params':self_mm_model.post_audio_layer_3.parameters(),},
        
        {'params':self_mm_model.post_video_dropout.parameters(),},
        {'params':self_mm_model.post_video_layer_1.parameters(),},
        {'params':self_mm_model.post_video_layer_2.parameters(),},
        {'params':self_mm_model.post_video_layer_3.parameters(),},
        
    ],lr=0.001,weight_decay=1e-5)
    return optimizer



# # 调试模型的形状函数

# In[5]:


def debug_model(model):
    d = MUStartDataset('valid')
    dl = DataLoader(d, batch_size=2, num_workers=0, shuffle=False)
    batch = iter(dl).next()
    batch.keys()
    inputs ={}
    for key in batch.keys():
        inputs[key] = batch[key].to(device)
    outputs = model(inputs)
    print('debug_model: ',type(model),outputs.shape)
    

def debug_self_mm_model(model):
    d = MUStartDataset('train')
    dl = DataLoader(d, batch_size=2, num_workers=0, shuffle=False)
    batch = iter(dl).next()
    batch.keys()
    inputs ={}
    for key in batch.keys():
        inputs[key] = batch[key].to(device)
    outputs = model(inputs)
    print('debug_model: ',type(model))
    print([(k,v.shape) for k,v in outputs.items()])
    
    
    


# # train and eval

# In[6]:


def evaluate_acc_f1(data_loader,model):
    n_correct, n_total = 0, 0
    targets_all, outputs_all = None, None
    model.eval()
    with torch.no_grad():
        for i_batch,batch in enumerate(data_loader):
            inputs ={}
            for key in batch.keys():
                inputs[key] = batch[key].to(device)
            outputs = model(inputs)
            targets = batch['labels'].to(device)
            
            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            
            if targets_all is None:
                targets_all = targets
                outputs_all = outputs
            else:
                targets_all = torch.cat((targets_all,targets), dim=0)
                outputs_all = torch.cat((outputs_all,outputs), dim=0)
    

    
    acc = n_correct / n_total
    f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all,-1).cpu(), labels=[0,1], average='macro', zero_division=0)
    precision = metrics.precision_score(targets_all.cpu(), torch.argmax(outputs_all,-1).cpu(), labels=[0,1], average='macro', zero_division=0)
    recall = metrics.recall_score(targets_all.cpu(), torch.argmax(outputs_all,-1).cpu(), labels=[0,1], average='macro', zero_division=0)

    return acc,f1,precision,recall

def evaluate_self_mm_acc_f1(data_loader,model):
    n_correct, n_total = 0, 0
    targets_all, outputs_all = None, None
    model.eval()
    with torch.no_grad():
        for i_batch,batch in enumerate(data_loader):
            inputs ={}
            for key in batch.keys():
                inputs[key] = batch[key].to(device)
            outputs = model(inputs)
            targets = batch['labels'].to(device)
            output_label = outputs['M'].view(-1)
            output_label[output_label>=0.5] = 1.0
            output_label[output_label<0.5] = 0.0
            
            n_correct += (output_label == targets).sum().item()
            n_total += len(outputs['M'])
            
            if targets_all is None:
                targets_all = targets
                outputs_all = outputs['M']
            else:
                targets_all = torch.cat((targets_all,targets), dim=0)
                outputs_all = torch.cat((outputs_all,outputs['M']), dim=0)
    
    
    acc = n_correct / n_total
    f1 = metrics.f1_score(targets_all.cpu(), outputs_all.cpu(), labels=[0,1], average='macro', zero_division=0)
    precision = metrics.precision_score(targets_all.cpu(), outputs_all.cpu(), labels=[0,1], average='macro', zero_division=0)
    recall = metrics.recall_score(targets_all.cpu(), outputs_all.cpu(), labels=[0,1], average='macro', zero_division=0)

    return acc,f1,precision,recall



def train(model,optimizer,model_save_path):
    max_val_acc , max_val_f1, max_val_epoch, global_step = 0, 0, 0, 0
    for i_epoch in range(num_epoch):
        print('i_epoch:', i_epoch)
        n_correct, n_total, loss_total = 0, 0, 0
        for i_batch,batch in enumerate(train_dataloader):
            global_step += 1
            model.train()
            optimizer.zero_grad()
            inputs ={}
            for key in batch.keys():
                inputs[key] = batch[key].to(device)
            outputs = model(inputs)
            targets = batch['labels'].to(device)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
            n_total += len(outputs)
            loss_total += loss.item() * len(outputs)

            train_acc = n_correct / n_total
            train_loss = loss_total / n_total

            if global_step % 1 == 0:
                val_acc, val_f1, val_precision, val_recall = evaluate_acc_f1(valid_dataloader,model)
                if val_acc >= max_val_acc:
                    max_val_f1 = val_f1
                    max_val_acc = val_acc
                    max_val_epoch = i_epoch
                    torch.save(model.state_dict(),model_save_path)
                    print('save the model to {}'.format(model_save_path))

        if i_epoch - max_val_epoch >= 0:
            print('early stop')
            break

        break
    model.load_state_dict(torch.load(model_save_path))
    test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(test_dataloader,model)
    print('test_acc:', test_acc)
    print('test_f1:', test_f1)
    print('test_precision', test_precision)
    print('test_recall', test_recall)


def train_self_mm(model,optimizer,model_save_path):
    # 多任务self逻辑
    train_samples = len(train_dataset)
    label_map = {
        'fusion': torch.zeros(train_samples, requires_grad=False).to(device),
        'text': torch.zeros(train_samples, requires_grad=False).to(device),
        'audio': torch.zeros(train_samples, requires_grad=False).to(device),
        'vision': torch.zeros(train_samples, requires_grad=False).to(device)
    }
    # init labels
    for batch_data in train_dataloader:
        labels_m = batch_data['labels'].float()
        label_map['fusion'] = labels_m
        label_map['text'] = labels_m
        label_map['audion'] = labels_m
        label_map['vision'] = labels_m
        
    max_val_acc , max_val_f1, max_val_epoch, global_step = 0, 0, 0, 0
    for i_epoch in range(num_epoch):
        epochs = i_epoch + 1             # for same with self_mm original code 
        print('epochs:', epochs)
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        train_loss = 0.0
        
        n_correct, n_total, loss_total = 0, 0, 0
        for i_batch,batch in enumerate(train_dataloader):
            global_step += 1
            model.train()
            optimizer.zero_grad()
            inputs ={}
            for key in batch.keys():
                inputs[key] = batch[key].to(device)
                
            indexes = batch['index'].view(-1)
            outputs = model(inputs)
            
            tasks = 'MTAV'
            name_map = {
                'M': 'fusion',
                'T': 'text',
                'A': 'audio',
                'V': 'vision'
            }
            for m in tasks:
                y_pred[m].append(outputs[m].cpu())
                y_true[m].append(label_map[name_map[m]][indexes].cpu())
                
            loss = 0.0
            def weighted_loss(y_pred,y_true,indexes=None, mode = 'fusion'):
                if mode == 'fusion':
                    weighted = torch.ones_like(y_pred)
                else:
                    weighted = torch.tanh(torch.abs(label_map[mode][indexes] - label_map['fusion'][indexes]))
                loss = torch.mean(weighted * torch.abs(y_pred - y_true))
                return loss
                
            for m in tasks:
                loss += weighted_loss(outputs[m],label_map[name_map[m]][indexes],indexes = indexes, mode = name_map[m])
            
            loss.backward()
            train_loss += loss.item()
            # update features 
            f_fusion = outputs['Feature_f'].detach()
            f_text = outputs['Feature_t'].detach()
            f_audio = outputs['Feature_a'].detach()
            f_vision = outputs['Feature_v'].detach()
            def update_labels(f_fusion, f_text, f_audio, f_vision, cur_epoches, indexes, outputs):
                MIN = 1e-8
                def update_single_label(f_single, mode):
                    d_sp = torch.norm(f_single - center_map[mode]['pos'], dim=-1) 
                    d_sn = torch.norm(f_single - center_map[mode]['neg'], dim=-1) 
                    delta_s = (d_sn - d_sp) / (d_sp + MIN)
                    alpha = delta_s / (delta_f + MIN)
                    
                    new_labels = 0.5 * alpha * label_map['fusion'][indexes] + \
                        0.5 * (label_map['fusion'][indexes] + delta_s - delta_f)
                    new_labels = torch.clamp(new_labels, min=0.0, max=1.0)
                    
                    n = cur_epoches
                    label_map[mode][indexes] = (n - 1) / (n + 1) * label_map[mode][indexes] \
                      + 2 / (n + 1) * new_labels
                    
                d_fp = torch.norm(f_fusion - center_map['fusion']['pos'], dim=-1)
                d_fn = torch.norm(f_fusion - center_map['fusion']['neg'], dim=-1) 
                delta_f = (d_fn - d_fp) / (d_fp + MIN)
                
                update_single_label(f_text, mode='text')
                update_single_label(f_audio, mode='audio')
                update_single_label(f_vision, mode='vision')
            
            if epochs > 1:
                update_labels(f_fusion, f_text, f_audio, f_vision, epochs, indexes, outputs)
            
            print(211, label_map)
            post_fusion_dim = 32
            post_text_dim = 32
            post_audio_dim = 8
            post_video_dim = 32
            feature_map = {
                'fusion': torch.zeros(train_samples, post_fusion_dim, requires_grad=False).to(device),
                'text': torch.zeros(train_samples, post_text_dim, requires_grad=False).to(device),
                'audio': torch.zeros(train_samples, post_audio_dim, requires_grad=False).to(device),
                'vision': torch.zeros(train_samples, post_video_dim, requires_grad=False).to(device),
            }
            def update_features(f_fusion, f_text, f_audio, f_vision, indexes):
                feature_map['fusion'][indexes] = f_fusion
                feature_map['text'][indexes] = f_text
                feature_map['audio'][indexes] = f_audio
                feature_map['vision'][indexes] = f_vision
            
            center_map = {
                'fusion': {
                    'pos': torch.zeros(post_fusion_dim, requires_grad=False).to(device),
                    'neg': torch.zeros(post_fusion_dim, requires_grad=False).to(device),
                },
                'text': {
                    'pos': torch.zeros(post_text_dim, requires_grad=False).to(device),
                    'neg': torch.zeros(post_text_dim, requires_grad=False).to(device),
                },
                'audio': {
                    'pos': torch.zeros(post_audio_dim, requires_grad=False).to(device),
                    'neg': torch.zeros(post_audio_dim, requires_grad=False).to(device),
                },
                'vision': {
                    'pos': torch.zeros(post_video_dim, requires_grad=False).to(device),
                    'neg': torch.zeros(post_video_dim, requires_grad=False).to(device),
                }
            }
            def update_centers():
                def update_single_center(mode):
                    neg_indexes = label_map[mode] <= 0.5   # [0 1] label 
                    pos_indexes = label_map[mode] > 0.5
                    if torch.any(pos_indexes):
                        center_map[mode]['pos'] = torch.mean(feature_map[mode][pos_indexes], dim=0)
                    if torch.any(neg_indexes):
                        center_map[mode]['neg'] = torch.mean(feature_map[mode][neg_indexes], dim=0)
                    # 如果样本只有正例 ，则负例会出现nan的情况
                    
                update_single_center(mode='fusion')
                update_single_center(mode='text')
                update_single_center(mode='audio')
                update_single_center(mode='vision')
                
                
            update_features(f_fusion, f_text, f_audio, f_vision, indexes)
            update_centers()
            optimizer.step()
            
            targets = batch['labels'].to(device)
            # outputs['M'] >= 0.5 1 
            # outputs['M'] < 0.5 0 
            output_label = outputs['M'].view(-1)
            output_label[output_label>=0.5] = 1.0
            output_label[output_label<0.5] = 0.0

            n_correct += (output_label == targets).sum().item()
            n_total += len(outputs['M'])
            loss_total += train_loss * len(outputs['M'])

            train_acc = n_correct / n_total
            train_loss = loss_total / n_total

            if global_step % 1 == 0:
                val_acc, val_f1, val_precision, val_recall = evaluate_self_mm_acc_f1(valid_dataloader,model)
                if val_acc >= max_val_acc:
                    max_val_f1 = val_f1
                    max_val_acc = val_acc
                    max_val_epoch = i_epoch
                    torch.save(model.state_dict(),model_save_path)
                    print('save the model to {}'.format(model_save_path))

        if i_epoch - max_val_epoch > 0:
            print('early stop')
            break

        # break
    model.load_state_dict(torch.load(model_save_path))
    test_acc, test_f1,test_precision,test_recall = evaluate_self_mm_acc_f1(test_dataloader,model)
    print('test_acc:', test_acc)
    print('test_f1:', test_f1)
    print('test_precision', test_precision)
    print('test_recall', test_recall)


# # 准备训练不同模型的函数

# In[7]:


# prepare model 
def train_cmgcni_model():
    cmgcni_model = CMGCNI(multimodal_config = multimodal_config ).to(device)
    init_cmgcn_improve_params(cmgcni_model) 
    cmgcni_optimizer = get_cmgcn_improve_optimizer(cmgcni_model)
    debug_model(cmgcni_model)
    cmgcni_model_path = '/tmp/cmgcni_model.pth'

    model = cmgcni_model
    optimizer = cmgcni_optimizer
    model_save_path = cmgcni_model_path
    print('start train:' + '-'*10)
    train(model,optimizer,model_save_path)

# prepare model 
def train_cmgcn_model():
    cmgcn_model = CMGCN().to(device)
    init_cmgcn_params(cmgcn_model) 
    cmgcn_optimizer = get_cmgcn_optimizer(cmgcn_model)
    debug_model(cmgcn_model)
    cmgcn_model_path = '/tmp/cmgcn_model.pth'

    model = cmgcn_model
    optimizer = cmgcn_optimizer
    model_save_path = cmgcn_model_path
    print('start train:' + '-'*10)
    train(model,optimizer,model_save_path)

def train_magbert_forseqcls():
    magbert_forseqcls = MagBertForSequenceClassification(multimodal_config = multimodal_config ).to(device)
    init_magbert_params(magbert_forseqcls)
    magbert_optimizer = get_magbert_optimizer(magbert_forseqcls)
    debug_model(magbert_forseqcls)
    magbert_model_path = '/tmp/magbert_model.pth'

    model = magbert_forseqcls
    optimizer = magbert_optimizer
    model_save_path = magbert_model_path
    print('start train:' + '-'*10)
    train(model,optimizer,model_save_path)

def train_ef_lstm_model():
    ef_lstm_model = EF_LSTM().to(device)
    init_ef_lstm_params(ef_lstm_model)
    ef_lstm_optimizer = get_ef_lstm_optimizer(ef_lstm_model)
    debug_model(ef_lstm_model)
    ef_lstm_model_path = '/tmp/ef_lstm_model.pth'

    model = ef_lstm_model
    optimizer = ef_lstm_optimizer
    model_save_path = ef_lstm_model_path
    print('start train:' + '-'*10)
    train(model,optimizer,model_save_path)
    
def train_lf_dnn_model():
    lf_dnn_model = LF_DNN().to(device)
    init_lf_dnn_params(lf_dnn_model)
    lf_dnn_optimizer = get_lf_dnn_optimizer(lf_dnn_model)
    debug_model(lf_dnn_model)
    lf_dnn_model_path = '/tmp/lf_dnn_model.pth'

    model = lf_dnn_model
    optimizer = lf_dnn_optimizer
    model_save_path = lf_dnn_model_path
    print('start train:' + '-'*10)
    train(model,optimizer,model_save_path)
    
def train_self_mm_model():
    self_mm_model = SELF_MM().to(device)
    init_self_mm_params(self_mm_model)
    self_mm_optimizer = get_self_mm_optimizer(self_mm_model)
    debug_self_mm_model(self_mm_model)
    self_mm_model_path = '/tmp/self_mm_model.pth'

    model = self_mm_model
    optimizer = self_mm_optimizer
    model_save_path = self_mm_model_path
    print('start train:' + '-'*10)
    train_self_mm(model,optimizer,model_save_path)


# # 开始训练

# In[8]:


beta_shift = 1.0 
dropout_prob = 0.5 
multimodal_config = MultimodalConfig(
    beta_shift=beta_shift, dropout_prob=dropout_prob
)

from torch.utils.data import DataLoader
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
BATCH_SIZE = 2 
num_epoch = 2

d = MUStartDataset('valid')
dl = DataLoader(d, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
batch_sample = iter(dl).next()

ACOUSTIC_DIM = batch_sample['audio_feature'].size(2)
VISUAL_DIM = batch_sample['video_features_p'].size(2)
TEXT_DIM = 768
TEXT_SEQ_LEN = batch_sample['bert_indices'].size(1)


train_dataset = MUStartDataset(mode='train')
valid_dataset = MUStartDataset(mode='valid')
test_dataset = MUStartDataset(mode='test')
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)
valid_dataloader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)


train_model_name = 'self_mm'

if train_model_name == 'EF_LSTM':
    train_ef_lstm_model()
elif train_model_name == 'lf_dnn':
    train_lf_dnn_model()
elif train_model_name == 'self_mm':
    train_self_mm_model()
elif train_model_name == 'mag_bert':
    train_magbert_forseqcls()
elif train_model_name == 'cmgcn':
    train_cmgcn_model()
elif train_model_name == 'cmgcni':
    train_cmgcni_model()

