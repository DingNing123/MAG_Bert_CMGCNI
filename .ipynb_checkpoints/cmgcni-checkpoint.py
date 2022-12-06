'''
cmgcni.py
4 层图卷积 记录实验结果 
'''
import logging
logger = logging.getLogger(__name__)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers import BertTokenizer
from mustards import MUStartDataset
from torch.utils.data import DataLoader
from train import debug_model
from layers.dynamic_rnn import DynamicLSTM
from configs import *
from train import train


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob

multimodal_config = MultimodalConfig(
    beta_shift=beta_shift, dropout_prob=dropout_prob
)

d = MUStartDataset('valid')
dl = DataLoader(d, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
batch_sample = iter(dl).next()

ACOUSTIC_DIM = batch_sample['audio_feature'].size(2)
VISUAL_DIM = batch_sample['video_features_p'].size(2)
TEXT_SEQ_LEN = batch_sample['bert_indices'].size(1)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
logger.info('use device: {}'.format(device))

def init_cmgcni_params(cmgcni_model):
    for child in cmgcni_model.children():
        #  BertModel|MAG_BertModel
        if type(child) not in [BertModel, MAG_BertModel] :
            for p in child.parameters():
                if p.requires_grad :
                    if len(p.shape) > 1:
                        torch.nn.init.xavier_uniform_(p)
                    else:
                        
                        stdv = 1.0 / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    logger.error('init_cmgcni_params()')

# prepare model 
def train_cmgcni_model(times=1):
    cmgcni_model = CMGCNI(multimodal_config = multimodal_config ).to(device)
    init_cmgcni_params(cmgcni_model) 
    cmgcni_optimizer = get_cmgcni_optimizer(cmgcni_model)
    debug_model(cmgcni_model)
    cmgcni_model_path = '../cmgcni_model.pth'

    model = cmgcni_model
    optimizer = cmgcni_optimizer
    model_save_path = cmgcni_model_path
    logger.error('start train:' + '-'*10)
    test_acc, test_f1,test_precision,test_recall  = train(model,optimizer,model_save_path,times)
    return test_acc, test_f1,test_precision,test_recall 


def get_cmgcni_optimizer(cmgcni_model):
    optimizer = torch.optim.Adam([
        {'params':cmgcni_model.mag_bert.parameters(),'lr':2e-5},
        # {'params':cmgcni_model.bert.parameters(),'lr':2e-5},
        {'params':cmgcni_model.text_lstm.parameters(),},
        {'params':cmgcni_model.vit_fc.parameters(),},
        {'params':cmgcni_model.gc1.parameters(),},
        {'params':cmgcni_model.gc2.parameters(),},
        {'params':cmgcni_model.gc3.parameters(),},
        {'params':cmgcni_model.gc4.parameters(),},
        {'params':cmgcni_model.fc.parameters(),},
    ],lr=0.001,weight_decay=1e-5)
    return optimizer

class CMGCNI(nn.Module):
    def __init__(self, multimodal_config):
        super(CMGCNI, self).__init__()
        print('create CMGCNI model')
        self.mag_bert = MAG_BertModel.from_pretrained(pretrained_root_path + 'bert-base-uncased/',multimodal_config=multimodal_config)
        # self.bert = BertModel.from_pretrained(pretrained_root_path + 'bert-base-uncased/')
        self.text_lstm = DynamicLSTM(768,4,num_layers=1,batch_first=True,bidirectional=True)
        self.vit_fc = nn.Linear(768,2*4)
        self.gc1 = GraphConvolution(2*4, 2*4)
        self.gc2 = GraphConvolution(2*4, 2*4)
        self.gc3 = GraphConvolution(2*4, 2*4)
        self.gc4 = GraphConvolution(2*4, 2*4)
        
        self.fc = nn.Linear(2*4,2)

    def forward(self, inputs, outAttention = False):
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
        
        # bert without mag
        # outputs = self.bert(bert_indices)
        # encoder_layer = outputs.last_hidden_state
        # pooled_output = outputs.pooler_output
        
        # mag_bert
        outputs = self.mag_bert(bert_indices, visual, acoustic)
        encoder_layer = outputs[0]
        pooled_output = outputs[1]
        
        bert_text_len = bert_text_len.cpu()
        
        text_out, (_, _) = self.text_lstm(encoder_layer, bert_text_len)
        # 与原始代码不同，这里因为进行了全局的特征填充，导致text_out可能无法达到填充长度，补充为0
        if text_out.shape[1] < encoder_layer.shape[1]:
            pad = torch.zeros((text_out.shape[0],encoder_layer.shape[1]-text_out.shape[1],text_out.shape[2])).to(device)
            text_out = torch.cat((text_out,pad),dim=1)

        box_vit = box_vit.float()
        box_vit = self.vit_fc(box_vit)
        features = torch.cat([text_out, box_vit], dim=1)

        graph = graph.float()
        x = features
        x = F.relu(self.gc1(x,graph))
        x = F.relu(self.gc2(x,graph))
        # x = F.relu(self.gc3(x,graph))
        # x = F.relu(self.gc4(x,graph))
        
        alpha_mat = torch.matmul(features,x.transpose(1,2))
        alpha_mat = alpha_mat.sum(1, keepdim=True)
        alpha = F.softmax(alpha_mat, dim = 2)
        x = torch.matmul(alpha, x).squeeze(1)
        
        output = self.fc(x)
        if outAttention:
            output = (output, alpha)
        return output
    
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