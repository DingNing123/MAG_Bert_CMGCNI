'''
train.py
'''
import logging
from configs import *

import torch
import torch.nn as nn
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import metrics

logger = logging.getLogger(__name__)
from mustards import MUStartDataset
from torch.utils.data import DataLoader
from tqdm import tqdm 


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0') if use_cuda else torch.device('cpu')

train_dataset = MUStartDataset(mode='train')
valid_dataset = MUStartDataset(mode='valid')
test_dataset = MUStartDataset(mode='test')

# logger.error('train_dataset len : {}'.format(len(train_dataset)))

train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,num_workers=0,shuffle=False)

def debug_model(model):
    d = MUStartDataset('valid')
    dl = DataLoader(d, batch_size=2, num_workers=0, shuffle=False)
    batch = iter(dl).next()
    batch.keys()
    inputs ={}
    for key in batch.keys():
        # print(key)
        if key not in ['video_ids']:
            inputs[key] = batch[key].to(device)
    outputs = model(inputs)
    logger.error('debug_model: ' + str(type(model)) + str(outputs.shape))

def train(model, optimizer, model_save_path, times):
    '''
    训练模型
    params:
        times : 0, 1, ..., 4 
    '''
    max_val_acc , max_val_f1, max_val_epoch, global_step = 0, 0, 0, 0
    val_accs = []
    for i_epoch in tqdm(range(num_epoch)):
        logger.info('i_epoch:{}'.format(i_epoch))
        n_correct, n_total, loss_total = 0, 0, 0
        for i_batch,batch in enumerate(train_dataloader):
            global_step += 1
            model.train()
            optimizer.zero_grad()
            inputs ={}
            for key in batch.keys():
                if key not in ['video_ids']:
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

            if global_step % log_step == 0:
                val_acc, val_f1, val_precision, val_recall = evaluate_acc_f1(valid_dataloader,model)
                # 可视化
                val_accs.append(val_acc)
                # 
                logger.info('train_loss:{} val_acc:{}'.format(train_loss,val_acc))
                if val_acc >= max_val_acc:
                    max_val_f1 = val_f1
                    max_val_acc = val_acc
                    max_val_epoch = i_epoch
                    torch.save(model.state_dict(),model_save_path)
                    logger.info('save the model to {}'.format(model_save_path))

        if i_epoch - max_val_epoch >= early_stop:
            logger.error('early stop')
            break

        # break
    model.load_state_dict(torch.load(model_save_path))
    test_acc, test_f1,test_precision,test_recall = evaluate_acc_f1(test_dataloader,model)
    
    logger.error('final results.'+'-' * 20)
    logger.error('test_acc: {}'.format(test_acc))
    logger.error('test_f1: {}'.format(test_f1))
    logger.error('test_precision: {}'.format( test_precision))
    logger.error('test_recall: {}'.format( test_recall))
    logger.error('final results.'+'-' * 20)

    data = {
        train_model_name:val_accs,
    }
    
    df = pd.DataFrame(data)
    df.to_csv('6.{}.accs.{}.csv'.format(times, train_model_name),index=False)

    fig = df.plot(title = train_model_name,figsize=(8, 6), fontsize=26).get_figure()
    fig.savefig('6.{}.accs.{}.jpg'.format(times, train_model_name))
    # plt.show()
    
    return test_acc, test_f1,test_precision,test_recall 


def evaluate_acc_f1(data_loader,model):
    n_correct, n_total = 0, 0
    targets_all, outputs_all = None, None
    model.eval()
    with torch.no_grad():
        for i_batch,batch in enumerate(data_loader):
            inputs ={}
            for key in batch.keys():
                if key not in ['video_ids']:
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
    # default average = 'binary'
    f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all,-1).cpu(), labels=[0,1], average='macro', zero_division=0)
    precision = metrics.precision_score(targets_all.cpu(), torch.argmax(outputs_all,-1).cpu(), labels=[0,1], average='macro', zero_division=0)
    recall = metrics.recall_score(targets_all.cpu(), torch.argmax(outputs_all,-1).cpu(), labels=[0,1], average='macro', zero_division=0)

    return acc,f1,precision,recall