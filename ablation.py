'''
ablation.py
'''
import logging
import pandas as pd

import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from cmgcni import train_cmgcni_model
from configs import *
from loggers import get_stderr_file_logger

logger = logging.getLogger(__name__)

def ablation_graph_convolution_layers(train_one_times):
    '''
    ablation 1-4 convolution layers
    '''
    p, r, f, accs = [], [], [], []
    for i in range(1):
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
    train_model_name = 'cmgcni'
    file_name = '6.wo_mag_{}.csv'.format(train_model_name)
    df2.to_csv(file_name,index=True)
    logger.error('save file to ' + file_name)
    

def plot_ablation_num_of_layers(ax):
    ablation_files = ['6.layer_{}_cmgcni.csv'.format(i) for i in range(5)]
    accs = []
    fs = []
    for a_file in ablation_files:
        # print(a_file)
        df = pd.read_csv(a_file,index_col = 0)
        acc = df['acc'][0]
        f = df['f'][0]
        accs.append(acc)
        fs.append(f)

    data = {'acc' : accs,
           'f1':fs,}
    df = pd.DataFrame(data)
    file_name = '6.layers_0_4'
    df.to_csv(file_name+'.csv')
    # print(df)
    # df.plot(title = 'number of layers',figsize=(5, 4), fontsize=12,xticks=[0, 1, 2, 3, 4], ylim = [0.5, 0.7], )
    ax = df.plot.line(ax = ax, title = '(b) number of layers', xticks=[0, 1, 2, 3, 4])
    # png_file_path = file_name + '.png'
    # plt.savefig(png_file_path)
    # plt.show()
    # print('save file {}'.format(png_file_path))  
    return ax
    
def plot_ablation_wo(ax ):
    names = ['mag', 'graph', 'none']
    ablation_files = ['6.wo_{}_cmgcni.csv'.format(i) for i in names]
    accs = []
    fs = []
    for a_file in ablation_files:
        # print(a_file)
        df = pd.read_csv(a_file,index_col = 0)
        acc = df['acc'][0]
        f = df['f'][0]
        accs.append(acc)
        fs.append(f)

    data = {'acc' : accs,
           'f1':fs,}
    
    index = ['w/o mag','w/o graph','cmgcni']
    df = pd.DataFrame(data, index = index)
    df = df.transpose()
    file_name = '6.wo'
    df.to_csv(file_name+'.csv')
    df.plot.bar(ax = ax,rot=0,title='(a) w/o ablation').legend(loc='lower center')
    # plt.legend(loc='lower left')

    
def plot_wo_layers():
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    plot_ablation_wo(ax1)
    plot_ablation_num_of_layers(ax2)
    file_name = '6.wo.layer'
    png_file_path = file_name + '.png'
    
    plt.savefig(png_file_path)
    plt.show()
    print('save file {}'.format(png_file_path)) 
    
if __name__=="__main__": 
    logger = get_stderr_file_logger(log_file)
    logging.info("This is an INFO message")
    logging.warning("This is a WARNING message")
    # ablation_graph_convolution_layers(train_cmgcni_model)
    # plot_ablation_num_of_layers()
    # plot_ablation_wo()
    plot_wo_layers()

    
    