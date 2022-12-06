'''
visions.py
'''

import numpy as np
import librosa
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

from configs import *


def audio_visual_by_id(video_id):
    
    # ../1124.mustard_bigfile/mmsd_raw_data/Processed/audio/1_60/tmp.wav
    audio_file_path = '{}{}/tmp.wav'.format(audio_path, video_id)
    y,sr = librosa.load(audio_file_path)
    plt.plot(y)
    # results/1_60_audio.png
    fp = '{}{}_audio.png'.format(results_path, video_id)
    plt.savefig(fp)
    print('saved in {}'.format(fp))
    
    
if __name__=="__main__": 
    video_id='1_80'
    audio_visual_by_id(video_id)
    
    