'''
configs
'''

log_file = '13.log'
json_file = 'sarcasm_data.json'
pretrained_root_path = '../tools/'
big_file_path = '../1124.mustard_bigfile/'  
feature_file = big_file_path + 'featuresIndepVit768.pkl'
audio_path = big_file_path + 'mmsd_raw_data/Processed/audio/'
frame_path = big_file_path + 'mmsd_raw_data/Processed/video/Frames/'
feature_sample_saved_path = big_file_path + './feature_sample/'
coordinate_path = big_file_path +  'mmsd_raw_data/Processed/video/box_coordinate/'
results_path = 'results/'
beta_shift = 1.0 
dropout_prob = 0.5 
BATCH_SIZE = 128
num_epoch = 100
early_stop = 20
log_step = 5
TEXT_DIM = 768
flag_train = False
train_model_name = 'cmgcni'




