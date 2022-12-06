import pickle
from configs import *
def save_sample_feature(audio_feature, video_feature, graph,cross_graph, vision_full_feat, cls_names_10, save_path):
    data  = audio_feature,video_feature,graph,cross_graph,vision_full_feat,cls_names_10
    with open(save_path,'wb') as f:
        pickle.dump(data,f)
        
def load_sample_feature(save_path):
    with open(save_path,'rb') as f:
        data = pickle.load(f) 
    return data


def save_box_coordinate(video_id, boxs):
    print('save box coordinate ',video_id)
    coord_file = coordinate_path + video_id + '.pkl'
    with open(coord_file, 'wb') as f:
        pickle.dump(boxs, f)
        
def load_box_coordinate(video_id):
    print('load box coordinate', video_id)
    coord_file = coordinate_path + video_id + '.pkl'
    with open(coord_file, 'rb') as f:
        boxs = pickle.load(f)
    return boxs