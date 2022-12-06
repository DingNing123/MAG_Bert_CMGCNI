# read ./featuresIndepResnet152.pkl

import pickle
name = './featuresIndepResnet152.pkl'
f = open(name,'rb')
data = pickle.load(f)
print(type(data),data.keys())
train = data['train']
print(type(train),train.keys())
'''
<class 'dict'> dict_keys(['train', 'valid', 'test'])
<class 'dict'> dict_keys(['id', 'raw_text', 'audio', 'vision', 'text', 'text_bert', 'audio_lengths', 'vision_lengths', 'annotations', 'classification_labels', 'regression_labels'])
'''
print(train['audio'].shape)
print(train['vision'].shape)
print(train['text'].shape)
print(train['text_bert'].shape)
print(train['regression_labels'])

