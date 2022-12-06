# 读取split_indices.p 文件 这是属于https://github.com/soujanyaporia/MUStARD/blob/master/data/split_indices.p

name = 'split_indices.p'
f = open(name,'rb')
import pickle 
data = pickle.load(f, encoding="latin1")
print(data)
