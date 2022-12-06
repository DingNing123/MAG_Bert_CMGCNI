# 生成平衡的划分 train_index5.csv  test_index5.csv indepedent.csv

name = 'sarcasm_data.json'
import json
data = json.load(open(name))
print(len(data),type(data))

test_ids = []
train_ids = []
for id, ID in enumerate(list(data.keys())[:]):
# for id, ID in enumerate(list(data.keys())[:10]):
    # print(id, ID)
    # print(data[ID].keys())
    speaker = data[ID]['speaker']
    if speaker == 'HOWARD' or speaker == 'SHELDON' :
        # print(speaker, id )
        test_ids.append(id)
    else :
        train_ids.append(id)

print(len(test_ids))
print(len(train_ids))
        
import pandas as pd
train_file = 'train_index5.csv'
test_file = 'test_index5.csv'
df = pd.DataFrame({'id':train_ids})
df.to_csv(train_file,index=False)

df = pd.DataFrame({'id':test_ids})
df.to_csv(test_file,index=False)


import numpy as np
data = pd.read_csv(train_file)
print(np.array(data).shape)
print(np.array(data).reshape(-1).shape)
data = pd.read_csv(test_file)
print(np.array(data).shape)
