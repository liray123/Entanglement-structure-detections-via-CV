# from __future__ import print_function
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
from ghz_w_jiegou_model_demo import  *
import torch
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import StratifiedKFold



class MyDataSet(Data.Dataset):
  def __init__(self, x, y):
    super(MyDataSet, self).__init__()
    self.x = x
    self.y = y
  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


particle_num = 5
data_ghz = torch.rand(100,2**particle_num,2**particle_num)
data_w = torch.rand(100,2**particle_num,2**particle_num) # generate data
label_ghz = torch.zeros(100)
label_w = torch.ones(100)
label_structure_ghz = torch.randint(0, 2**particle_num, [100])
label_structure_w= torch.randint(0, 2**particle_num, [100])
data_all = torch.cat((data_ghz,data_w),0)

### GHZ W classification
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
# label_all = torch.cat((label_ghz,label_w),0)
# epochs = 5
# batchsize = 32
# criterion = nn.CrossEntropyLoss()
# for train_ind,test_ind in skf.split(data_all,label_all):
#     data_train = data_all[train_ind,:,:].cuda()
#     data_val = data_all[test_ind, :, :].cuda()
#     label_train = label_all[train_ind].cuda()
#     label_val = label_all[test_ind].cuda()
#     model = cnn_vit_fc(particle_num,256,128,256,128,2,64).cuda()  # [8, 16, 32, 64, 64]
#     optimizer = optim.Adam(model.parameters(),
#                            lr=0.001, weight_decay=0)
#     loader1 = Data.DataLoader(MyDataSet(data_train, label_train), batch_size=batchsize, shuffle=False)
#     loader2 = Data.DataLoader(MyDataSet(data_val, label_val), batch_size=batchsize, shuffle=False)
#     for epoch in range(epochs):
#         for x1, y1 in loader1:
#             optimizer.zero_grad()
#             prd_train = model(x1)
#             loss_train = criterion(prd_train, y1.long())
#             loss_train.backward()
#             optimizer.step()
#         for x2, y2 in loader2:
#             prd_test = model(x2)


### structure classification
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
label_all = torch.cat((label_structure_ghz,label_structure_w),0)
epochs = 5
batchsize = 32
criterion = nn.CrossEntropyLoss()
for train_ind,test_ind in skf.split(data_all,label_all):
    data_train = data_all[train_ind,:,:].cuda()
    data_val = data_all[test_ind, :, :].cuda()
    label_train = label_all[train_ind].cuda()
    label_val = label_all[test_ind].cuda()
    model = cnn_vit_fc(particle_num,1024,256,256,128,2**particle_num,64).cuda()  # [8, 16, 32, 64, 64]
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001, weight_decay=0)
    loader1 = Data.DataLoader(MyDataSet(data_train, label_train), batch_size=batchsize, shuffle=False)
    loader2 = Data.DataLoader(MyDataSet(data_val, label_val), batch_size=batchsize, shuffle=False)
    for epoch in range(epochs):
        for x1, y1 in loader1:
            optimizer.zero_grad()
            prd_train = model(x1)
            loss_train = criterion(prd_train, y1.long())
            loss_train.backward()
            optimizer.step()
        for x2, y2 in loader2:
            prd_test = model(x2)
