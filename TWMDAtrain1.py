#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:11:13 2019

@author: user9
"""

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.chdir('/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1')
root_dir='/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1'

from logs1 import *
from MME_fun1 import *
import sys
sys.stdout = Logger(os.path.join(root_dir, 'Logs_CPQRIs1.txt'))
import torch
import torch.utils.data
from torch import nn
from torch import optim

from torchvision import datasets, models, transforms
from torchvision import transforms as tfs
from torch.utils.data import Dataset, DataLoader


from torchvision.datasets import ImageFolder
SI='/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1/infograph/'
SP='/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1/painting/'
SQ='/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1/quickdraw/'
SR='/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1/real/'
SS='/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1/sketch/'
SC='/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1/clipart/'


dataset_s1=datasets.ImageFolder(SC,transformtr2)
dataset_s2=datasets.ImageFolder(SP,transformtr2)
dataset_s3=datasets.ImageFolder(SQ,transformtr2)
dataset_s4=datasets.ImageFolder(SR,transformtr2)
dataset_s5=datasets.ImageFolder(SI,transformtr2)
dataset_T =datasets.ImageFolder(SS,transformtr2)

mo_dir='MMEstepC_CPQRIs_'

###################################################Data Processing
train_loader1  = DataLoader(dataset=dataset_s1, num_workers=8, batch_size=8, shuffle=True,drop_last=True)
train_loader2  = DataLoader(dataset=dataset_s2, num_workers=8, batch_size=8, shuffle=True,drop_last=True)
train_loader3  = DataLoader(dataset=dataset_s3, num_workers=8, batch_size=8, shuffle=True,drop_last=True)
train_loader4  = DataLoader(dataset=dataset_s4, num_workers=8, batch_size=8, shuffle=True,drop_last=True)
train_loader5  = DataLoader(dataset=dataset_s5, num_workers=8, batch_size=8, shuffle=True,drop_last=True)
val_loader1     = DataLoader(dataset=dataset_T , num_workers=8, batch_size=8, shuffle=True,drop_last=True)
val_loader2     = DataLoader(dataset=dataset_T , num_workers=8, batch_size=8, shuffle=True,drop_last=True)

criterion = nn.CrossEntropyLoss()
net1=torch.load('/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1/CPQRIs1_/CPQRIsPre1_2_Resnet1_030.pth')
net2=torch.load('/data2/yonghui_/Multi_Sous_/MinMaxEn1/DomainNet1/CPQRIs1_/CPQRIsPre1_2_Resnet2_030.pth')



#net1 = torch.nn.DataParallel(net1, device_ids=[0,1]).cuda()
#net2 = torch.nn.DataParallel(net2, device_ids=[0,1]).cuda()
optimizer=optim.SGD([{'params':net1.parameters()},{'params':net2.parameters()}],
                      lr=0.0001,momentum=0.9)
from torch.optim import lr_scheduler
lr_decay1=lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2
                                         , patience=2, verbose=True
                                         , threshold=0.000001,threshold_mode='rel'
                                         , cooldown=1, min_lr=0.0000001, eps=0.0001)
trainOneStep1(net1,net2,train_loader1,train_loader2,train_loader3,train_loader4,train_loader5, val_loader1,val_loader2, 395, optimizer, criterion,lr_decay1,mo_dir)
