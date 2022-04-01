#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:08:32 2019

@author: user9
"""

import torch.nn.functional as F
from datetime import datetime

import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function
import os
from PIL import Image
from torchvision import transforms as tfs
import time
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

import torchvision.transforms.functional as TransF
import random

class dataset_txt(Dataset):  #
    def __init__(self,name_file,image_root_dir,transformim=None):
        self.index1=pd.read_csv(name_file,sep=' ',header=None)
        self.root_dir=image_root_dir
        self.transformim=transformim
    def __len__(self):
        return len(self.index1)
    def __getitem__(self,idx):
        img_name1=os.path.join(self.root_dir,self.index1.iloc[idx,0])  #Image
        label1=np.array(self.index1.iloc[idx,1],dtype=np.int64)  
        image=Image.open(img_name1);
        label1=np.ravel(label1)
        if self.transformim:
            image=self.transformim(image)
        return image,label1

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TransF.rotate(x, angle)
rotation_transform = MyRotationTransform(angles=[-115,-25, 0, 25, 115])

transformte2=tfs.Compose([
        tfs.Resize((224,224)),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformtr2=tfs.Compose([
        tfs.Resize((224,224)),
        rotation_transform,
        tfs.RandomHorizontalFlip(),
        tfs.RandomVerticalFlip(),
        tfs.ColorJitter(brightness=32. / 255.,saturation=0.5, contrast = 0.5, hue = 0.2),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
transform3=tfs.Compose([
        tfs.CenterCrop((224,224)),
        tfs.ToTensor(),
        tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
cr0=tfs.CenterCrop(896)
cr1=tfs.CenterCrop(1344)
cr21=tfs.CenterCrop(1792)
cr2=tfs.CenterCrop(2688)
cr3=tfs.CenterCrop(3360)
cr4=tfs.CenterCrop(4256)
resz0=tfs.RandomResizedCrop(224)
resz1=tfs.Resize(896)
resz2=tfs.Resize(672)
resz3=tfs.Resize(448)
resz4=tfs.Resize(224)
class dataset_Sca2(Dataset):
    def __init__(self,name_file,image_root_dir,transformim=None):
        self.index1=pd.read_csv(name_file,sep=',',header=0)
        self.root_dir=image_root_dir
        self.transformim=transformim
    def __len__(self):
        return len(self.index1)
    def __getitem__(self,idx):
        img_name1=os.path.join(self.root_dir,self.index1.iloc[idx,0]+'.jpg')  #Image
        img_name2=self.index1.iloc[idx,0]
        label1=self.index1.iloc[idx,1]
        label1=np.array(label1,dtype=np.int64)  #
        image=Image.open(img_name1)
        lx,ly=image.size
        if ly>=4000:
            image=cr4(image)
        elif ly >=3500 :
            image=cr3(image)
        elif ly >=2700:
            image=cr2(image)
        elif ly >=2000:
            image=cr21(image)
        elif ly>=1400:
            image=cr1(image)
        else:
            image=cr0(image)
        label1=np.ravel(label1)
        
        res1=resz1(image)
        if self.transformim:
            res1=self.transformim(res1)
        return res1,label1,img_name2
import torch.nn.init as init
def weights_init1(m):
    if type(m) in [nn.ConvTranspose2d,nn.Conv2d]:
        init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        init.normal_(m.weight,1.0,0.02)
        init.constant_(m.bias,0)
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
class GradReverse(Function):
    def __init__(self, lambd=1):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)
def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)
class Predictor1(nn.Module):
    def __init__(self, num_class):
        super(Predictor1, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1_fc = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.bn2_fc = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_class)
        self.bn_fc3 = nn.BatchNorm1d(num_class)

    def set_lambda(self, lambd=1):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x)
        x=F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x

softmax1=nn.Softmax(dim=1)
from sklearn.metrics import roc_auc_score , average_precision_score,recall_score,confusion_matrix

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2,dim=1)))


def Pretrain1(net1,train_data1,train_data2,train_data3, valid_data, num_epochs, optimizer, criterion,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        net1 = net1.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net1.train()
        train_auc = 0
        val_label=[]
        val_plabel=[]
        name_df1=pd.DataFrame()
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        i = 1        
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3))
        while i < len_loader_S+1:
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3

            im1 = Variable(im1.cuda())  # (bs, 3, h, w)
            label1 = Variable(label1.cuda())  # (bs, h, w)
            im2 = Variable(im2.cuda())  # (bs, 3, h, w)
            label2 = Variable(label2.cuda())  # (bs, h, w)
            im3 = Variable(im3.cuda())  # (bs, 3, h, w)
            label3 = Variable(label3.cuda())  # (bs, h, w)
            # forward
            output1,_= net1(im1)
            label1=label1.squeeze()
            output2,_= net1(im2)
            label2=label2.squeeze()
            output3,_= net1(im3)
            label3=label3.squeeze()
            loss = criterion(output1, label1)+criterion(output2, label2)+criterion(output3, label3)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (get_acc(output1, label1)+get_acc(output2, label2)+get_acc(output3, label3))
            i +=1

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            
            val_label=[]
            val_plabel=[]
            val_plabel2=[]
            net1.eval()
            for im,label in valid_data:
                label=label.squeeze()
                label2=label.numpy()
                val_label.append(label2)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                output,_= net1(im)
                
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
                
                output=F.softmax(output,dim=1)              

                
                _, pred_label = output.max(1)
                pred_prob2=pred_label.cpu()
                pred_prob2=pred_prob2.detach().numpy()
                val_plabel2.append(pred_prob2)#Pred labels,

        val_plabel2=np.array(val_plabel2)
        lenT=val_plabel2.shape[0]*val_plabel2.shape[1]
        val_label=np.array(val_label)
        val_label=val_label.reshape(lenT,1)
        val_plabel2=val_plabel2.reshape(lenT,1)
        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2

        if valid_acc / len(valid_data) > best_val:                
            best_val = valid_acc / len(valid_data)
            torch.save(net1,os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(epoch)))
            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)
            if best_idx>0:
                os.remove(os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(best_idx))) #remove Old models
                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
            best_idx=epoch
        epoch_str = (
                "Epoch %d. TrLoss: %f, VaLoss: %f, Train Acc: %f, Valid Acc: %f"
                % (epoch, train_loss / (3.0*len_loader_S),valid_loss / len(valid_data),train_acc/(3.0*len_loader_S),
                    valid_acc / len(valid_data)))
        lr_decay1.step(valid_acc / len(valid_data))
        #lr_decay1.step() 
        prev_time = cur_time
        print(epoch_str + time_str)
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def Pretrain2(net1,train_data1,train_data2,train_data3, valid_data, num_epochs, optimizer, criterion,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        net1 = net1.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net1.train()
        train_auc = 0
        val_label=[]
        val_plabel=[]
        name_df1=pd.DataFrame()
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        i = 1        
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3))
        while i < len_loader_S+1:
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3

            im1 = Variable(im1.cuda())  # (bs, 3, h, w)
            label1 = Variable(label1.cuda())  # (bs, h, w)
            im2 = Variable(im2.cuda())  # (bs, 3, h, w)
            label2 = Variable(label2.cuda())  # (bs, h, w)
            im3 = Variable(im3.cuda())  # (bs, 3, h, w)
            label3 = Variable(label3.cuda())  # (bs, h, w)
            # forward
            _,output1,_= net1(im1)
            label1=label1.squeeze()
            _,output2,_= net1(im2)
            label2=label2.squeeze()
            _,output3,_= net1(im3)
            label3=label3.squeeze()
            loss = criterion(output1, label1)+criterion(output2, label2)+criterion(output3, label3)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (get_acc(output1, label1)+get_acc(output2, label2)+get_acc(output3, label3))
            i +=1

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            
            val_label=[]
            val_plabel=[]
            val_plabel2=[]
            net1.eval()
            for im,label in valid_data:
                label=label.squeeze()
                label2=label.numpy()
                val_label.append(label2)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                _,output,_= net1(im)
                
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
                
                output=F.softmax(output,dim=1)              

                
                _, pred_label = output.max(1)
                pred_prob2=pred_label.cpu()
                pred_prob2=pred_prob2.detach().numpy()
                val_plabel2.append(pred_prob2)#Pred labels,

        val_plabel2=np.array(val_plabel2)
        lenT=val_plabel2.shape[0]*val_plabel2.shape[1]
        val_label=np.array(val_label)
        val_label=val_label.reshape(lenT,1)
        val_plabel2=val_plabel2.reshape(lenT,1)
        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2

        if valid_acc / len(valid_data) > best_val:                
            best_val = valid_acc / len(valid_data)
            torch.save(net1,os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(epoch)))
            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)
            if best_idx>0:
                os.remove(os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(best_idx))) #remove Old models
                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
            best_idx=epoch
        epoch_str = (
                "Epoch %d. TrLoss: %f, VaLoss: %f, Train Acc: %f, Valid Acc: %f"
                % (epoch, train_loss / (3.0*len_loader_S),valid_loss / len(valid_data),train_acc/(3.0*len_loader_S),
                    valid_acc / len(valid_data)))
        lr_decay1.step(valid_acc / len(valid_data))
        #lr_decay1.step() 
        prev_time = cur_time
        print(epoch_str + time_str)
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def Pretrain_att1(net1,train_data1,train_data2,train_data3, valid_data, num_epochs, optimizer, criterion,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        net1 = net1.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net1.train()
        train_auc = 0
        val_label=[]
        val_plabel=[]
        name_df1=pd.DataFrame()
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        i = 1        
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3))
        while i < len_loader_S+1:
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3

            im1 = Variable(im1.cuda())  # (bs, 3, h, w)
            label1 = Variable(label1.cuda())  # (bs, h, w)
            im2 = Variable(im2.cuda())  # (bs, 3, h, w)
            label2 = Variable(label2.cuda())  # (bs, h, w)
            im3 = Variable(im3.cuda())  # (bs, 3, h, w)
            label3 = Variable(label3.cuda())  # (bs, h, w)
            # forward
            output1,_,_,_,_= net1(im1)
            label1=label1.squeeze()
            output2,_,_,_,_= net1(im2)
            label2=label2.squeeze()
            output3,_,_,_,_= net1(im3)
            label3=label3.squeeze()
            loss = criterion(output1, label1)+criterion(output2, label2)+criterion(output3, label3)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (get_acc(output1, label1)+get_acc(output2, label2)+get_acc(output3, label3))
            i +=1

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            
            val_label=[]
            val_plabel=[]
            val_plabel2=[]
            net1.eval()
            for im,label in valid_data:
                label=label.squeeze()
                label2=label.numpy()
                val_label.append(label2)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                output,_,_,_,_= net1(im)
                
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
                
                output=F.softmax(output,dim=1)              

                
                _, pred_label = output.max(1)
                pred_prob2=pred_label.cpu()
                pred_prob2=pred_prob2.detach().numpy()
                val_plabel2.append(pred_prob2)#Pred labels,

        val_plabel2=np.array(val_plabel2)
        lenT=val_plabel2.shape[0]*val_plabel2.shape[1]
        val_label=np.array(val_label)
        val_label=val_label.reshape(lenT,1)
        val_plabel2=val_plabel2.reshape(lenT,1)
        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2

        if valid_acc / len(valid_data) > best_val:                
            best_val = valid_acc / len(valid_data)
            torch.save(net1,os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(epoch)))
            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)
            if best_idx>0:
                os.remove(os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(best_idx))) #remove Old models
                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
            best_idx=epoch
        epoch_str = (
                "Epoch %d. TrLoss: %f, VaLoss: %f, Train Acc: %f, Valid Acc: %f"
                % (epoch, train_loss / (3.0*len_loader_S),valid_loss / len(valid_data),train_acc/(3.0*len_loader_S),
                    valid_acc / len(valid_data)))
        lr_decay1.step(valid_acc / len(valid_data))
        #lr_decay1.step() 
        prev_time = cur_time
        print(epoch_str + time_str)
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def Pretrain_DomainNet1(net1,net2,train_data1,train_data2,train_data3,train_data4,train_data5,valid_data, num_epochs, optimizer, criterion,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        net1 = net1.cuda()
        net2 = net2.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net1.train()
        train_auc = 0
        val_label=[]
        val_plabel=[]
        name_df1=pd.DataFrame()
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)

        i = 1        
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5))
        while i < len_loader_S+1:
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()


            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5
            
            im_sou=torch.cat((im1,im2,im3,im4,im5),0)
            lb_sou=torch.cat((label1,label2,label3,label4,label5),0)

            im_sou = Variable(im_sou.cuda())  # (bs, 3, h, w)
            lb_sou = Variable(lb_sou.cuda())  # (bs, h, w)


            # forward
            _,output1,_,_,_,_= net1(im_sou)
            output1 = net2(output1.squeeze())
            lb_sou=lb_sou.squeeze()

            loss = criterion(output1, lb_sou)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += get_acc(output1, lb_sou)
            i +=1

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            
            val_label=[]
            val_plabel=[]
            val_plabel2=[]
            net1.eval()
            for im,label in valid_data:
                label=label.squeeze()
                label2=label.numpy()
                val_label.append(label2)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                _,output,_,_,_,_= net1(im)
                output=net2(output.squeeze())
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
                
                output=F.softmax(output,dim=1)              

                
                _, pred_label = output.max(1)
                pred_prob2=pred_label.cpu()
                pred_prob2=pred_prob2.detach().numpy()
                val_plabel2.append(pred_prob2)#Pred labels,

        val_plabel2=np.array(val_plabel2)
        lenT=val_plabel2.shape[0]*val_plabel2.shape[1]
        val_label=np.array(val_label)
        val_label=val_label.reshape(lenT,1)
        val_plabel2=val_plabel2.reshape(lenT,1)
        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2

        if valid_acc / len(valid_data) > best_val:                
            best_val = valid_acc / len(valid_data)
            torch.save(net1,os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(epoch)))
            torch.save(net2,os.path.join(mo_dir,'Resnet2_{:03d}.pth'.format(epoch)))

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)
            if best_idx>0:
                os.remove(os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(best_idx))) #remove Old models
                os.remove(os.path.join(mo_dir,'Resnet2_{:03d}.pth'.format(best_idx))) #remove Old models
                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
            best_idx=epoch
        epoch_str = (
                "Epoch %d. TrLoss: %f, VaLoss: %f, Train Acc: %f, Valid Acc: %f"
                % (epoch, train_loss / (2.0*len_loader_S),valid_loss / len(valid_data),train_acc/(2.0*len_loader_S),
                    valid_acc / len(valid_data)))
        lr_decay1.step(valid_acc / len(valid_data))
        #lr_decay1.step() 
        prev_time = cur_time
        print(epoch_str + time_str)
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def Pretrain_DomainNet2(net1,net2,train_data1,train_data2,train_data3,train_data4,train_data5,valid_data, num_epochs, optimizer, criterion,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        net1 = net1.cuda()
        net2 = net2.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net1.train()
        train_auc = 0
        val_label=[]
        val_plabel=[]
        name_df1=pd.DataFrame()
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)

        i = 1        
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5))
        while i < len_loader_S+1:
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()


            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5


            output1,_= net1(im1)
            label1=label1.squeeze()
            output2,_= net1(im2)
            label2=label2.squeeze()
            output3,_= net1(im3)
            label3=label3.squeeze()
            output4,_= net1(im4)
            label4=label4.squeeze()
            output5,_= net1(im5)
            label5=label5.squeeze()

            
            im_sou=torch.cat((im1,im2,im3,im4,im5),0)
            lb_sou=torch.cat((label1,label2,label3,label4,label5),0)

            im_sou = Variable(im_sou.cuda())  # (bs, 3, h, w)
            lb_sou = Variable(lb_sou.cuda())  # (bs, h, w)


            # forward
            _,output1,_,_,_,_= net1(im_sou)
            output1 = net2(output1.squeeze())
            lb_sou=lb_sou.squeeze()

            loss = criterion(output1, lb_sou)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += get_acc(output1, lb_sou)
            i +=1

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            
            val_label=[]
            val_plabel=[]
            val_plabel2=[]
            net1.eval()
            for im,label in valid_data:
                label=label.squeeze()
                label2=label.numpy()
                val_label.append(label2)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                _,output,_,_,_,_= net1(im)
                output=net2(output.squeeze())
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
                
                output=F.softmax(output,dim=1)              

                
                _, pred_label = output.max(1)
                pred_prob2=pred_label.cpu()
                pred_prob2=pred_prob2.detach().numpy()
                val_plabel2.append(pred_prob2)#Pred labels,

        val_plabel2=np.array(val_plabel2)
        lenT=val_plabel2.shape[0]*val_plabel2.shape[1]
        val_label=np.array(val_label)
        val_label=val_label.reshape(lenT,1)
        val_plabel2=val_plabel2.reshape(lenT,1)
        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2

        if valid_acc / len(valid_data) > best_val:                
            best_val = valid_acc / len(valid_data)
            torch.save(net1,os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(epoch)))
            torch.save(net2,os.path.join(mo_dir,'Resnet2_{:03d}.pth'.format(epoch)))

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)
            if best_idx>0:
                os.remove(os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(best_idx))) #remove Old models
                os.remove(os.path.join(mo_dir,'Resnet2_{:03d}.pth'.format(best_idx))) #remove Old models
                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
            best_idx=epoch
        epoch_str = (
                "Epoch %d. TrLoss: %f, VaLoss: %f, Train Acc: %f, Valid Acc: %f"
                % (epoch, train_loss / (2.0*len_loader_S),valid_loss / len(valid_data),train_acc/(2.0*len_loader_S),
                    valid_acc / len(valid_data)))
        lr_decay1.step(valid_acc / len(valid_data))
        #lr_decay1.step() 
        prev_time = cur_time
        print(epoch_str + time_str)
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def Pretrain_DomainNet3(net1,net2,train_data1,train_data2,train_data3,train_data4,train_data5,valid_data, num_epochs, optimizer, criterion,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        net1 = net1.cuda()
        net2 = net2.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net1.train()
        train_auc = 0
        val_label=[]
        val_plabel=[]
        name_df1=pd.DataFrame()
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)

        i = 1        
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5))
        while i < len_loader_S+1:
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()


            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5
            
            im_sou=torch.cat((im1,im2,im3,im4,im5),0)
            lb_sou=torch.cat((label1,label2,label3,label4,label5),0)

            im_sou = Variable(im_sou.cuda())  # (bs, 3, h, w)
            lb_sou = Variable(lb_sou.cuda())  # (bs, h, w)


            # forward
            _,output1,_,_,_,_= net1(im_sou)
            output1 = net2(output1.squeeze())
            lb_sou=lb_sou.squeeze()

            loss = criterion(output1, lb_sou)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += get_acc(output1, lb_sou)
            i +=1

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            
            val_label=[]
            val_plabel=[]
            val_plabel2=[]
            net1.eval()
            for im,label in valid_data:
                label=label.squeeze()
                label2=label.numpy()
                val_label.append(label2)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                _,output,_,_,_,_= net1(im)
                output=net2(output.squeeze())
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
                
                output=F.softmax(output,dim=1)              

                
                _, pred_label = output.max(1)
                pred_prob2=pred_label.cpu()
                pred_prob2=pred_prob2.detach().numpy()
                val_plabel2.append(pred_prob2)#Pred labels,

        val_plabel2=np.array(val_plabel2)
        lenT=val_plabel2.shape[0]*val_plabel2.shape[1]
        val_label=np.array(val_label)
        val_label=val_label.reshape(lenT,1)
        val_plabel2=val_plabel2.reshape(lenT,1)
        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2

        if valid_acc / len(valid_data) > best_val:                
            best_val = valid_acc / len(valid_data)
            torch.save(net1,os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(epoch)))
            torch.save(net2,os.path.join(mo_dir,'Resnet2_{:03d}.pth'.format(epoch)))

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)
            if best_idx>0:
                os.remove(os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(best_idx))) #remove Old models
                os.remove(os.path.join(mo_dir,'Resnet2_{:03d}.pth'.format(best_idx))) #remove Old models
                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
            best_idx=epoch
        epoch_str = (
                "Epoch %d. TrLoss: %f, VaLoss: %f, Train Acc: %f, Valid Acc: %f"
                % (epoch, train_loss / (5.0*len_loader_S),valid_loss / len(valid_data),train_acc/(5.0*len_loader_S),
                    valid_acc / len(valid_data)))
        #lr_decay1.step(valid_acc / len(valid_data))
        #lr_decay1.step() 
        prev_time = cur_time
        print(epoch_str + time_str)
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def Pretrain_DomainNet4(net1,net2,train_data1,train_data2,train_data3,train_data4,train_data5,valid_data, num_epochs, optimizer, criterion,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        net1 = net1.cuda()
        net2 = net2.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net1.train()
        train_auc = 0
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)

        i = 1        
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5))
        while i < len_loader_S+1:
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()


            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5
            
            im1 = Variable(im1.cuda())
            label1 = Variable(label1.cuda())
            im2 = Variable(im2.cuda())
            label2 = Variable(label2.cuda())
            im3 = Variable(im3.cuda())
            label3 = Variable(label3.cuda())
            im4 = Variable(im4.cuda())
            label4 = Variable(label4.cuda())
            im5 = Variable(im5.cuda())
            label5 = Variable(label5.cuda())

            # forward
            output1= net1(im1)
            output1 = net2(output1.squeeze())
            label1=label1.squeeze()
            loss1 = criterion(output1, label1)

            output2= net1(im2)
            output2 = net2(output2.squeeze())
            label2=label2.squeeze()
            loss2 = criterion(output2, label2)

            output3= net1(im3)
            output3 = net2(output3.squeeze())
            label3=label3.squeeze()
            loss3 = criterion(output3, label3)

            output4= net1(im4)
            output4 = net2(output4.squeeze())
            label4=label4.squeeze()
            loss4 = criterion(output4, label4)

            output5= net1(im5)
            output5 = net2(output5.squeeze())
            label5=label5.squeeze()
            loss5 = criterion(output5, label5)
            loss=loss1+loss2+loss3+loss4+loss5

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc  += (get_acc(output1, label1)+get_acc(output2, label2)+get_acc(output3, label3)+get_acc(output4, label4)+get_acc(output5, label5))
            i +=1

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0

            net1.eval()
            net2.eval()
            for im,label in valid_data:
                label=label.squeeze()
                if torch.cuda.is_available():
                    with torch.no_grad():
                        im = Variable(im.cuda())
                        label = Variable(label.cuda())
                output= net1(im)
                output=net2(output.squeeze())
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)

        if valid_acc / len(valid_data) > best_val:                
            best_val = valid_acc / len(valid_data)
            torch.save(net1,os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(epoch)))
            torch.save(net2,os.path.join(mo_dir,'Resnet2_{:03d}.pth'.format(epoch)))

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)
            if best_idx>0:
                os.remove(os.path.join(mo_dir,'Resnet1_{:03d}.pth'.format(best_idx))) #remove Old models
                os.remove(os.path.join(mo_dir,'Resnet2_{:03d}.pth'.format(best_idx))) #remove Old models
                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
            best_idx=epoch
        epoch_str = (
                "Epoch %d. TrLoss: %f, VaLoss: %f, Train Acc: %f, Valid Acc: %f"
                % (epoch, train_loss / (5.0*len_loader_S),valid_loss / len(valid_data),train_acc/(5.0*len_loader_S),
                    valid_acc / len(valid_data)))
        lr_decay1.step(valid_acc / len(valid_data))
        #lr_decay1.step() 
        prev_time = cur_time
        print(epoch_str + time_str)
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
MSE_loss=nn.MSELoss()
from loss import adentropy
def trainOneStep1(Fea1,Class1, train_data1,train_data2, train_data3,train_data4, train_data5, Target_data1,Target_data2, num_epochs, optimizer1, criterion1,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        Fea1=Fea1.cuda()
        Class1 = Class1.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        PredLoss = 0
        AdvLoss1 = 0
        StepLoss1    = 0
        StepLoss2 = 0
        train_acc1 = 0
        train_acc2 = 0
        test_acc1 =0
        test_acc2 =0
        test_acc_Sum =0
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)
         
        Target_data_iter1=iter(Target_data1)
               
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5),len(Target_data_iter1))

        val_label=[]
        val_plabel=[]
        val_plabel2=[]
        name_df1=pd.DataFrame()
        Fea1.train()
        Class1.train()
        i = 1 
        while i < len_loader_S+1 :
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()


            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5


            data_target1 = Target_data_iter1.next()
            im1tx,_ =data_target1
                      
            im1 = Variable(im1.cuda())
            im2 = Variable(im2.cuda())
            im3 = Variable(im3.cuda())
            im4 = Variable(im4.cuda())
            im5 = Variable(im5.cuda())

            im1tx = Variable(im1tx.cuda())
            
            label1 = Variable(label1.cuda()).squeeze()
            label2 = Variable(label2.cuda()).squeeze()
            label3 = Variable(label3.cuda()).squeeze()
            label4 = Variable(label4.cuda()).squeeze()
            label5 = Variable(label5.cuda()).squeeze()
   
            ##Use source samples
            optimizer1.zero_grad()

            _,gsx1,_,_,_,_=Fea1(im1)
            pred1sx1=Class1(gsx1.squeeze())
            pred1loss1=criterion1(pred1sx1,label1)
            
            _,gsx2,_,_,_,_=Fea1(im2)
            pred1sx2=Class1(gsx2.squeeze())
            pred1loss2=criterion1(pred1sx2,label2)
            _,gsx3,_,_,_,_=Fea1(im3)
            pred1sx3=Class1(gsx3.squeeze())
            pred1loss3=criterion1(pred1sx3,label3)
            _,gsx4,_,_,_,_=Fea1(im4)
            pred1sx4=Class1(gsx4.squeeze())
            pred1loss4=criterion1(pred1sx4,label4)
            _,gsx5,_,_,_,_=Fea1(im5)
            pred1sx5=Class1(gsx5.squeeze())
            pred1loss5=criterion1(pred1sx5,label5)
         
            Loss_Clasi = pred1loss1+pred1loss2+pred1loss3+pred1loss4+pred1loss5 #Classification loss,
            
            Loss_Clasi.backward(retain_graph=True) 
            optimizer1.step()
            optimizer1.zero_grad()
                        
            ####    use target samples
            _,gtx,_,_,_,_=Fea1(im1tx)
            gtx=gtx.squeeze()
            loss_entropy = adentropy(Class1, gtx, lamda=0.1)
                       
            loss_entropy.backward()            
            optimizer1.step()
            optimizer1.zero_grad()       
                        
            PredLoss +=Loss_Clasi.item()
            AdvLoss1 += loss_entropy.item() 
            train_acc1 +=(get_acc(pred1sx1,label1)+get_acc(pred1sx2,label2)+get_acc(pred1sx3,label3)+get_acc(pred1sx4,label4)+get_acc(pred1sx5,label5))
            i +=1
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        ##########Test model
        Fea1.eval()
        Class1.eval()
        for im1tx,ty in Target_data2:
            
            ty=ty.squeeze()
            im1tx = Variable(im1tx.cuda())
            ty = Variable(ty.cuda())
            
            _,gtx,_,_,_,_=Fea1(im1tx)
            pred1tx_test=Class1(gtx.squeeze())
            predsum= softmax1(pred1tx_test)
            
            test_acc_Sum +=get_acc(predsum,ty)            
            
            _, pred_label = predsum.max(1) #Pred label
            pred_prob2=pred_label.cpu()
            pred_prob2=pred_prob2.detach().numpy()
            val_plabel2.append(pred_prob2)
            
            label2=ty.cpu()
            label2=label2.detach().numpy()
            
            val_label.append(label2)
            
        val_label=np.array(val_label)        
        len1=val_label.shape[0]*val_label.shape[1]        
        val_label=val_label.reshape(len1,1)

        val_plabel2=np.array(val_plabel2)
        val_plabel2=val_plabel2.reshape(len1,1) #Pred label

        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2
        epoch_str=("Epoch %d.PreLS: %f,AdLS1: %f,tra1: %f,Valid Acc: %f"
                % (epoch, PredLoss /(5.0*len_loader_S), AdvLoss1 /(5.0*len_loader_S),
                   train_acc1 / (5.0*len_loader_S),test_acc_Sum /len(Target_data2)))
        
        if test_acc_Sum /len(Target_data2) > best_val:                
            best_val = test_acc_Sum /len(Target_data2)

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)          
            torch.save(Fea1,os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(epoch)))
            torch.save(Class1,os.path.join(mo_dir,'Class1_{:03d}.pth'.format(epoch)))
            if best_idx>0:

                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Class1_{:03d}.pth'.format(best_idx)))
            best_idx=epoch        
          
        prev_time = cur_time
        #lr_decay1.step()
        lr_decay1.step(test_acc_Sum /len(Target_data2))
        print(epoch_str + time_str)        
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def trainOneStep2(Fea1,Class1, train_data1,train_data2,train_data3, Target_data1,Target_data2, num_epochs, optimizer1, criterion1,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        Fea1=Fea1.cuda()
        Class1 = Class1.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        PredLoss = 0
        AdvLoss1 = 0
        StepLoss1    = 0
        StepLoss2 = 0
        train_acc1 = 0
        train_acc2 = 0
        test_acc1 =0
        test_acc2 =0
        test_acc_Sum =0
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)

        Target_data_iter1=iter(Target_data1)
               
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Target_data_iter1))

        val_label=[]
        val_plabel=[]
        val_plabel2=[]
        name_df1=pd.DataFrame()
        Fea1.train()
        Class1.train()
        i = 1 
        while i < len_loader_S+1 :
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3

            data_target1 = Target_data_iter1.next()
            im1tx,_ =data_target1
                      
            im1 = Variable(im1.cuda())
            im2 = Variable(im2.cuda())
            im3 = Variable(im3.cuda())
            im1tx = Variable(im1tx.cuda())
            
            label1 = Variable(label1.cuda()).squeeze()
            label2 = Variable(label2.cuda()).squeeze()
            label3 = Variable(label3.cuda()).squeeze()       
            ##Use source samples
            optimizer1.zero_grad()

            _,gsx1,_,_,_,_=Fea1(im1)
            pred1sx1=Class1(gsx1.squeeze())
            pred1loss1=criterion1(pred1sx1,label1)
            
            _,gsx2,_,_,_,_=Fea1(im2)
            pred1sx2=Class1(gsx2.squeeze())
            pred1loss2=criterion1(pred1sx2,label2)

            
            _,gsx3,_,_,_,_=Fea1(im3)
            pred1sx3=Class1(gsx3.squeeze())
            pred1loss3=criterion1(pred1sx3,label3)
            
            
         
            Loss_Clasi = pred1loss1+pred1loss2+pred1loss3 #Classification loss,
            
            Loss_Clasi.backward(retain_graph=True) 
            optimizer1.step()
            optimizer1.zero_grad()
                        
            ####    use target samples
            _,gtx,fea1_cT,fea2_cT,fea3_cT,fea4_cT=Fea1(im1tx)
            gtx=gtx.squeeze()
            loss_entropy = adentropy(Class1, gtx, lamda=0.1)
            LossMom=msda_regulizer3(gsx1.squeeze(),gsx2.squeeze(),gsx3.squeeze(),gtx,4)/(im1tx.size(0)*im1tx.size(0))    
            Loss_adv=0.5*LossMom+loss_entropy
            Loss_adv.backward()            
            optimizer1.step()
            optimizer1.zero_grad()       
                        
            PredLoss +=Loss_Clasi.item()
            AdvLoss1 += Loss_adv.item() 
            train_acc1 +=(get_acc(pred1sx1,label1)+get_acc(pred1sx2,label2)+get_acc(pred1sx3,label3))
            i +=1
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        ##########Test model
        Fea1.eval()
        Class1.eval()
        for im1tx,ty in Target_data2:
            
            ty=ty.squeeze()
            im1tx = Variable(im1tx.cuda())
            ty = Variable(ty.cuda())
            
            _,gtx,_,_,_,_=Fea1(im1tx)
            pred1tx_test=Class1(gtx.squeeze())
            predsum= softmax1(pred1tx_test)
            
            test_acc_Sum +=get_acc(predsum,ty)            
            
            _, pred_label = predsum.max(1) #Pred label
            pred_prob2=pred_label.cpu()
            pred_prob2=pred_prob2.detach().numpy()
            val_plabel2.append(pred_prob2)
            
            label2=ty.cpu()
            label2=label2.detach().numpy()
            
            val_label.append(label2)
            
        val_label=np.array(val_label)        
        len1=val_label.shape[0]*val_label.shape[1]        
        val_label=val_label.reshape(len1,1)

        val_plabel2=np.array(val_plabel2)
        val_plabel2=val_plabel2.reshape(len1,1) #Pred label

        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2
        epoch_str=("Epoch %d.PreLS: %f,AdLS1: %f,tra1: %f,Valid Acc: %f"
                % (epoch, PredLoss /(3.0*len_loader_S), AdvLoss1 /(3.0*len_loader_S),
                   train_acc1 / (3.0*len_loader_S),test_acc_Sum /len(Target_data2)))
        
        if test_acc_Sum /len(Target_data2) > best_val:                
            best_val = test_acc_Sum /len(Target_data2)

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)          
            torch.save(Fea1,os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(epoch)))
            torch.save(Class1,os.path.join(mo_dir,'Class1_{:03d}.pth'.format(epoch)))
            if best_idx>0:

                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Class1_{:03d}.pth'.format(best_idx)))
            best_idx=epoch        
          
        prev_time = cur_time
        #lr_decay1.step()
        lr_decay1.step(test_acc_Sum /len(Target_data2))
        print(epoch_str + time_str)        
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
import torch

def euclidean(x1,x2):
	return ((x1-x2)**2).sum().sqrt()

def k_moment3(output_s1, output_s2, output_s3, output_t, k):
	output_s1 = (output_s1**k).mean(0)
	output_s2 = (output_s2**k).mean(0)
	output_s3 = (output_s3**k).mean(0)
	output_t = (output_t**k).mean(0)
	return  euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t)+ \
		euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1)

def msda_regulizer3(output_s1, output_s2, output_s3, output_t, belta_moment):
	# print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))
	s1_mean = output_s1.mean(0)
	s2_mean = output_s2.mean(0)
	s3_mean = output_s3.mean(0)
	t_mean = output_t.mean(0)
	output_s1 = output_s1 - s1_mean
	output_s2 = output_s2 - s2_mean
	output_s3 = output_s3 - s3_mean
	output_t = output_t - t_mean
	moment1 = euclidean(output_s1, output_t) + euclidean(output_s2, output_t) + euclidean(output_s3, output_t)+\
		euclidean(output_s1, output_s2) + euclidean(output_s2, output_s3) + euclidean(output_s3, output_s1) 
	reg_info = moment1
	#print(reg_info)
	for i in range(belta_moment-1):
		reg_info += k_moment3(output_s1,output_s2,output_s3, output_t,i+2)
	
	return reg_info/6
def trainOneStep3(Fea1,Class1, train_data1,train_data2,train_data3, Target_data1,Target_data2, num_epochs, optimizer1, criterion1,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        Fea1=Fea1.cuda()
        Class1 = Class1.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        PredLoss = 0
        AdvLoss1 = 0
        StepLoss1    = 0
        StepLoss2 = 0
        train_acc1 = 0
        train_acc2 = 0
        test_acc1 =0
        test_acc2 =0
        test_acc_Sum =0
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)

        Target_data_iter1=iter(Target_data1)
               
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Target_data_iter1))

        val_label=[]
        val_plabel=[]
        val_plabel2=[]
        name_df1=pd.DataFrame()
        Fea1.train()
        Class1.train()
        i = 1 
        while i < len_loader_S+1 :
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3

            data_target1 = Target_data_iter1.next()
            im1tx,_ =data_target1
                      
            im1 = Variable(im1.cuda())
            im2 = Variable(im2.cuda())
            im3 = Variable(im3.cuda())
            im1tx = Variable(im1tx.cuda())
            
            label1 = Variable(label1.cuda()).squeeze()
            label2 = Variable(label2.cuda()).squeeze()
            label3 = Variable(label3.cuda()).squeeze()       
            ##Use source samples
            optimizer1.zero_grad()

            _,gsx1,fea1_c1,fea2_c1,fea3_c1,fea4_c1=Fea1(im1) #Four layer features,
            pred1sx1=Class1(gsx1.squeeze())
            pred1loss1=criterion1(pred1sx1,label1)
            
            _,gsx2,fea1_c2,fea2_c2,fea3_c2,fea4_c2=Fea1(im2)
            pred1sx2=Class1(gsx2.squeeze())
            pred1loss2=criterion1(pred1sx2,label2)

            
            _,gsx3,fea1_c3,fea2_c3,fea3_c3,fea4_c3=Fea1(im3)
            pred1sx3=Class1(gsx3.squeeze())
            pred1loss3=criterion1(pred1sx3,label3)
            
            Loss_atten1=MSE_loss(fea1_c1,fea1_c2)+MSE_loss(fea1_c1,fea1_c3)+MSE_loss(fea1_c2,fea1_c3) #layer1
            Loss_atten2=MSE_loss(fea2_c1,fea2_c2)+MSE_loss(fea2_c1,fea2_c3)+MSE_loss(fea2_c2,fea2_c3)
            Loss_atten3=MSE_loss(fea3_c1,fea3_c2)+MSE_loss(fea3_c1,fea3_c3)+MSE_loss(fea3_c2,fea3_c3)
            Loss_atten4=MSE_loss(fea4_c1,fea4_c2)+MSE_loss(fea4_c1,fea4_c3)+MSE_loss(fea4_c2,fea4_c3)
         
            Loss_Clasi = pred1loss1+pred1loss2+pred1loss3 +0.1*(Loss_atten1+Loss_atten2+Loss_atten3+Loss_atten4) #Classification loss,
            
            Loss_Clasi.backward(retain_graph=True) 
            optimizer1.step()
            optimizer1.zero_grad()
                        
            ####    use target samples
            _,gtx,fea1_cT,fea2_cT,fea3_cT,fea4_cT=Fea1(im1tx)
            gtx=gtx.squeeze()

            Loss_attenT1=MSE_loss(fea1_c1,fea1_cT)+MSE_loss(fea2_c1,fea2_cT)+MSE_loss(fea3_c1,fea3_cT)+MSE_loss(fea4_c1,fea4_cT) #source1-->target
            Loss_attenT2=MSE_loss(fea1_c2,fea1_cT)+MSE_loss(fea2_c2,fea2_cT)+MSE_loss(fea3_c2,fea3_cT)+MSE_loss(fea4_c2,fea4_cT) #source2-->target
            Loss_attenT3=MSE_loss(fea1_c3,fea1_cT)+MSE_loss(fea2_c3,fea2_cT)+MSE_loss(fea3_c3,fea3_cT)+MSE_loss(fea4_c3,fea4_cT) #source3-->target

            loss_entropy = adentropy(Class1, gtx, lamda=0.1)+0.1*(Loss_attenT1+Loss_attenT2+Loss_attenT3)
            loss_entropy.backward()            
            optimizer1.step()
            optimizer1.zero_grad()       
                        
            PredLoss +=Loss_Clasi.item()
            AdvLoss1 += loss_entropy.item() 
            train_acc1 +=(get_acc(pred1sx1,label1)+get_acc(pred1sx2,label2)+get_acc(pred1sx3,label3))
            i +=1
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        ##########Test model
        Fea1.eval()
        Class1.eval()
        for im1tx,ty in Target_data2:
            
            ty=ty.squeeze()
            im1tx = Variable(im1tx.cuda())
            ty = Variable(ty.cuda())
            
            _,gtx,_,_,_,_=Fea1(im1tx)
            pred1tx_test=Class1(gtx.squeeze())
            predsum= softmax1(pred1tx_test)
            
            test_acc_Sum +=get_acc(predsum,ty)            
            
            _, pred_label = predsum.max(1) #Pred label
            pred_prob2=pred_label.cpu()
            pred_prob2=pred_prob2.detach().numpy()
            val_plabel2.append(pred_prob2)
            
            label2=ty.cpu()
            label2=label2.detach().numpy()
            
            val_label.append(label2)
            
        val_label=np.array(val_label)        
        len1=val_label.shape[0]*val_label.shape[1]        
        val_label=val_label.reshape(len1,1)

        val_plabel2=np.array(val_plabel2)
        val_plabel2=val_plabel2.reshape(len1,1) #Pred label

        #name_df1['real']=val_label
        #name_df1['pred2']=val_plabel2
        epoch_str=("Epoch %d.PreLS: %f,AdLS1: %f,tra1: %f,Valid Acc: %f"
                % (epoch, PredLoss /(3.0*len_loader_S), AdvLoss1 /(3.0*len_loader_S),
                   train_acc1 / (3.0*len_loader_S),test_acc_Sum /len(Target_data2)))
        
        if test_acc_Sum /len(Target_data2) > best_val:                
            best_val = test_acc_Sum /len(Target_data2)

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)          
            torch.save(Fea1,os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(epoch)))
            torch.save(Class1,os.path.join(mo_dir,'Class1_{:03d}.pth'.format(epoch)))
            if best_idx>0:

                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Class1_{:03d}.pth'.format(best_idx)))
            best_idx=epoch        
          
        prev_time = cur_time
        #lr_decay1.step()
        lr_decay1.step(test_acc_Sum /len(Target_data2))
        print(epoch_str + time_str)        
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
soft0=nn.Softmax(dim=0)
def trainOneStep10(Fea1,Class1,Dom_C, train_data1,train_data2,train_data3,train_data4,train_data5, Target_data1,Target_data2, num_epochs, optimizer1, criterion1,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        Fea1=Fea1.cuda()
        Class1 = Class1.cuda()
        Dom_C =Dom_C.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        PredLoss = 0
        AdvLoss1 = 0
        StepLoss1    = 0
        StepLoss2 = 0
        train_acc1 = 0
        train_acc2 = 0
        test_acc1 =0
        test_acc2 =0
        test_acc_Sum =0
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)

        Target_data_iter1=iter(Target_data1)
               
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5),len(Target_data_iter1))

        val_label=[]
        val_plabel=[]
        val_plabel2=[]
        name_df1=pd.DataFrame()
        Fea1.train()
        Class1.train()
        Dom_C.train()
        i = 1 
        while i < len_loader_S+1 :
            p = float(i + epoch * len_loader_S) / num_epochs / len_loader_S
            alpha = 2. / (1. + np.exp(-10 * p)) - 1 #alpha的作用,类似于一种权重,反向传播时对domainloss加权
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5

            data_target1 = Target_data_iter1.next()
            im1tx,_ =data_target1
                      
            im1 = Variable(im1.cuda())
            im2 = Variable(im2.cuda())
            im3 = Variable(im3.cuda())
            im4 = Variable(im4.cuda())
            im5 = Variable(im5.cuda())
            im1tx = Variable(im1tx.cuda())
            
            label1 = Variable(label1.cuda()).squeeze()
            label2 = Variable(label2.cuda()).squeeze()
            label3 = Variable(label3.cuda()).squeeze()
            label4 = Variable(label4.cuda()).squeeze()
            label5 = Variable(label5.cuda()).squeeze()
            domain_label = torch.ones(label1.size(0)).long().cuda()
            ##Use source samples
            optimizer1.zero_grad()

            _,gsx1,_,_,_,_=Fea1(im1)
            _,gsx2,_,_,_,_=Fea1(im2)            
            _,gsx3,_,_,_,_=Fea1(im3)
            _,gsx4,_,_,_,_=Fea1(im4)            
            _,gsx5,_,_,_,_=Fea1(im5)
            
            f_Dom1=Dom_C(gsx1.squeeze(),reverse=True,eta=alpha)
            LS_Dom1=criterion1(f_Dom1,domain_label)
            f_Dom2=Dom_C(gsx2.squeeze(),reverse=True,eta=alpha)
            LS_Dom2=criterion1(f_Dom2,domain_label)
            f_Dom3=Dom_C(gsx3.squeeze(),reverse=True,eta=alpha)
            LS_Dom3=criterion1(f_Dom3,domain_label)
            f_Dom4=Dom_C(gsx4.squeeze(),reverse=True,eta=alpha)
            LS_Dom4=criterion1(f_Dom4,domain_label)
            f_Dom5=Dom_C(gsx5.squeeze(),reverse=True,eta=alpha)
            LS_Dom5=criterion1(f_Dom5,domain_label)
            Loss_Dom=LS_Dom1+LS_Dom2+LS_Dom3+LS_Dom4+LS_Dom5
            #print('f_Dom1***',f_Dom1.shape)
            pred1sx1=Class1((soft0(f_Dom1[:,1].unsqueeze(1)).detach())*gsx1.squeeze())
            pred1loss1=criterion1(pred1sx1,label1)
            pred1sx2=Class1((soft0(f_Dom2[:,1].unsqueeze(1)).detach())*gsx2.squeeze())
            pred1loss2=criterion1(pred1sx2,label2)            
            pred1sx3=Class1((soft0(f_Dom3[:,1].unsqueeze(1)).detach())*gsx3.squeeze())
            pred1loss3=criterion1(pred1sx3,label3)
            pred1sx4=Class1((soft0(f_Dom4[:,1].unsqueeze(1)).detach())*gsx4.squeeze())
            pred1loss4=criterion1(pred1sx4,label4)
            pred1sx5=Class1((soft0(f_Dom5[:,1].unsqueeze(1)).detach())*gsx5.squeeze())
            pred1loss5=criterion1(pred1sx5,label5)
            
            Loss_Clasi = pred1loss1+pred1loss2+pred1loss3+pred1loss4+pred1loss5 #Classification loss,
            Loss_Clasi.backward(retain_graph=True) 
            Loss_Dom.backward(retain_graph=True)
            optimizer1.step()
            optimizer1.zero_grad()
                        
            ####    use target samples
            domain_label = torch.zeros(im1tx.size(0)).long().cuda()
            _,gtx,_,_,_,_=Fea1(im1tx)
            gtx=gtx.squeeze()
            loss_entropy = adentropy(Class1, gtx, lamda=0.1)
            
            f_DomT=Dom_C(gtx,reverse=True,eta=alpha)
            LS_DomT=criterion1(f_DomT,domain_label)
            
            LS_DomT.backward(retain_graph=True)
            loss_entropy.backward()            
            optimizer1.step()
            optimizer1.zero_grad()       
                        
            PredLoss +=Loss_Clasi.item()
            AdvLoss1 += loss_entropy.item() 
            train_acc1 +=(get_acc(pred1sx1,label1)+get_acc(pred1sx2,label2)+get_acc(pred1sx3,label3)+get_acc(pred1sx4,label4)+get_acc(pred1sx5,label5))
            i +=1
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        ##########Test model
        Fea1.eval()
        Class1.eval()
        Dom_C.eval()
        with torch.no_grad():
            total_correct=0
            total_num=0
            for im1tx,ty in Target_data2:
                im1tx = Variable(im1tx.cuda())
                ty = Variable(ty.cuda())
                _,gtx,_,_,_,_=Fea1(im1tx) # (4,2048)
                #print('gtx***',gtx.shape)
                pred1tx_test=Class1(gtx)
                _, pred = torch.max(pred1tx_test, 1)
                correct=torch.eq(pred,ty).float().sum().item()
                total_correct +=correct
                total_num     +=im1tx.size(0)
            Val_ACC=total_correct/total_num            
        epoch_str=("Epoch %d.PreLS: %f,AdLS1: %f,tra1: %f,Valid Acc: %f"
                % (epoch, PredLoss /(3.0*len_loader_S), AdvLoss1 /(3.0*len_loader_S),
                   train_acc1 / (3.0*len_loader_S),Val_ACC))
        
        if Val_ACC > best_val:                
            best_val = Val_ACC

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)          
            torch.save(Fea1,os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(epoch)))
            torch.save(Class1,os.path.join(mo_dir,'Class1_{:03d}.pth'.format(epoch)))
            if best_idx>0:

                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Class1_{:03d}.pth'.format(best_idx)))
            best_idx=epoch        
          
        prev_time = cur_time
        #lr_decay1.step()
        lr_decay1.step(Val_ACC)
        print(epoch_str + time_str)        
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def trainOneStep11(Fea1,Class1,Dom_C, train_data1,train_data2,train_data3,train_data4,train_data5, Target_data1,Target_data2, num_epochs, optimizer1, criterion1,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        Fea1=Fea1.cuda()
        Class1 = Class1.cuda()
        Dom_C =Dom_C.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        PredLoss = 0
        AdvLoss1 = 0
        StepLoss1    = 0
        StepLoss2 = 0
        train_acc1 = 0
        train_acc2 = 0
        test_acc1 =0
        test_acc2 =0
        test_acc_Sum =0
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)

        Target_data_iter1=iter(Target_data1)
               
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5),len(Target_data_iter1))

        val_label=[]
        val_plabel=[]
        val_plabel2=[]
        name_df1=pd.DataFrame()
        Fea1.train()
        Class1.train()
        Dom_C.train()
        i = 1 
        while i < len_loader_S+1 :
            p = float(i + epoch * len_loader_S) / num_epochs / len_loader_S
            alpha = 2. / (1. + np.exp(-10 * p)) - 1 #alpha的作用,类似于一种权重,反向传播时对domainloss加权
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5

            data_target1 = Target_data_iter1.next()
            im1tx,_ =data_target1
                      
            im1 = Variable(im1.cuda())
            im2 = Variable(im2.cuda())
            im3 = Variable(im3.cuda())
            im4 = Variable(im4.cuda())
            im5 = Variable(im5.cuda())
            im1tx = Variable(im1tx.cuda())
            
            label1 = Variable(label1.cuda()).squeeze()
            label2 = Variable(label2.cuda()).squeeze()
            label3 = Variable(label3.cuda()).squeeze()
            label4 = Variable(label4.cuda()).squeeze()
            label5 = Variable(label5.cuda()).squeeze()
            domain_label = torch.ones(label1.size(0)).long().cuda()
            ##Use source samples
            optimizer1.zero_grad()

            _,gsx1,_,_,_,_=Fea1(im1)
            _,gsx2,_,_,_,_=Fea1(im2)            
            _,gsx3,_,_,_,_=Fea1(im3)
            _,gsx4,_,_,_,_=Fea1(im4)            
            _,gsx5,_,_,_,_=Fea1(im5)
            
            f_Dom1=Dom_C(gsx1.squeeze(),reverse=True,eta=alpha)
            LS_Dom1=criterion1(f_Dom1,domain_label)
            f_Dom2=Dom_C(gsx2.squeeze(),reverse=True,eta=alpha)
            LS_Dom2=criterion1(f_Dom2,domain_label)
            f_Dom3=Dom_C(gsx3.squeeze(),reverse=True,eta=alpha)
            LS_Dom3=criterion1(f_Dom3,domain_label)
            f_Dom4=Dom_C(gsx4.squeeze(),reverse=True,eta=alpha)
            LS_Dom4=criterion1(f_Dom4,domain_label)
            f_Dom5=Dom_C(gsx5.squeeze(),reverse=True,eta=alpha)
            LS_Dom5=criterion1(f_Dom5,domain_label)
            Loss_Dom=LS_Dom1+LS_Dom2+LS_Dom3+LS_Dom4+LS_Dom5
            #print('f_Dom1***',f_Dom1.shape)
            pred1sx1=Class1((soft0(f_Dom1[:,1].unsqueeze(1)).detach())*gsx1.squeeze())
            pred1loss1=criterion1(pred1sx1,label1)
            pred1sx2=Class1((soft0(f_Dom2[:,1].unsqueeze(1)).detach())*gsx2.squeeze())
            pred1loss2=criterion1(pred1sx2,label2)            
            pred1sx3=Class1((soft0(f_Dom3[:,1].unsqueeze(1)).detach())*gsx3.squeeze())
            pred1loss3=criterion1(pred1sx3,label3)
            pred1sx4=Class1((soft0(f_Dom4[:,1].unsqueeze(1)).detach())*gsx4.squeeze())
            pred1loss4=criterion1(pred1sx4,label4)
            pred1sx5=Class1((soft0(f_Dom5[:,1].unsqueeze(1)).detach())*gsx5.squeeze())
            pred1loss5=criterion1(pred1sx5,label5)
            
            Loss_Clasi = pred1loss1+pred1loss2+pred1loss3+pred1loss4+pred1loss5 #Classification loss,
            Loss_Clasi.backward(retain_graph=True) 
            Loss_Dom.backward(retain_graph=True)
            optimizer1.step()
            optimizer1.zero_grad()
                        
            ####    use target samples
            domain_label = torch.zeros(im1tx.size(0)).long().cuda()
            _,gtx,_,_,_,_=Fea1(im1tx)
            gtx=gtx.squeeze()
            loss_entropy = adentropy(Class1, gtx, lamda=0.1)
            
            f_DomT=Dom_C(gtx,reverse=True,eta=alpha)
            LS_DomT=criterion1(f_DomT,domain_label)
            
            LS_DomT.backward(retain_graph=True)
            loss_entropy.backward()            
            optimizer1.step()
            optimizer1.zero_grad()       
                        
            PredLoss +=Loss_Clasi.item()
            AdvLoss1 += loss_entropy.item() 
            train_acc1 +=(get_acc(pred1sx1,label1)+get_acc(pred1sx2,label2)+get_acc(pred1sx3,label3)+get_acc(pred1sx4,label4)+get_acc(pred1sx5,label5))
            i +=1
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        ##########Test model
        Fea1.eval()
        Class1.eval()
        Dom_C.eval()
        with torch.no_grad():
            total_correct=0
            total_num=0
            for im1tx,ty in Target_data2:
                im1tx = Variable(im1tx.cuda())
                ty = Variable(ty.cuda())
                _,gtx,_,_,_,_=Fea1(im1tx) # (4,2048)
                #print('gtx***',gtx.shape)
                pred1tx_test=Class1(gtx)
                _, pred = torch.max(pred1tx_test, 1)
                correct=torch.eq(pred,ty).float().sum().item()
                total_correct +=correct
                total_num     +=im1tx.size(0)
            Val_ACC=total_correct/total_num            
        epoch_str=("Epoch %d.PreLS: %f,AdLS1: %f,tra1: %f,Valid Acc: %f"
                % (epoch, PredLoss /(3.0*len_loader_S), AdvLoss1 /(3.0*len_loader_S),
                   train_acc1 / (3.0*len_loader_S),Val_ACC))
        
        if Val_ACC > best_val:                
            best_val = Val_ACC

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)          
            torch.save(Fea1,os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(epoch)))
            torch.save(Class1,os.path.join(mo_dir,'Class1_{:03d}.pth'.format(epoch)))
            torch.save(Dom_C,os.path.join(mo_dir,'Dom_C_{:03d}.pth'.format(epoch)))
            if best_idx>0:

                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Class1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Dom_C_{:03d}.pth'.format(best_idx)))
            best_idx=epoch        
          
        prev_time = cur_time
        #lr_decay1.step()
        lr_decay1.step(Val_ACC)
        print(epoch_str + time_str)        
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def trainOneStep11ii(Fea1,Class1,Dom_C, train_data1,train_data2,train_data3,train_data4,train_data5, Target_data1,Target_data2, num_epochs, optimizer1, criterion1,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        Fea1=Fea1.cuda()
        Class1 = Class1.cuda()
        Dom_C =Dom_C.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        PredLoss = 0
        AdvLoss1 = 0
        StepLoss1    = 0
        StepLoss2 = 0
        train_acc1 = 0
        train_acc2 = 0
        test_acc1 =0
        test_acc2 =0
        test_acc_Sum =0
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)

        Target_data_iter1=iter(Target_data1)
               
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5),len(Target_data_iter1))

        val_label=[]
        val_plabel=[]
        val_plabel2=[]
        name_df1=pd.DataFrame()
        Fea1.train()
        Class1.train()
        Dom_C.train()
        i = 1 
        while i < len_loader_S+1 :
            p = float(i + epoch * len_loader_S) / num_epochs / len_loader_S
            alpha = 2. / (1. + np.exp(-10 * p)) - 1 #alpha的作用,类似于一种权重,反向传播时对domainloss加权
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5

            data_target1 = Target_data_iter1.next()
            im1tx,_ =data_target1
                      
            im1 = Variable(im1.cuda())
            im2 = Variable(im2.cuda())
            im3 = Variable(im3.cuda())
            im4 = Variable(im4.cuda())
            im5 = Variable(im5.cuda())
            im1tx = Variable(im1tx.cuda())
            
            label1 = Variable(label1.cuda()).squeeze()
            label2 = Variable(label2.cuda()).squeeze()
            label3 = Variable(label3.cuda()).squeeze()
            label4 = Variable(label4.cuda()).squeeze()
            label5 = Variable(label5.cuda()).squeeze()
            domain_label = torch.zeros(label1.size(0)).long().cuda()
            ##Use source samples
            optimizer1.zero_grad()

            _,gsx1,_,_,_,_=Fea1(im1)
            _,gsx2,_,_,_,_=Fea1(im2)            
            _,gsx3,_,_,_,_=Fea1(im3)
            _,gsx4,_,_,_,_=Fea1(im4)            
            _,gsx5,_,_,_,_=Fea1(im5)
            
            f_Dom1=Dom_C(gsx1.squeeze(),reverse=False,eta=alpha)
            LS_Dom1=criterion1(f_Dom1,domain_label)
            f_Dom2=Dom_C(gsx2.squeeze(),reverse=False,eta=alpha)
            LS_Dom2=criterion1(f_Dom2,domain_label)
            f_Dom3=Dom_C(gsx3.squeeze(),reverse=False,eta=alpha)
            LS_Dom3=criterion1(f_Dom3,domain_label)
            f_Dom4=Dom_C(gsx4.squeeze(),reverse=False,eta=alpha)
            LS_Dom4=criterion1(f_Dom4,domain_label)
            f_Dom5=Dom_C(gsx5.squeeze(),reverse=False,eta=alpha)
            LS_Dom5=criterion1(f_Dom5,domain_label)
            Loss_Dom=LS_Dom1+LS_Dom2+LS_Dom3+LS_Dom4+LS_Dom5
            #print('f_Dom1***',f_Dom1.shape)
            pred1sx1=Class1((soft0(f_Dom1[:,1].unsqueeze(1)).detach())*gsx1.squeeze())
            pred1loss1=criterion1(pred1sx1,label1)
            pred1sx2=Class1((soft0(f_Dom2[:,1].unsqueeze(1)).detach())*gsx2.squeeze())
            pred1loss2=criterion1(pred1sx2,label2)            
            pred1sx3=Class1((soft0(f_Dom3[:,1].unsqueeze(1)).detach())*gsx3.squeeze())
            pred1loss3=criterion1(pred1sx3,label3)
            pred1sx4=Class1((soft0(f_Dom4[:,1].unsqueeze(1)).detach())*gsx4.squeeze())
            pred1loss4=criterion1(pred1sx4,label4)
            pred1sx5=Class1((soft0(f_Dom5[:,1].unsqueeze(1)).detach())*gsx5.squeeze())
            pred1loss5=criterion1(pred1sx5,label5)
            
            Loss_Clasi = pred1loss1+pred1loss2+pred1loss3+pred1loss4+pred1loss5 #Classification loss,
            Loss_Clasi.backward(retain_graph=True) 
            Loss_Dom.backward(retain_graph=True)
            optimizer1.step()
            optimizer1.zero_grad()
                        
            ####    use target samples
            domain_label = torch.ones(im1tx.size(0)).long().cuda()
            _,gtx,_,_,_,_=Fea1(im1tx)
            gtx=gtx.squeeze()
            loss_entropy = adentropy(Class1, gtx, lamda=0.1)
            
            f_DomT=Dom_C(gtx,reverse=True,eta=alpha)
            LS_DomT=criterion1(f_DomT,domain_label)
            
            LS_DomT.backward(retain_graph=True)
            loss_entropy.backward()            
            optimizer1.step()
            optimizer1.zero_grad()       
                        
            PredLoss +=Loss_Clasi.item()
            AdvLoss1 += loss_entropy.item() 
            train_acc1 +=(get_acc(pred1sx1,label1)+get_acc(pred1sx2,label2)+get_acc(pred1sx3,label3)+get_acc(pred1sx4,label4)+get_acc(pred1sx5,label5))
            i +=1
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        ##########Test model
        Fea1.eval()
        Class1.eval()
        Dom_C.eval()
        with torch.no_grad():
            total_correct=0
            total_num=0
            for im1tx,ty in Target_data2:
                im1tx = Variable(im1tx.cuda())
                ty = Variable(ty.cuda())
                _,gtx,_,_,_,_=Fea1(im1tx) # (4,2048)
                #print('gtx***',gtx.shape)
                pred1tx_test=Class1(gtx)
                _, pred = torch.max(pred1tx_test, 1)
                correct=torch.eq(pred,ty).float().sum().item()
                total_correct +=correct
                total_num     +=im1tx.size(0)
            Val_ACC=total_correct/total_num            
        epoch_str=("Epoch %d.PreLS: %f,AdLS1: %f,tra1: %f,Valid Acc: %f"
                % (epoch, PredLoss /(5.0*len_loader_S), AdvLoss1 /(5.0*len_loader_S),
                   train_acc1 / (5.0*len_loader_S),Val_ACC))
        
        if Val_ACC > best_val:                
            best_val = Val_ACC

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)          
            torch.save(Fea1,os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(epoch)))
            torch.save(Class1,os.path.join(mo_dir,'Class1_{:03d}.pth'.format(epoch)))
            torch.save(Dom_C,os.path.join(mo_dir,'Dom_C_{:03d}.pth'.format(epoch)))
            if best_idx>0:

                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Class1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Dom_C_{:03d}.pth'.format(best_idx)))
            best_idx=epoch        
          
        prev_time = cur_time
        #lr_decay1.step()
        lr_decay1.step(Val_ACC)
        print(epoch_str + time_str)        
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
def trainOneStep12(Fea1,Class1,Dom_C, train_data1,train_data2,train_data3,train_data4,train_data5, Target_data1,Target_data2, num_epochs, optimizer1, criterion1,lr_decay1,mo_dir):
    if torch.cuda.is_available():
        Fea1=Fea1.cuda()
        Class1 = Class1.cuda()
        Dom_C =Dom_C.cuda()
    prev_time = datetime.now()
    best_val =0
    best_idx =0
    for epoch in range(num_epochs):
        PredLoss = 0
        AdvLoss1 = 0
        StepLoss1    = 0
        StepLoss2 = 0
        train_acc1 = 0
        train_acc2 = 0
        test_acc1 =0
        test_acc2 =0
        test_acc_Sum =0
        Source_data_iter1=iter(train_data1)
        Source_data_iter2=iter(train_data2)
        Source_data_iter3=iter(train_data3)
        Source_data_iter4=iter(train_data4)
        Source_data_iter5=iter(train_data5)

        Target_data_iter1=iter(Target_data1)
               
        len_loader_S = min(len(Source_data_iter1),len(Source_data_iter2),len(Source_data_iter3),len(Source_data_iter4),len(Source_data_iter5),len(Target_data_iter1))

        val_label=[]
        val_plabel=[]
        val_plabel2=[]
        name_df1=pd.DataFrame()
        Fea1.train()
        Class1.train()
        Dom_C.train()
        i = 1 
        while i < len_loader_S+1 :
            p = float(i + epoch * len_loader_S) / num_epochs / len_loader_S
            alpha = 2. / (1. + np.exp(-10 * p)) - 1 #alpha的作用,类似于一种权重,反向传播时对domainloss加权
            data_source1 = Source_data_iter1.next()
            data_source2 = Source_data_iter2.next()
            data_source3 = Source_data_iter3.next()
            data_source4 = Source_data_iter4.next()
            data_source5 = Source_data_iter5.next()

            im1,label1 =data_source1
            im2,label2 =data_source2
            im3,label3 =data_source3
            im4,label4 =data_source4
            im5,label5 =data_source5

            data_target1 = Target_data_iter1.next()
            im1tx,_ =data_target1
                      
            im1 = Variable(im1.cuda())
            im2 = Variable(im2.cuda())
            im3 = Variable(im3.cuda())
            im4 = Variable(im4.cuda())
            im5 = Variable(im5.cuda())
            im1tx = Variable(im1tx.cuda())
            
            label1 = Variable(label1.cuda()).squeeze()
            label2 = Variable(label2.cuda()).squeeze()
            label3 = Variable(label3.cuda()).squeeze()
            label4 = Variable(label4.cuda()).squeeze()
            label5 = Variable(label5.cuda()).squeeze()
            domain_label = torch.zeros(label1.size(0)).long().cuda()
            ##Use source samples
            optimizer1.zero_grad()

            _,gsx1,_,_,_,_=Fea1(im1)
            _,gsx2,_,_,_,_=Fea1(im2)            
            _,gsx3,_,_,_,_=Fea1(im3)
            _,gsx4,_,_,_,_=Fea1(im4)            
            _,gsx5,_,_,_,_=Fea1(im5)
            
            f_Dom1=Dom_C(gsx1.squeeze(),reverse=True,eta=alpha)
            LS_Dom1=criterion1(f_Dom1,domain_label)
            f_Dom2=Dom_C(gsx2.squeeze(),reverse=True,eta=alpha)
            LS_Dom2=criterion1(f_Dom2,domain_label)
            f_Dom3=Dom_C(gsx3.squeeze(),reverse=True,eta=alpha)
            LS_Dom3=criterion1(f_Dom3,domain_label)
            f_Dom4=Dom_C(gsx4.squeeze(),reverse=True,eta=alpha)
            LS_Dom4=criterion1(f_Dom4,domain_label)
            f_Dom5=Dom_C(gsx5.squeeze(),reverse=True,eta=alpha)
            LS_Dom5=criterion1(f_Dom5,domain_label)
            Loss_Dom=LS_Dom1+LS_Dom2+LS_Dom3+LS_Dom4+LS_Dom5
            #print('f_Dom1***',f_Dom1.shape)
            pred1sx1=Class1((soft0(f_Dom1[:,1].unsqueeze(1)).detach())*gsx1.squeeze())
            pred1loss1=criterion1(pred1sx1,label1)
            pred1sx2=Class1((soft0(f_Dom2[:,1].unsqueeze(1)).detach())*gsx2.squeeze())
            pred1loss2=criterion1(pred1sx2,label2)            
            pred1sx3=Class1((soft0(f_Dom3[:,1].unsqueeze(1)).detach())*gsx3.squeeze())
            pred1loss3=criterion1(pred1sx3,label3)
            pred1sx4=Class1((soft0(f_Dom4[:,1].unsqueeze(1)).detach())*gsx4.squeeze())
            pred1loss4=criterion1(pred1sx4,label4)
            pred1sx5=Class1((soft0(f_Dom5[:,1].unsqueeze(1)).detach())*gsx5.squeeze())
            pred1loss5=criterion1(pred1sx5,label5)
            
            Loss_Clasi = pred1loss1+pred1loss2+pred1loss3+pred1loss4+pred1loss5 #Classification loss,
            Loss_Clasi.backward(retain_graph=True) 
            Loss_Dom.backward(retain_graph=True)
            optimizer1.step()
            optimizer1.zero_grad()
                        
            ####    use target samples
            domain_label = torch.ones(im1tx.size(0)).long().cuda()
            _,gtx,_,_,_,_=Fea1(im1tx)
            gtx=gtx.squeeze()
            loss_entropy = adentropy(Class1, gtx, lamda=0.1)
            
            f_DomT=Dom_C(gtx,reverse=True,eta=alpha)
            LS_DomT=criterion1(f_DomT,domain_label)
            
            LS_DomT.backward(retain_graph=True)
            loss_entropy.backward()            
            optimizer1.step()
            optimizer1.zero_grad()       
                        
            PredLoss +=Loss_Clasi.item()
            AdvLoss1 += loss_entropy.item() 
            train_acc1 +=(get_acc(pred1sx1,label1)+get_acc(pred1sx2,label2)+get_acc(pred1sx3,label3)+get_acc(pred1sx4,label4)+get_acc(pred1sx5,label5))
            i +=1
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        ##########Test model
        Fea1.eval()
        Class1.eval()
        Dom_C.eval()
        with torch.no_grad():
            total_correct=0
            total_num=0
            for im1tx,ty in Target_data2:
                im1tx = Variable(im1tx.cuda())
                ty = Variable(ty.cuda())
                _,gtx,_,_,_,_=Fea1(im1tx) # (4,2048)
                #print('gtx***',gtx.shape)
                pred1tx_test=Class1(gtx)
                _, pred = torch.max(pred1tx_test, 1)
                correct=torch.eq(pred,ty).float().sum().item()
                total_correct +=correct
                total_num     +=im1tx.size(0)
            Val_ACC=total_correct/total_num            
        epoch_str=("Epoch %d.PreLS: %f,AdLS1: %f,tra1: %f,Valid Acc: %f"
                % (epoch, PredLoss /(3.0*len_loader_S), AdvLoss1 /(3.0*len_loader_S),
                   train_acc1 / (3.0*len_loader_S),Val_ACC))
        
        if Val_ACC > best_val:                
            best_val = Val_ACC

            #name_df1.to_csv(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(epoch)),header =None)          
            torch.save(Fea1,os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(epoch)))
            torch.save(Class1,os.path.join(mo_dir,'Class1_{:03d}.pth'.format(epoch)))
            torch.save(Dom_C,os.path.join(mo_dir,'Dom_C_{:03d}.pth'.format(epoch)))
            if best_idx>0:

                #os.remove(os.path.join(mo_dir,'val_name_{:03d}.csv'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Fea1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Class1_{:03d}.pth'.format(best_idx)))
                os.remove(os.path.join(mo_dir,'Dom_C_{:03d}.pth'.format(best_idx)))
            best_idx=epoch        
          
        prev_time = cur_time
        #lr_decay1.step()
        lr_decay1.step(Val_ACC)
        print(epoch_str + time_str)        
    print("Best Valid Model is epoch",best_idx,"Valid ACC is ",best_val)
