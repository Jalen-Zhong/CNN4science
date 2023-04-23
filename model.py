'''
Author: Jalen-Zhong jelly_zhong.qz@foxmail.com
Date: 2023-04-12 11:25:10
LastEditors: Jalen-Zhong jelly_zhong.qz@foxmail.com
LastEditTime: 2023-04-22 20:06:33
FilePath: \local ability of CNN\model.py
Description: 
Reference or Citation: transformer-https://github.com/Runist/torch_Vision_Transformer/blob/master/model.py

Copyright (c) 2023 by jelly_zhong.qz@foxmail.com, All Rights Reserved. 
'''
import os
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from functools import partial
from collections import OrderedDict


class OneKernel(pl.LightningModule):
    def __init__(self, in_channel):
        super(OneKernel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=179, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.fc = nn.Sequential(
            nn.Linear(64, 2))

    def forward(self, x):
        x = self.layer1(x)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out
    
class TwoKernel(pl.LightningModule):
    def __init__(self, in_channel):
        super(TwoKernel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=90, stride=1, padding=0),#64_channel
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128,128, kernel_size=90, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        
        self.fc = nn.Sequential(
            nn.Linear(128, 2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out

class DCNN(pl.LightningModule):
    def __init__(self, in_channel):
        super(DCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),#64_channel
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.fc = nn.Sequential(
            nn.Linear(5*5*512, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out = x.view(x.size(0), -1)
        out = self.fc(out)
        return out   

class Hybird(pl.LightningModule):
    def __init__(self, in_channel):
        super(Hybird, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=179, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.fc = nn.Sequential(
            nn.Linear(5*5*512+32, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))
        
    def forward(self, x):
        x1 = self.layer1(x)
        out1 = x1.view(x1.size(0), -1)
        x2 = self.layer2(x)
        x2 = self.layer3(x2)
        x2= self.layer4(x2)
        x2 = self.layer5(x2)
        x2 = self.layer6(x2)
        out2 = x2.view(x2.size(0), -1)
        out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return out   