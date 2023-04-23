'''
Author: Jalen-Zhong jelly_zhong.qz@foxmail.com
Date: 2023-04-22 16:14:09
LastEditors: Jalen-Zhong jelly_zhong.qz@foxmail.com
LastEditTime: 2023-04-22 16:56:09
FilePath: \local ability of CNN\model_ploy.py
Description: 
Reference or Citation: 

Copyright (c) 2023 by jelly_zhong.qz@foxmail.com, All Rights Reserved. 
'''
from torchviz import make_dot
from graphviz import Source
from model import OneKernel, DCNN, TwoKernel
import torch
# 创建模型
model = DCNN(in_channel=3)

# 生成输入张量
x = torch.randn(1, 3, 179, 179).requires_grad_(True)

# 使用torchviz绘制模型图
dot = make_dot(model(x),params=dict(model.named_parameters()))
dot.format = 'pdf'
dot.render(filename='DCNN', directory='Visual', cleanup=True)