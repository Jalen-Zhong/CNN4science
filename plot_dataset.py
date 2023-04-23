'''
Author: Jalen-Zhong jelly_zhong.qz@foxmail.com
Date: 2023-04-11 15:50:19
LastEditors: Jalen-Zhong jelly_zhong.qz@foxmail.com
LastEditTime: 2023-04-20 16:04:13
FilePath: \local ability of CNN\plot.py
Description: 
Reference or Citation: 

Copyright (c) 2023 by jelly_zhong.qz@foxmail.com, All Rights Reserved. 
'''
import numpy as np
from matplotlib import pyplot as plt

r = np.load('dataset/dataset_5x5.npz')
images = r['images']
labels = r['labels']
print(images.shape)
print(labels.shape)

for i in range(10):
    print(labels[i])
    plt.figure(figsize=(5, 5))
    plt.imshow(images[i][0],cmap='gray')
    print(images[i][0])
    plt.show()
