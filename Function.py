'''
Author: Jalen-Zhong jelly_zhong.qz@foxmail.com
Date: 2023-04-20 14:13:41
LastEditors: Jalen-Zhong jelly_zhong.qz@foxmail.com
LastEditTime: 2023-04-20 15:12:19
FilePath: \local ability of CNN\Function.py
Description: 
Reference or Citation: 

Copyright (c) 2023 by jelly_zhong.qz@foxmail.com, All Rights Reserved. 
'''
import numpy as np

def func_display(mat_size, kernel_size):
    kernel_num = ((mat_size - kernel_size) // 2) + 1
    


    # 生成元素为x的矩阵
    matrix = [['X_{}{}'.format(i, j) for j in range(mat_size)] for i in range(mat_size)]
    print(matrix)
    for num in range(kernel_num):
        w = 'w{}'.format(num)

        #生成元素为w的kernel
        kernel = [['{}_{}{}'.format(w, i, j) for j in range(kernel_size)] for i in range(kernel_size)]
        kernel = np.array(kernel)
        print(kernel)
    # 将列表转换为numpy数组
    # mat = np.array(mat)

func_display(7, 3)