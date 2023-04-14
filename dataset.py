'''
Author: Jalen-Zhong jelly_zhong.qz@foxmail.com
Date: 2023-04-11 15:02:59
LastEditors: Jalen-Zhong jelly_zhong.qz@foxmail.com
LastEditTime: 2023-04-14 11:05:18
FilePath: \local ability of CNN\dataset.py
Description: 
Reference or Citation: 

Copyright (c) 2023 by jelly_zhong.qz@foxmail.com, All Rights Reserved. 
'''
import numpy as np
from random import randint
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import torch.utils.data as data
import h5py
import torch
from PIL import Image


def get_square_label(coords, pixels, rectangle_size):

    x, y = coords[0], coords[1]
    pixels[int(x - rectangle_size): int(x + rectangle_size), int(y - rectangle_size): int(y + rectangle_size)] = 255

    density_map = pixels / 255

    return density_map

def save_h5(path, images, labels):
    print('saving', path)
    with h5py.File(name=path, mode='w') as file:
        file['images'] = images
        file['labels'] = labels

def load_h5(path):
    print('loading', path)
    file = h5py.File(name=path, mode='r')
    return file['images'][:], file['labels'][:]

class DataFromH5File(data.Dataset):
    def __init__(self, filepath):
        h5File = h5py.File(filepath, 'r')
        self.x = h5File['images']
        self.y = h5File['labels']
        
    def __getitem__(self, idx):
        data = torch.from_numpy(self.x[idx]).float()
        label = torch.tensor(self.y[idx])

        return data, label
    
    def __len__(self):
        assert self.x.shape[0] == self.y.shape[0], "Wrong data length"
        return self.x.shape[0]

class DataFromFileFolder(data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.filelength = len(file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        tip = img_path.split("/")[-1].split(".")[0]
        iszero = 1 if tip == 'cat' else 0
        isone = 1 if tip == 'dog' else 0
        label = [iszero, isone]
        label = torch.tensor(label).float()
        return img_transformed, label
    
    def __len__(self):
        return self.filelength
    
# class DataFromFileFolder(data.Dataset):
#     def __init__(self, file_list, transform=None, use_gpu=torch.cuda.is_available(), test=False):
#         self.file_list = file_list
#         self.transform = transform
#         self.filelength = len(file_list)
#         self.use_gpu = use_gpu
#         self.test = test

#     def __len__(self):
#         return self.filelength

#     def __getitem__(self, idx):
#         img_path = self.file_list[idx]
#         img = Image.open(img_path)
#         img_transformed = self.transform(img)
#         label = img_path.split("/")[-1].split(".")[0]
#         if not self.test:
#             label = [1] if label == "dog" else [0]
#         else:
#             label = int(label)
#         label = torch.tensor(label).float()
#         return img_transformed, label


def generator(examples, image_size, rectangle_size, left_up_coors, right_down_coors, middle_coors, fix_coors):

    list_images = []
    list_labels = []

    for i in tqdm(range(examples)):
        left_up_pixels = np.zeros((image_size, image_size), np.uint8)
        right_down_pixels = np.zeros((image_size, image_size), np.uint8)
        middle_pixels = np.zeros((image_size, image_size), np.uint8)
        fix_pixels = np.zeros((image_size, image_size), np.uint8)
    
        left_up_seed = randint(0,1)
        right_down_seed = randint(0,1)
        middle_seed = randint(0,1)

        XOR = ( left_up_seed ^ right_down_seed )
        temp = np.int64(1) - XOR
        label = temp
        # iszero = 1 if temp == 0 else 0
        # isone = 1 if temp == 1 else 0
        # label = [iszero, isone]

        fix_pixels = get_square_label(fix_coors, fix_pixels, rectangle_size)
        if left_up_seed == 1:
            left_up_pixels = get_square_label(left_up_coors, left_up_pixels, rectangle_size)
        if right_down_seed == 1:
            right_down_pixels = get_square_label(right_down_coors, right_down_pixels, rectangle_size)
        if middle_seed == 1:
            middle_pixels = get_square_label(middle_coors, middle_pixels, rectangle_size)
        image = left_up_pixels + right_down_pixels + middle_pixels + fix_pixels
        image = image.reshape(1, image_size, image_size) / image.max()

        list_images.append(image)
        list_labels.append(label)

    return list_images, list_labels

if __name__ == "__main__":

    random.seed(2023)
    train_examples = 10000
    test_examples = 2000
    image_size = 179
    patch_number = 5
    patch_size = image_size / patch_number
    rectangle_size = patch_size / 4

    fix_coors = [image_size - patch_size/2, patch_size / 2]
    left_up_coors = [patch_size / 2, patch_size / 2]
    right_down_coors = [image_size - patch_size/2, image_size - patch_size/2]
    middle_coors = [image_size / 2, image_size / 2]

    images, labels = generator(train_examples, image_size, rectangle_size, left_up_coors, right_down_coors, middle_coors, fix_coors)
    save_h5('dataset/train_random_dataset_%dx%d.h5' % (patch_number, patch_number), images = images, labels = labels)
    images, labels = generator(test_examples, image_size, rectangle_size, left_up_coors, right_down_coors, middle_coors, fix_coors)
    save_h5('dataset/test_random_dataset_%dx%d.h5' % (patch_number, patch_number), images = images, labels = labels)

    # np.savez('dataset/dataset_%dx%d.npz' % (patch_number, patch_number), images = images, labels = labels)





