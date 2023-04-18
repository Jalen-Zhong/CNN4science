'''
Author: Jalen-Zhong jelly_zhong.qz@foxmail.com
Date: 2023-04-11 16:45:23
LastEditors: Jalen-Zhong jelly_zhong.qz@foxmail.com
LastEditTime: 2023-04-18 18:15:25
FilePath: \local ability of CNN\train.py
Description: 
Reference or Citation: 

Copyright (c) 2023 by jelly_zhong.qz@foxmail.com, All Rights Reserved. 
'''
import numpy as np
from dataset import DataFromH5File
import h5py
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from lit_models import MyModel, MetricTracker
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from model import OneKernel, TwoKernel, DCNN, VisionTransformer
from torchvision import transforms
from os import listdir
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from linformer import Linformer  
from vit_pytorch import ViT


def LocVsGlob(NN,NN_name,Data_name,GPU):
    
    # NetWork = 'Twokernel'
    dataset = DataFromH5File('dataset/180_train_%s_dataset_5x5.h5' % Data_name)
    # dataset = DataFromH5File('dataset/ViT_train_%s_dataset_5x5.h5' % Data_name)

    # split the train set into two
    train_set_size = int(len(dataset) * 0.9)
    valid_set_size = int(len(dataset) * 0.1)

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set, batch_size=64, num_workers=4, shuffle=True)
    val_loader = DataLoader(valid_set, batch_size=64, num_workers=4, shuffle=True)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    # init the autoencoder
    # NN = TwoKernel()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    metric_tracker = MetricTracker()
    model = MyModel(NN)
    logger = TensorBoardLogger(save_dir='logs/', log_graph=True,
                            name='%s_%s_locvsglob_dataset_5x5' % (NN_name, Data_name))
    checkpoint_callback = ModelCheckpoint(dirpath= './weights', 
                                      filename = '%s_%s_locvsglob_dataset_5x5' % (NN_name, Data_name),
                                        monitor = "val_loss", auto_insert_metric_name=True)
    trainer = pl.Trainer(accelerator="gpu", devices=[GPU], min_epochs=30, max_epochs=100, callbacks=[metric_tracker, checkpoint_callback, lr_monitor], logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

def CatVsDog(NN,NN_name,GPU):

    transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
    ])

    train_data = ImageFolder("dataset/catvsdog/training_set/training_set", transform=transform)
    test_data = ImageFolder("dataset/catvsdog/test_set/test_set", transform=transform)

    validation_split = 0.2
    random_seed = 11
    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size * validation_split))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    val_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, sampler=val_sampler)

    test_loader = DataLoader(test_data, batch_size=batch_size)

    # init the autoencoder
    # NN = TwoKernel()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    metric_tracker = MetricTracker()
    logger = TensorBoardLogger(save_dir='logs/', log_graph=True,
                            name='%s_catvsdog_dataset_5x5' % (NN_name))
    checkpoint_callback = ModelCheckpoint(dirpath= './weights', 
                                      filename = '%s_catvsdog_dataset_5x5' % (NN_name),
                                        monitor = "val_loss", auto_insert_metric_name=True)
    model = MyModel(NN)
    
    trainer = pl.Trainer(accelerator="gpu", devices=[GPU], min_epochs=30, max_epochs=100, callbacks=[metric_tracker, checkpoint_callback, lr_monitor], logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if "__main__" == __name__:

  # LocVsGlob(NN=TwoKernel(in_channel=1), NN_name='TwoKernel', Data_name = 'none', GPU = 0)
  LocVsGlob(NN=ViT(image_size=180, patch_size=18, num_classes=2, dim=128, depth=12, heads=12, mlp_dim=2048, channels=1), NN_name='VisionTransformer_18', Data_name = 'none', GPU = 2)
  # CatVsDog(NN=TwoKernel(in_channel=3), NN_name='TwoKernel', GPU = 0)
  # CatVsDog(NN=ViT(image_size=180, patch_size=18, num_classes=2, dim=128, depth=12, heads=12, mlp_dim=2048, channels=3), NN_name='VisionTransformer_18', GPU = 1)