import glob
import os
import zipfile
from dataset import DataFromH5File
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from lit_models import MyModel
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from model import OneKernel, TwoKernel, DCNN, AlexNet, Net
from torchvision import transforms
from os import listdir

from dataset import DataFromFileFolder

def locVsglob(NN, NN_name):
    
    ckpt_path = "weights/%s_locvsglob_dataset_5x5" % (NN_name)
  
    test_dataset = DataFromH5File('dataset/test_dataset_5x5.h5')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=True)

    # init the autoencoder
    model = MyModel(NN)
    logger = TensorBoardLogger(save_dir='logs/', log_graph=True,
                               name='%s_locvsglob_dataset_5x5' % (NN_name))
    checkpoint_callback = ModelCheckpoint(dirpath= './weights', 
                                          filename = '%s_locvsglob_dataset_5x5' % (NN_name),
                                            monitor="val_loss", auto_insert_metric_name=True)
    trainer = pl.Trainer(accelerator="gpu", devices=[4], min_epochs=30, max_epochs=60, callbacks=[checkpoint_callback], logger=logger)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path, verbose=True)

def catVsdog(NN, NN_name):
    
    ckpt_path = "weights/%s_catvsdog_dataset_5x5.ckpt" % (NN_name)

    transform = transforms.Compose([
            transforms.Resize(224), # makes it easier for the GPU
            transforms.CenterCrop((179,179)),
            transforms.ToTensor()])


    train_dir = 'train'
    test_dir = 'test'
    
    data_list = glob.glob(os.path.join('dataset/catvsdog/test','*.jpg'))

    test_dataset = DataFromFileFolder(data_list, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=True)

    # init the autoencoder
    # NN = TwoKernel()
    model = MyModel(NN)
    logger = TensorBoardLogger(save_dir='logs/', log_graph=True,
                            name='%s_catvsdog_dataset_5x5' % (NN_name))
    checkpoint_callback = ModelCheckpoint(dirpath= './weights', 
                                      filename = '%s_catvsdog_dataset_5x5' % (NN_name),
                                        monitor = "val_loss", auto_insert_metric_name=True)
    trainer = pl.Trainer(accelerator="gpu", devices=[5], min_epochs=30, max_epochs=60, callbacks=[checkpoint_callback], logger=logger)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=ckpt_path, verbose=True)

if "__main__" == __name__:

    # locVsglob(NN=TwoKernel(in_channel=1), NN_name='TwoKernel')
    catVsdog(NN=Net(), NN_name='Net')