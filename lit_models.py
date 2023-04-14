'''
Author: Jalen-Zhong jelly_zhong.qz@foxmail.com
Date: 2023-04-11 16:45:23
LastEditors: Jalen-Zhong jelly_zhong.qz@foxmail.com
LastEditTime: 2023-04-14 11:06:15
FilePath: \local ability of CNN\lit_models.py
Description: 
Reference or Citation: 

Copyright (c) 2023 by jelly_zhong.qz@foxmail.com, All Rights Reserved. 
'''
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from pytorch_lightning.callbacks import Callback


# define the LightningModule
class MyModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=2)
        
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)

        tensorboard_logs = {"train_loss":loss, "train_acc": self.train_acc}
        self.log_dict(tensorboard_logs, on_step=False, on_epoch=True, prog_bar=True)
    
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)

        tensorboard_logs = {"val_loss":loss, "val_acc":self.val_acc}
        self.log_dict(tensorboard_logs, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        logits = self.model(batch)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=0.0002)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, min_lr=1e-15)
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss"}

class MetricTracker(Callback):
    
    def __init__(self):
        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []
        self.lr = []

    def on_validation_epoch_end(self, trainer, module):
        val_loss = trainer.logged_metrics['val_loss'].item()
        val_acc = trainer.logged_metrics['val_acc'].item()
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        
    def on_train_epoch_end(self, trainer, module):
        train_loss = trainer.logged_metrics['train_loss'].item()
        train_acc = trainer.logged_metrics['train_acc'].item()
        lr = module.optimizers().param_groups[0]['lr']
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.lr.append(lr)