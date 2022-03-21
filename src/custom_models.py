
# tez ----------------------------
import os
import albumentations as A
import pandas as pd
import numpy as np


import tez
from tez.datasets import ImageDataset
from tez.callbacks import EarlyStopping

import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn import metrics, model_selection, preprocessing
import timm

from sklearn.model_selection import KFold

# ignoring warnings
import warnings

warnings.simplefilter("ignore")

import os, cv2, json
from PIL import Image

import random
from custom_models import *
# ------------------------------

class UModel(tez.Model):
    def __init__(
        self,
        model_name,
        num_classes,
        learning_rate,
        n_train_steps
        #                 , warmup_ratio
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.n_train_steps = n_train_steps
        #        self.warmup_ratio = warmup_ratio
        self.model = timm.create_model(
            model_name, pretrained=True, in_chans=3, num_classes=num_classes
        )
        self.step_scheduler_after = "batch"

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        return opt

    def fetch_scheduler(self):
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return sch

    def forward(self, image, targets=None):
        x = self.model(image)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(x, targets)
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}