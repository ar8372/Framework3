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

# ------------------------------


class trainer_p1:
    def __init__(self, model, train_loader, valid_loader, optimizer, scheduler):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", verbose=True, patience=7, factor=0.5
        )
        self.locc_fn = nn.CrossEntropyLoss()

    def loss_fn(self, targets, output):
        return nn.BCEWithLogitsLoss()(output, targets)

    def scheduler_fn(self):
        pass

    def optimizer_fn(self):
        learning_rate = 0.001
        pass

    def train_one_epoch(self):
        self.model.train()  # put model in train mode
        total_loss = 0
        for batch_index, data in enumerate(self.train_loader):
            loss = self.train_one_step(data)
            loss = loss_fn(data["y"], output)
            train_loss += loss
        return total_loss

    def train_one_step(self, data):
        self.optimizer.zero_grad()

        for k, v in data.items():
            data[k] = v.to("cuda")
        # make sure forward function of model has same keys
        # dictinary is passed using **
        output = self.model(**data)
        loss = self.loss_fn(data["y"], output)
        #
        self.scheduler.step()
        loss.backward()
        self.optimizer.step()

        return loss

    def validate_one_epoch(self):
        total_loss = 0
        for batch_index, data in enumerate(self.valid_loader):
            with torch.no_grad():
                loss = self.validate_one_step(data)
            train_loss += loss
        return total_loss

    def validate_one_step(self, data):
        for k, v in data.items():
            data[k] = v.to("cuda")
        # make sure forward function of model has same keys
        output = self.model(**data)
        loss = self.loss_fn(data["y"], output)
        return loss

    def fit(self, n_iter):
        for epoch in range(n_iter):
            epoch_loss = 0
            counter = 0
            train_loss = self.train_one_epoch()
            valid_loss = self.validate_one_epoch()
            if epoch % 2 == 0:
                print(f"train loss {train_loss}, valid loss {valid_loss}")

    def predict_one_step(self, data):
        output, _, _ = self.model(data)
        return output

    def predict(self, test_loader):
        outputs = []

        with torch.no_grad():
            for batch_index, data in enumerate(test_loader):
                out = predict_one_step(data)

                outputs.append(out)
        # outputs is list of tensors
        preds = torch.cat(outputs).view(-1)
        return preds


class p1_model(nn.Module):
    # basic pytorch model
    def __init__(self, no_features):
        super().__init__()
        self.layer1 = nn.Linear(no_features, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)

    def forward(self, xtrain):
        # batch_size, no_featrues : xtrain.shape
        x = self.layer1(xtrain)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class UModel(tez.Model):  # nn.Module): #tez.Model):
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


#  tez2
class DigitRecognizerModel(nn.Module):
    def __init__(self, model_name, num_classes, learning_rate, n_train_steps):
        super().__init__()

        self.learning_rate = learning_rate
        self.n_train_steps = n_train_steps
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=1,
            num_classes=num_classes,
        )

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        targets = targets.cpu().detach().numpy()
        acc = metrics.accuracy_score(targets, outputs)
        acc = torch.tensor(acc, device=device)
        return {"accuracy": acc}

    def optimizer_scheduler(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            patience=2,
            verbose=True,
            mode="max",
            threshold=1e-4,
        )
        return opt, sch

    def forward(self, image, targets=None):
        x = self.model(image)
        if targets is not None:
            loss = nn.CrossEntropyLoss()(x, targets)
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}
