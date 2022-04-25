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
from torch.nn import (
    Linear,
    ReLU,
    CrossEntropyLoss,
    Sequential,
    Conv2d,
    MaxPool2d,
    Module,
    Softmax,
    BatchNorm2d,
    Dropout,
)
from torch.optim import Adam, SGD

# -----------------
import pretrainedmodels

class trainer_p1:
    def __init__(
        self, model, train_loader, valid_loader, optimizer, scheduler, use_cutmix
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", verbose=True, patience=7, factor=0.5
        )
        self.locc_fn = nn.CrossEntropyLoss()
        self.use_cutmix = use_cutmix

    def loss_fn(self, targets, output):
        device = "cuda"
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device)
        output = output.to(device)

        return nn.CrossEntropyLoss()(output, targets)

        # output is the prediction
        # targets is the true label
        output = torch.argmax(output, dim=1)
        output = output.unsqueeze(1)
        targets = targets.unsqueeze(1)  # use it for conv make it 2D
        print(output.shape, targets.shape)

        # For nn.CrossEntropyLoss the target has to be a single number from the interval [0, #classes]
        return nn.CrossEntropyLoss()(output, targets)
        # return nn.BCEWithLogitsLoss()(output, targets)

    def scheduler_fn(self):
        pass

    def optimizer_fn(self):
        learning_rate = 0.001
        pass

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self, data):
        inputs = data["image"]
        targets = data["targets"]

        self.lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(inputs.size()[0])

        target = data["targets"]
        self.shuffled_targets = target[rand_index]

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(inputs.size(), self.lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        self.lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2])
        )
        return inputs, targets

    def train_one_epoch(self):
        self.model.train()  # put model in train mode
        total_loss = 0
        for batch_index, data in enumerate(self.train_loader):
            loss = self.train_one_step(data)
            # loss = self.loss_fn(data["targets"], output)
            total_loss += loss
        return total_loss

    def train_one_step(self, data):
        self.optimizer.zero_grad()

        for k, v in data.items():
            data[k] = v.to("cuda")
        # make sure forward function of model has same keys
        # dictinary is passed using **
        if self.use_cutmix == True:
            inputs, targets = self.cutmix_data(data)
            output = self.model(inputs)  # **data)
            loss = self.loss_fn(targets, output) * self.lam + self.loss_fn(
                self.shuffled_targets, output
            ) * (1 - self.lam)
        else:
            output = self.model(data["image"])
            loss = self.loss_fn(data["targets"], output)

        
        loss.backward()
        self.optimizer.step()

        return loss

    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        for batch_index, data in enumerate(self.valid_loader):
            with torch.no_grad():
                loss = self.validate_one_step(data)
            total_loss += loss
        return total_loss

    def validate_one_step(self, data):
        for k, v in data.items():
            data[k] = v.to("cuda")
        # make sure forward function of model has same keys
        output = self.model(data["image"])  # **data)

        loss = self.loss_fn(data["targets"], output)
        return loss

    def fit(self, n_iter):
        for epoch in range(n_iter):
            epoch_loss = 0
            counter = 0
            train_loss = self.train_one_epoch()
            valid_loss = self.validate_one_epoch()
            if epoch % 2 == 0:
                print(
                    f"epoch {epoch}, train loss {train_loss}, valid loss {valid_loss}"
                )
        #self.optimizer.swap_swa_sgd() ## Here

    def predict_one_step(self, data):
        for k, v in data.items():
            data[k] = v.to("cuda")
        output = self.model(data["image"])
        return output

    def predict(self, test_loader):
        outputs = []

        with torch.no_grad():
            for batch_index, data in enumerate(test_loader):
                out = self.predict_one_step(data)

                outputs.append(
                    out
                )  # out.argmax(1) required when the final layer gives probabilites of classes and we want hard class

        preds = torch.cat(outputs)  # .view(-1) view(-1) is needed when we want 1D array
        return preds

    def save(self, path):
        state_dict = self.model.cpu().state_dict()
        self.model = self.model.cuda()
        torch.save(state_dict, path)

class pretrained_models(nn.Module):
    # basic pytorch model
    # conv2d(in_channels, out_channels):
    # in_channels:- no of channels in the input image
    # out_channel:- no of channels in the output image
    # kernel_size:- size of convolving kernel
    def __init__(self, no_features):
        super().__init__()
        model_name = 'resnet34'
        self.model = pretrainedmodels.__dict__[model_name](pretrained="imagenet")
        # adding a head
        in_features = self.model.last_linear.in_features 
        self.model.last_linear = torch.nn.Linear(in_features, 10)
        # self.layer0 = nn.Conv2d(in_channels = 3, out_channels = 50, kernel_size=3, padding=1)
        # self.layer1 = nn.Linear(50, 32)
        # self.layer2 = nn.Linear(32, 16)
        # self.layer3 = nn.Linear(16, 1)

        # self.cnn_layers = Sequential(
        #     # Defining a 2D convolution layer
        #     Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
        #     BatchNorm2d(4),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=2, stride=2),
        #     # Defining another 2D convolution layer
        #     Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
        #     BatchNorm2d(4),
        #     ReLU(inplace=True),
        #     MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.linear_layers = Sequential(Linear(4 * 7 * 7, 10))

    def forward(self, data):
        # batch_size, no_featrues : xtrain.shape
        # use this if now using 1D array in starting
        # xtrain = data
        # x = self.layer1(xtrain)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # return x
        
        x = data
        return self.model(x)

        # x = self.cnn_layers(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        # return x

class p1_model(nn.Module):
    # basic pytorch model
    # conv2d(in_channels, out_channels):
    # in_channels:- no of channels in the input image
    # out_channel:- no of channels in the output image
    # kernel_size:- size of convolving kernel
    def __init__(self, no_features):
        super().__init__()
        # self.layer0 = nn.Conv2d(in_channels = 3, out_channels = 50, kernel_size=3, padding=1)
        # self.layer1 = nn.Linear(50, 32)
        # self.layer2 = nn.Linear(32, 16)
        # self.layer3 = nn.Linear(16, 1)
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(Linear(4 * 7 * 7, 10))

    def forward(self, data):
        # batch_size, no_featrues : xtrain.shape
        # use this if now using 1D array in starting
        # xtrain = data
        # x = self.layer1(xtrain)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # return x
        x = data
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


# tez1
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
            targets = targets.type(torch.LongTensor)
            targets = targets.to("cuda")
            #print(targets.device, x.device, "these are devices") # very strong sanity check
            loss = nn.CrossEntropyLoss()(x, targets)
            metrics = self.monitor_metrics(x, targets)
            return x, loss, metrics
        return x, 0, {}
