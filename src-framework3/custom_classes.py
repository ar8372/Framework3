# --> to pass it to pytorch dataloader
# > __init__
# > __len__
# > __getitem__

# each competition may require different
# preprocessing and scaling

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os
import sys
import pickle
import joblib
import imageio
import cv2
import os.path
import albumentations
from PIL import Image


class TabularDataset:
    def __init__(self, data, target):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx, :]
        target = self.targets[idx]

        return {
            "x": torch.tensor(sample, dtype=float),
            "y": torch.tensor(target, dtype=long),
        }


# classification / regression
class TextDataset:
    def __init__(self, data, targets, tokenizer):
        self.data = data  # list of texts
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(data)

    def __getitem__(self, idx):
        # main part
        text = self.data[idx]

        # len(self.target.shape) will catch (30,)
        if len(self.target.shape) == 2 and self.target.shape[1] > 1:
            target = self.targets[idx, :]
        else:
            target = self.targets[idx]
        # binary: 0, 1, 1, 0
        # multiclass: 1, 2, 0, 1
        # regr(single col/ multicol): 0.3, 4, 5
        # multilabel classification: [1, 0, 0, 1, 0], [1, 1, 0, 0, 0] # entity extraction
        # input_ids: text=> tokens i.e numbers

        input_ids = tokenizer(text)  # transformers
        # input_ids : set of numbers [101, 42, 27, 216]
        # these seq can be of different length so do padding

        return {
            "text": torch.tensor(input_ids, dtype=torch.long),
            "target": torch.tensor(
                target
            ),  # dtype= classification: torch.long, reg: torch.float)
        }


class BengaliDataset(Dataset):
    def __init__(self, image_paths, targets, img_height, img_width, transform):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../configs/configs-{comp_name}/locker.pkl", "rb") as f:
            self.locker = pickle.load(f)

        # self.csv = csv.reset_index()
        self.paths = image_paths
        self.targets = targets
        self.img_ids = self.paths  # csv[self.locker["id_name"]]
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        # return len(self.csv)
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        path = f"{path}.png"
        # print("Found", path)
        # img = Image.open(path)
        img = Image.open(r"../input/input-bengaliai/train_images/Train_10.png")
        img = np.array(img)
        # print("Found-->")
        # img = joblib.load(
        #     f"../input/input-{self.locker['comp_name']}/train_images/{img_id}.pkl"
        # )

        # reshape 137, 236
        # new dataset 128,128
        img = img.reshape(self.img_height, self.img_width).astype(np.uint8)
        img = 255 - img  #

        # make it 3dimensional (X,Y, RGB) if not
        img = img[:, :, np.newaxis]

        # np.repeat(item, no_times, along axis)
        # it repeats item along an axis
        # duplicates whole image 3 times to create RGB channels
        # img = img[img, 3, 2]
        img = np.repeat(img, 3, 2)
        # or
        # most of the models take RGB image convert PIL grayscale to "RGB"
        # img = Image.fromarray(image).convert("RGB")

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = torch.tensor(img, dtype=torch.float)
        img = torch.permute(img, (2, 1, 0))

        target_list = self.locker["target_name"]
        # target_1 = self.csv.iloc[index][target_list[0]].values
        # target_2 = self.csv.iloc[index][target_list[1]].values
        # target_3 = self.csv.iloc[index][target_list[2]].values
        target_1 = self.targets[idx, 0]  # .iloc[index][target_list[0]].values
        target_2 = self.targets[idx, 1]  # .iloc[index][target_list[1]].values
        target_3 = self.targets[idx, 2]  # .iloc[index][target_list[2]].values

        return {
            "image": torch.tensor(img, dtype=torch.float),
            "grapheme_root": torch.tensor(target_1, dtype=torch.long),
            "vowel_diacritic": torch.tensor(target_2, dtype=torch.long),
            "consonant_diacritic": torch.tensor(target_3, dtype=torch.long),
        }

        # return img, np.array([target_1, target_2, target_3])


# tez2 1 channel pretrained 3channel models
class DigitRecognizerDataset:
    def __init__(self, df, augmentations, model_name):  # temp adding model name
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../configs/configs-{comp_name}/locker.pkl", "rb") as f:
            self.locker = pickle.load(f)

        self.df = df
        self.targets = self.df[self.locker["target_name"]].values
        self.df = self.df.drop(columns=[self.locker["target_name"]])
        self.augmentations = augmentations

        if self.locker["comp_name"] == "twistmnist":
            self.images = self.df.to_numpy(dtype=np.float32).reshape((-1, 28, 50))
            self.images = self.images[:, :, -28:]  # remove the 1's
        else:
            self.images = self.df.to_numpy(dtype=np.float32).reshape((-1, 28, 28))
        self.model_name = model_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        # item: index_no
        target = self.targets[item]
        image = self.images[item]
        if self.model_name == "pretrained":
            image = 255 - image
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, 2)
            image = torch.tensor(image, dtype=torch.float)
            image = torch.permute(image, (2, 1, 0))
            # or
            # image = np.transpose(iamge, (2,0,1)).astype(np.float32)
            # pytorch expects batch size * no of channels * height * weidth\

        else:
            image = np.expand_dims(image, axis=0)

        # experimenting this is just for p1 which takes 1D input
        # image = image.reshape((-1))
        # target = target.reshape((-1))
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.float),
        }


# this is for keras cutmix
class CutMixImageDataGenerator:
    def __init__(self, generator1, generator2, img_size, batch_size):
        self.batch_index = 0
        self.samples = generator1.samples
        self.class_indices = generator1.class_indices
        self.generator1 = generator1
        self.generator2 = generator2
        self.img_size = img_size
        self.batch_size = batch_size

    def reset_index(self):  # Ordering Reset (If Shuffle is True, Shuffle Again)
        self.generator1._set_index_array()
        self.generator2._set_index_array()

    def reset(self):
        self.batch_index = 0
        self.generator1.reset()
        self.generator2.reset()
        self.reset_index()

    def get_steps_per_epoch(self):
        quotient, remainder = divmod(self.samples, self.batch_size)
        return (quotient + 1) if remainder else quotient

    def __len__(self):
        self.get_steps_per_epoch()

    def __next__(self):
        if self.batch_index == 0:
            self.reset()

        crt_idx = self.batch_index * self.batch_size
        if self.samples > crt_idx + self.batch_size:
            self.batch_index += 1
        else:  # If current index over number of samples
            self.batch_index = 0

        reshape_size = self.batch_size
        last_step_start_idx = (self.get_steps_per_epoch() - 1) * self.batch_size
        if crt_idx == last_step_start_idx:
            reshape_size = self.samples - last_step_start_idx

        X_1, y_1 = self.generator1.next()
        X_2, y_2 = self.generator2.next()

        cut_ratio = np.random.beta(a=1, b=1, size=reshape_size)
        cut_ratio = np.clip(cut_ratio, 0.2, 0.8)
        label_ratio = cut_ratio.reshape(reshape_size, 1)
        cut_img = X_2

        X = X_1
        for i in range(reshape_size):
            cut_size = int((self.img_size - 1) * cut_ratio[i])
            y1 = random.randint(0, (self.img_size - 1) - cut_size)
            x1 = random.randint(0, (self.img_size - 1) - cut_size)
            y2 = y1 + cut_size
            x2 = x1 + cut_size
            cut_arr = cut_img[i][y1:y2, x1:x2]
            cutmix_img = X_1[i]
            cutmix_img[y1:y2, x1:x2] = cut_arr
            X[i] = cutmix_img

        y = y_1 * (1 - (label_ratio**2)) + y_2 * (label_ratio**2)
        return X, y

    def __iter__(self):
        while True:
            yield next(self)
