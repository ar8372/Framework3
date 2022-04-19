
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import os

def prep():
    data_dir = '../input/input-bengali/'
    files_train = [f'train_image_data_{fid}.parquet' for fid in range(4)]

    for fname in files_train:
        F = os.path.join(data_dir, fname) 
        df_train = pd.read_parquet(F)
        img_ids = df_train['image_id'].values
        img_array = df_train.iloc[:, 1:].values
        for idx in tqdm(range(len(df_train))):
            img_id = img_ids[idx]
            img = img_array[idx]
            joblib.dump(img, f"../input/input-bengali/train_images/{img_id}.pkl")

if __name__ == "__main__":
    prep()