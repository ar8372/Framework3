import pandas as pd 
import numpy as np 
import os
import sys
import glob 
import joblib
from tqdm import tqdm  
import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        o = pickle.load(f)
    return o

if __name__ == "__main__":
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
    parquet_list = glob.glob(f"../input/input-{comp_name}/train_image_data_*.parquet")
    print(parquet_list)
    print()
    for f in parquet_list:
        df = pd.read_parquet(f)
        image_id = df.image_id.values 
        df.drop("image_id", axis=1, inplace=True)
        image_array = df.values # dataframe is super slow
        for j,img_id in tqdm(enumerate(image_id),total=len(image_id)):
            joblib.dump(image_array[j,:],f"../input/input-{locker['comp_name']}/{img_id}.pkl")
        