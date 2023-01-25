import pandas as pd 
import numpy as np 
import os 
import sys  
import gc 
from utils import * 

def show_input_folders():
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    try:
        print("my_folds: ")
        my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
        for col in my_folds.columns:
            print(col)
        #my_folds = my_folds[["0","1"]]
        print(my_folds.head(3))
        print()
        print(my_folds.columns)
        print()
        print(my_folds.info())
        print()
        # for f in list(zip(my_folds.columns, my_folds.dtypes, my_folds.nunique())):
        #     print(f)
        # print()
        print(my_folds.shape)
        del my_folds 
        gc.collect()
        print("="*40)
        print()
    except: 
        print("Train not found")

    try:
        print("Train: ")
        train = pd.read_parquet(f"../input/input-{comp_name}/train.parquet")
        #train = train[["0","1"]]
        print(train.head(3))
        print()
        print(train.columns)
        print()
        print(train.info())
        print()
        print(train.shape)
        del train 
        gc.collect()
        print("="*40)
        print()
    except: 
        print("Train not found")

    try:
        print("Test: ")
        test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
        # for col in test.columns:
        #     print(col)
        # #test = test[["0","1"]]
        print(test.head(3))
        print()
        print(test.columns)
        print()
        print(test.info())
        print()
        print(test.shape)
        del test 
        gc.collect()
        print("="*40)
        print()
    except: 
        print("Test not found")

    try:
        print("Sample: ")
        sample = pd.read_parquet(f"../input/input-{comp_name}/sample.parquet")
        print(sample.head(3))
        print()
        print(sample.columns)
        print()
        print(sample.info())
        print()
        print(sample.shape)
        print("="*40)
        print()
    except: 
        print("Sample not found")

    try:
        import glob
        # All files and directories ending with .txt and that don't begin with a dot:
        names = glob.glob(f"../configs/configs-{comp_name}/train_feats/*.pkl")
        for n in names:
            print(f"{n}: ")
            sample = load_pickle(n)
            print(sample)
            #input()
            if len(list(set(sample))) == 1:
                print("same value")
                # nan_count, num_missing_std, medianad, median, min, skew, 
            print()

        # print(sample.head(3))
        # print()
        # print(sample.columns)
        # print()
        # print(sample.info())
        # print()
        # print(sample.shape)
        # print("="*40)
        # print()
    except: 
        print(f"../configs/configs-{comp_name}/train_feats/train_feat_l_1_f_10_std.pkl not found")


if __name__ == "__main__":
    show_input_folders()