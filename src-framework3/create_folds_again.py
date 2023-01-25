import pandas as pd
from sklearn import model_selection
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import os
import sys
import pickle
from collections import defaultdict
from utils import *


# Here we just creates folds and we don't recreate tables
"""
import os
import sys 
import pickle
with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
    for i in x:
        comp_name = i
x.close()
with open(f"../configs/configs-{comp_name}/locker.pkl", "rb") as f:
    a = pickle.load(f)
"""

if __name__ == "__main__":
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    with open(f"../configs/configs-{comp_name}/locker.pkl", "rb") as f:
        a = pickle.load(f)
    locker = a 

    source= 'dummy'
    title = 'ver2'
    target_name = locker['target_name'] #"prediction"
    id_name = locker['id_name'] # "customer_ID"
    fold_list = ["fold3", "fold5", "fold10", "fold20"]
    # # test = pd.read_csv(f"../input/input-{comp_name}/test.csv")
    # test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
    # # test.to_csv(f"../configs/configs-{comp_name}/test.csv",index=False)
    # test.to_parquet(f"../configs/configs-{comp_name}/test.parquet", index=False)

    # df = pd.read_csv(f"../input/input-{comp_name}/train.csv")


    # df = df.sample(frac=1).reset_index(drop=True) # use later as it shuffles index
    # do it when you don't have id column or do it before creating Id columns
    # because for id col 0,1,2,3 it makes it look bad 12,3,0,111,...
    # for string id it is ok

    if a["comp_type"] == "multi_label":
        # df = df.sample(fracc=1).reset_index(drop=True)
        raise Exception("Now we have fold_dict instead of no_fold so change multi_label according in create folds")
        mskf = MultilabelStratifiedKFold(
            n_splits=a["no_folds"], shuffle=True, random_state=23
        )
        for fold, (train_idx, val_idx) in enumerate(
            mskf.split(df[a["id_name"]].values, df[a["target_name"]].values)
        ):
            print(len(train_idx), len(val_idx))
            df.loc[val_idx, "fold"] = fold
    else:
        try:
            df = pd.read_parquet(f"../input/input-{source}/train.parquet")
        except:
            try:
                df = pd.read_csv(f"../input/input-{source}/train.csv")
            except:
                raise Exception("train is neither parquet nor csv")
        if a["id_name"] in df.columns:  # always create new ID column for train # keep it simple
            df.drop(a["id_name"], axis=1, inplace=True)

        # reset for all
        fix_random(231) # to make result reproducible
        df = df.sample(frac=1).reset_index(drop=True)  # use later as it changes index
        df.index.name = a["id_name"]
        df = df.sort_index().reset_index() # sorting is very necessary
        for f,v in a['fold_dict'].items():
            # 'fold5', 5
            print(f,v,"=============>")
            df[f] = 1
            kf = model_selection.StratifiedKFold(
                n_splits=v, shuffle=True, random_state=23
            )
            target_name = a["target_name"]
            for fold, (train_idx, val_idx) in enumerate(
                kf.split(X=df, y=df[target_name].values)
            ):
                print(fold+1,":",len(train_idx), len(val_idx))

                df.loc[val_idx, f] = fold
            df[f] = df[f].astype('int8') # bydefault it is int64
            print()

    if a["data_type"] in ["image_path", "image_df"]:
        raise Exception()
        # df.to_csv(f"../configs/configs-{comp_name}/my_folds.csv", index=False)
        df.to_parquet(f"../configs/configs-{comp_name}/my_folds.parquet", index=False)

        useful_features = [a["id_name"]]
        with open(
            f"../configs/configs-{a['comp_name']}/useful_features_l_1.pkl", "wb"
        ) as f:
            pickle.dump(useful_features, f)
    elif a["data_type"] == "tabular":
        # df.to_csv(f"../configs/configs-{comp_name}/my_folds.csv", index=False)
        #df.to_parquet(f"../input/input-{comp_name}/my_folds.parquet", index=False)
        # test = pd.read_csv(f"../input/input-{comp_name}/test.csv")

        # Now no need to save test in cnfigs as we are saving it in input only [heavy files in input folder]
        #test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
        ## test.to_csv(f"../configs/configs-{comp_name}/test.csv", index=False)
        #test.to_parquet(f"../configs/configs-{comp_name}/test.parquet", index=False)
        try:
            test = pd.read_parquet(f"../input/input-{source}/test.parquet")
        except:
            try:
                # csv format data is present 
                test = pd.read_csv(f"../input/input-{source}/test.csv")
                #test.to_parquet(f"../input/input-{comp_name}/test.parquet", index=False)

                # sample = pd.read_csv(f"../input/input-{comp_name}/sample.csv")
                # sample.to_parquet(f"../input/input-{comp_name}/sample.parquet", index=False)
            except:
                raise Exception("test is neither parquet nor csv")


        all_columns = list(test.drop(id_name, axis=1).columns)
        useful_features_l_1 = load_pickle(f"../configs/configs-{a['comp_name']}/useful_features_l_1.pkl")


        # any true
        if any(x in useful_features_l_1 for x in all_columns): #all_columns in useful_features_l_1):
            # some already present
            pass 
            #raise Exception("Some features are already present in useful_features_l_1")

        print("These features will be added:")
        print(all_columns)
        v = input("Do you want to proceed? [y/n] : ")
        if v.lower() == 'n':
            raise Exception("Process Terminated")
        useful_features_l_1 += all_columns 
        useful_features_l_1 = list(set(useful_features_l_1))
        with open(
            f"../configs/configs-{a['comp_name']}/useful_features_l_1.pkl", "wb"
        ) as f:
            pickle.dump(useful_features_l_1, f)
        # 
        train = df[all_columns].copy()
        test = test[all_columns].copy()
        train.to_parquet(f"../input/input-{comp_name}/train_{title}.parquet")
        test.to_parquet(f"../input/input-{comp_name}/test_{title}.parquet")

        input_dict = load_pickle(f"../input/input-{comp_name}/input_dict.pkl")
        input_dict[title] = all_columns
        save_pickle(f"../input/input-{comp_name}/input_dict.pkl", input_dict)

        #-----------
        print("New data Sucessfully added")



