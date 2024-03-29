import pandas as pd
from sklearn import model_selection
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import os
import sys
import pickle
from collections import defaultdict
from utils import *

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
    elif a["comp_type"] in ["binary", "multiclass", "2class"]:
        df = pd.read_parquet(f"../input/input-{comp_name}/train.parquet")
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
    else:
        df = pd.read_parquet(f"../input/input-{comp_name}/train.parquet")
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
            kf = model_selection.KFold(
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
        # df.to_csv(f"../configs/configs-{comp_name}/my_folds.csv", index=False)
        df.to_parquet(f"../configs/configs-{comp_name}/my_folds.parquet", index=False)

        useful_features = [a["id_name"]]
        with open(
            f"../configs/configs-{a['comp_name']}/useful_features_l_1.pkl", "wb"
        ) as f:
            pickle.dump(useful_features, f)
    elif a["data_type"] == "tabular":
        # df.to_csv(f"../configs/configs-{comp_name}/my_folds.csv", index=False)
        df.to_parquet(f"../input/input-{comp_name}/my_folds.parquet", index=False)
        # test = pd.read_csv(f"../input/input-{comp_name}/test.csv")

        # Now no need to save test in cnfigs as we are saving it in input only [heavy files in input folder]
        #test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
        ## test.to_csv(f"../configs/configs-{comp_name}/test.csv", index=False)
        #test.to_parquet(f"../configs/configs-{comp_name}/test.parquet", index=False)

        test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
        useful_features = test.drop(a["id_name"], axis=1).columns.tolist()
        with open(
            f"../configs/configs-{a['comp_name']}/useful_features_l_1.pkl", "wb"
        ) as f:
            pickle.dump(useful_features, f)

    # --------------------------------dump current
    current_dict = defaultdict()
    current_dict["current_level"] = 1
    current_dict["current_feature_no"] = 0
    current_dict["current_exp_no"] = 0
    current_dict["current_ens_no"] = 0
    with open(f"../configs/configs-{a['comp_name']}/current_dict.pkl", "wb") as f:
        pickle.dump(current_dict, f)
    # --------------------------------dump features_dict
    feat_dict = defaultdict()
    feat_dict["base"] = [useful_features, 0]
    #feat_dict["l_1_f_0"] = [useful_features, 0, "base"]
    with open(f"../configs/configs-{a['comp_name']}/features_dict.pkl", "wb") as f:
        pickle.dump(feat_dict, f)
    # ---------------------------------dump Table
    Table = pd.DataFrame(
        columns=[
            "exp_no",
            "model_name",
            "bv",
            "bp",
            "random_state",
            "with_gpu",
            "aug_type",
            "_dataset",
            "use_cutmix",
            "callbacks_list",
            "features_list",
            "level_no",
            "oof_fold_name",
            "opt_fold_name",
            "fold_no",
            "no_iterations",
            "prep_list",
            "metrics_name",
            "seed_mean",
            "seed_std",  # ---\
            "fold_mean",
            "fold_std",
            "pblb_single_seed",
            "pblb_all_seed",
            "pblb_all_fold",
            "notes",
        ]
    )
    with open(f"../configs/configs-{a['comp_name']}/Table.pkl", "wb") as f:
        pickle.dump(Table, f)
    # -------------------------------------------
