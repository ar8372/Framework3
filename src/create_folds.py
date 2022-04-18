import pandas as pd
from sklearn import model_selection
import os
import sys
import pickle
from collections import defaultdict

"""
import os 
import sys 
import pickle
with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
    for i in x:
        comp_name = i
x.close()
with open(f"../models-{comp_name}/locker.pkl", "rb") as f:
    a = pickle.load(f)
"""

if __name__ == "__main__":
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    with open(f"../models-{comp_name}/locker.pkl", "rb") as f:
        a = pickle.load(f)

    df = pd.read_csv(f"../models-{comp_name}/train.csv")
    df["fold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(
        n_splits=a["no_folds"], shuffle=True, random_state=23
    )

    target_name = a["target_name"]
    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X=df, y=df[target_name].values)
    ):
        print(len(train_idx), len(val_idx))

        df.loc[val_idx, "fold"] = fold

    if a["data_type"] in ["image_path", "image_df"]:
        df.to_csv(f"../models-{comp_name}/my_folds.csv", index=False)
        useful_features = [a["id_name"]]
        with open(f"../models-{a['comp_name']}/useful_features_l_1.pkl", "wb") as f:
            pickle.dump(useful_features, f)
    elif a["data_type"] == "tabular":
        df.to_csv(f"../models-{comp_name}/my_folds.csv", index=False)
        test = pd.read_csv(f"../input-{comp_name}/test.csv")
        test.to_csv(f"../models-{comp_name}/test.csv", index=False)

        useful_features = test.drop(a["id_name"], axis=1).columns.tolist()
        with open(f"../models-{a['comp_name']}/useful_features_l_1.pkl", "wb") as f:
            pickle.dump(useful_features, f)

    # --------------------------------dump current
    current_dict = defaultdict()
    current_dict["current_level"] = 1
    current_dict["current_feature_no"] = 0
    current_dict["current_exp_no"] = 0
    with open(f"../models-{a['comp_name']}/current_dict.pkl", "wb") as f:
        pickle.dump(current_dict, f)
    # --------------------------------dump features_dict
    feat_dict = defaultdict()
    feat_dict["l_1_f_0"] = [useful_features, 0, "base"]
    with open(f"../models-{a['comp_name']}/features_dict.pkl", "wb") as f:
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
            "features_list",
            "level_no",
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
    with open(f"../models-{a['comp_name']}/Table.pkl", "wb") as f:
        pickle.dump(Table, f)
    # -------------------------------------------
