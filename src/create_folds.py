import pandas as pd
from sklearn import model_selection
import os 
import sys 
import pickle

"""
import os 
import sys 
import pickle
with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
    for i in x:
        comp_name = i
x.close()
with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
    a = pickle.load(f)
"""

if __name__ == "__main__":
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
        a = pickle.load(f)

    df = pd.read_csv(f"../input_{comp_name}/train.csv")
    df["fold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

    target_name = a['target_name']
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[target_name].values)):
        print(len(train_idx), len(val_idx))

        df.loc[val_idx, "fold"] = fold

    df.to_csv(f"../input_{comp_name}/train_folds.csv", index=False)
