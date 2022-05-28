import pandas as pd
from sklearn import model_selection
import os
import sys
import pickle
from collections import defaultdict
from utils import *

"""
generates new features on top of some existing featrues. 
and stores this info as a dictionary.
"""


class features:
    def __init__(self):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        a = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
        # --------------------------------------
        self.locker = a
        # -------------------------------------
        self.level_no = None
        self.current_dict = None
        self.get_feat_no()  # load level_no and current_feature_no
        self.useful_features = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/useful_features_l_{self.level_no}.pkl"
        )
        self.my_folds = pd.read_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv"
        )
        self.test = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/test.csv")

    def change_level(self, new_val="--|--"):
        if new_val != "--|--":
            self.level_no = new_val
        else:
            self.level_no += 1
        self.current_dict["current_level"] = self.level_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl", self.current_dict
        )

    def display_features_generated(self):
        # display all the feature engineering done so far
        # Key:- f"l{self.level_no}_f{feat_no}"
        # value:- [created, from , info]
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        for key, value in self.feat_dict.items():
            print(key, f"{value[-1]} :-")
            print("features created:")
            print(value[0])
            print("from:")
            print(value[1])
            print()

    def show_variables(self):
        print()
        for i, (k, v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>", v)
        print()

    def get_feat_no(self):
        # exp_no, current_level, current_feature_no
        self.current_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl"
        )
        self.level_no = int(self.current_dict["current_level"])
        self.current_feature_no = int(self.current_dict["current_feature_no"])

    def isRepetition(self, gen_features, old_features, feat_title):
        # self.curr
        for key, value in self.feat_dict.items():
            f1, f2, ft = value
            if f2 == 0:
                # from base
                pass
            elif len(f1[0].split("_")[0]) < 5 or (
                f1[0].split("_")[0][0] == "l" and f1[0].split("_")[0][2] == "f"
            ):
                # originate from base so f2 can't be split
                f1 = ["_".join(f.split("_")[2:]) for f in f1]
                gen_features = ["_".join(f.split("_")[2:]) for f in gen_features]
            else:
                f2 = ["_".join(f.split("_")[2:]) for f in f2]
                old_features = ["_".join(f.split("_")[2:]) for f in old_features]
                f1 = ["_".join(f.split("_")[2:]) for f in f1]
                gen_features = ["_".join(f.split("_")[2:]) for f in gen_features]
            if f1 == gen_features and f2 == old_features and ft == feat_title:
                raise Exception("This feature is already present!")

    def create_statistical_features(self, useful_features="--|--"):
        feat_title = "create_statistical_features"

        self.my_folds = pd.read_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv"
        )
        self.test = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/test.csv")
        if useful_features == "--|--":
            useful_features = self.useful_features
        self.get_feat_no()  # --updated self.current_feature_no to the latest feat no
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_no = self.current_feature_no + 1
        # ------------------------------------------
        new_features = [
            f"l_{self.level_no}_f_{feat_no}_nan_count",
            f"l_{self.level_no}_f_{feat_no}_abs_sum",
            f"l_{self.level_no}_f_{feat_no}_sem",
            f"l_{self.level_no}_f_{feat_no}_std",
            f"l_{self.level_no}_f_{feat_no}_mad",
            f"l_{self.level_no}_f_{feat_no}_avg",
            f"l_{self.level_no}_f_{feat_no}_median",
            f"l_{self.level_no}_f_{feat_no}_max",
            f"l_{self.level_no}_f_{feat_no}_min",
            f"l_{self.level_no}_f_{feat_no}_skew",
            f"l_{self.level_no}_f_{feat_no}_num_missing_std",
        ]
        # -------------------------------------------------
        self.isRepetition(
            new_features, useful_features, feat_title
        )  # check for duplicate process
        # -------------------------------------------------
        self.test[f"l_{self.level_no}_f_{feat_no}_nan_count"] = self.test.isnull().sum(
            axis=1
        )
        self.my_folds[
            f"l_{self.level_no}_f_{feat_no}_nan_count"
        ] = self.my_folds.isnull().sum(axis=1)

        # self.my_folds[f'l{self.level_no}_f{feat_no}_n_missing'] = my_folds[useful_features].isna().sum(axis=1)
        self.my_folds[f"l_{self.level_no}_f_{feat_no}_abs_sum"] = (
            self.my_folds[useful_features].abs().sum(axis=1)
        )
        self.my_folds[f"l_{self.level_no}_f_{feat_no}_sem"] = self.my_folds[
            useful_features
        ].sem(axis=1)
        self.my_folds[f"l_{self.level_no}_f_{feat_no}_std"] = self.my_folds[
            useful_features
        ].std(axis=1)
        self.my_folds[f"l_{self.level_no}_f_{feat_no}_mad"] = self.my_folds[
            useful_features
        ].mad(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_avg"] = self.my_folds[
            useful_features
        ].mean(axis=1)
        self.my_folds[f"l_{self.level_no}_f_{feat_no}_median"] = self.my_folds[
            useful_features
        ].median(axis=1)
        self.my_folds[f"l_{self.level_no}_f-{feat_no}_max"] = self.my_folds[
            useful_features
        ].max(axis=1)
        self.my_folds[f"l_{self.level_no}_f_{feat_no}_min"] = self.my_folds[
            useful_features
        ].min(axis=1)
        self.my_folds[f"l_{self.level_no}_f_{feat_no}_skew"] = self.my_folds[
            useful_features
        ].skew(axis=1)
        self.my_folds[f"l_{self.level_no}_f_{feat_no}_num_missing_std"] = (
            self.my_folds[useful_features].isna().std(axis=1).astype("float")
        )

        # test[f'l{self.level_no}_f{feat_no}_n_missing'] = test[useful_features].isna().sum(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_abs_sum"] = (
            self.test[useful_features].abs().sum(axis=1)
        )
        self.test[f"l_{self.level_no}_f_{feat_no}_sem"] = self.test[
            useful_features
        ].sem(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_std"] = self.test[
            useful_features
        ].std(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_mad"] = self.test[
            useful_features
        ].mad(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_avg"] = self.test[
            useful_features
        ].mean(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_median"] = self.test[
            useful_features
        ].median(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_max"] = self.test[
            useful_features
        ].max(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_min"] = self.test[
            useful_features
        ].min(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_skew"] = self.test[
            useful_features
        ].skew(axis=1)
        self.test[f"l_{self.level_no}_f_{feat_no}_num_missing_std"] = (
            self.test[useful_features].isna().std(axis=1).astype("float")
        )

        # -----------------------------dump data
        self.my_folds.to_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv", index=False
        )
        self.test.to_csv(f"../configs/configs-{self.locker['comp_name']}/test.csv", index=False)

        # -----------------------------dump current dict
        self.current_feature_no = feat_no
        self.current_dict["current_level"] = self.level_no
        self.current_dict["current_feature_no"] = self.current_feature_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl", self.current_dict
        )
        # -----------------------------dump feature dictionary
        feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_dict[f"l_{self.level_no}_f_{feat_no}"] = [
            new_features,
            useful_features,
            feat_title,
        ]
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl", feat_dict
        )
        print("New features create:- ")
        print(new_features)

    def create_unique_characters(self, useful_features="--|--"):
        feat_title = "unique_characters"

        self.my_folds = pd.read_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv"
        )
        self.test = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/test.csv")
        if useful_features == "--|--":
            useful_features = self.useful_features
        self.get_feat_no()  # --updated self.current_feature_no to the latest feat no
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_no = self.current_feature_no + 1
        # ------------------------------------------
        # From https://www.kaggle.com/ambrosm/tpsmay22-eda-which-makes-sense
        new_features = []
        for i in range(10):
            new_features.append(f'ch{i}')
        new_features.append(f'unique_characters')
        # -------------------------------------------------
        self.isRepetition(
            new_features, useful_features, feat_title
        )  # check for duplicate process
        # -------------------------------------------------
        for df in [self.test, self.my_folds]:
            for i in range(10):
                df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
            df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))

        # -----------------------------dump data
        self.my_folds.to_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv", index=False
        )
        self.test.to_csv(f"../configs/configs-{self.locker['comp_name']}/test.csv", index=False)

        # -----------------------------dump current dict
        self.current_feature_no = feat_no
        self.current_dict["current_level"] = self.level_no
        self.current_dict["current_feature_no"] = self.current_feature_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl", self.current_dict
        )
        # -----------------------------dump feature dictionary
        feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_dict[f"l_{self.level_no}_f_{feat_no}"] = [
            new_features,
            useful_features,
            feat_title,
        ]
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl", feat_dict
        )
        print("New features create:- ")
        print(new_features)

    def create_interaction_features(self, useful_features="--|--"):
        feat_title = "interaction_features"

        self.my_folds = pd.read_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv"
        )
        self.test = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/test.csv")
        if useful_features == "--|--":
            useful_features = self.useful_features
        self.get_feat_no()  # --updated self.current_feature_no to the latest feat no
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_no = self.current_feature_no + 1
        # ------------------------------------------
        # From https://www.kaggle.com/ambrosm/tpsmay22-eda-which-makes-sense
        new_features = ['i_02_21', 'i_05_22', 'i_00_01_26']
        # -------------------------------------------------
        self.isRepetition(
            new_features, useful_features, feat_title
        )  # check for duplicate process
        # -------------------------------------------------
        for df in [self.test, self.my_folds]:
            df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
            df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
            i_00_01_26 = df.f_00 + df.f_01 + df.f_26
            df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)

        # -----------------------------dump data
        self.my_folds.to_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv", index=False
        )
        self.test.to_csv(f"../configs/configs-{self.locker['comp_name']}/test.csv", index=False)

        # -----------------------------dump current dict
        self.current_feature_no = feat_no
        self.current_dict["current_level"] = self.level_no
        self.current_dict["current_feature_no"] = self.current_feature_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl", self.current_dict
        )
        # -----------------------------dump feature dictionary
        feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_dict[f"l_{self.level_no}_f_{feat_no}"] = [
            new_features,
            useful_features,
            feat_title,
        ]
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl", feat_dict
        )
        print("New features create:- ")
        print(new_features)

if __name__ == "__main__":
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    with open(f"../configs/configs-{comp_name}/locker.pkl", "rb") as f:
        a = pickle.load(f)
    # ----------------------------------------------------------
    # -----------------------------------------------------------
    ft = features()
    #ft.create_statistical_features(["Age", "SibSp", "Parch"])  # ------------
    #ft.create_unique_characters()
    #ft.create_interaction_features()

    ft.display_features_generated()
    print("===================")
    #ft.show_variables()
