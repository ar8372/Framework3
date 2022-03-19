import pandas as pd
from sklearn import model_selection
import os
import sys
import pickle
from collections import defaultdict


class features:
    def __init__(self):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        a = self.load_pickle(f"../models_{comp_name}/locker.pkl")
        # --------------------------------------
        self.locker = a
        # -------------------------------------
        self.level_no = None
        self.current_dict = None
        self.get_feat_no()  # load level_no and current_feature_no
        self.useful_features = self.load_pickle(
            f"../models_{self.locker['comp_name']}/useful_features_l{self.level_no}.pkl"
        )
        self.my_folds = pd.read_csv(
            f"../models_{self.locker['comp_name']}/my_folds.csv"
        )
        self.test = pd.read_csv(f"../models_{self.locker['comp_name']}/test.csv")

    def change_level(self, new_val="--|--"):
        if new_val != "--|--":
            self.level_no = new_val
        else:
            self.level_no += 1
        self.current_dict["current_level"] = self.level_no
        with open(f"../models_{self.locker['comp_name']}/current_dict.pkl", "wb") as f:
            pickle.dump(self.current_dict, f)

    def display_features_generated(self):
        # display all the feature engineering done so far
        # Key:- f"l{self.level_no}_f{feat_no}"
        # value:- [created, from , info]
        self.feat_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/features_dict.pkl"
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
        for i,(k,v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>",v)
        print()

    def get_feat_no(self):
        # exp_no, current_level, current_feature_no
        self.current_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/current_dict.pkl"
        )
        self.level_no = int(self.current_dict["current_level"])
        self.current_feature_no = int(self.current_dict["current_feature_no"])

    def save_pickle(self, path, to_dump):
        with open(path, "wb") as f:
            pickle.dump(to_dump, f)

    def load_pickle(self, path):
        with open(path, "rb") as f:
            o = pickle.load(f)
        return o

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
            f"../models_{self.locker['comp_name']}/my_folds.csv"
        )
        self.test = pd.read_csv(f"../models_{self.locker['comp_name']}/test.csv")
        if useful_features == "--|--":
            useful_features = self.useful_features
        self.get_feat_no()  # --updated self.current_feature_no to the latest feat no
        self.feat_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_no = self.current_feature_no + 1
        # ------------------------------------------
        new_features = [
            f"l{self.level_no}_f{feat_no}_nan_count",
            f"l{self.level_no}_f{feat_no}_abs_sum",
            f"l{self.level_no}_f{feat_no}_sem",
            f"l{self.level_no}_f{feat_no}_std",
            f"l{self.level_no}_f{feat_no}_mad",
            f"l{self.level_no}_f{feat_no}_avg",
            f"l{self.level_no}_f{feat_no}_median",
            f"l{self.level_no}_f{feat_no}_max",
            f"l{self.level_no}_f{feat_no}_min",
            f"l{self.level_no}_f{feat_no}_skew",
            f"l{self.level_no}_f{feat_no}_num_missing_std",
        ]
        # -------------------------------------------------
        self.isRepetition(
            new_features, useful_features, feat_title
        )  # check for duplicate process
        # -------------------------------------------------
        self.test[f"l{self.level_no}_f{feat_no}_nan_count"] = self.test.isnull().sum(
            axis=1
        )
        self.my_folds[
            f"l{self.level_no}_f{feat_no}_nan_count"
        ] = self.my_folds.isnull().sum(axis=1)

        # self.my_folds[f'l{self.level_no}_f{feat_no}_n_missing'] = my_folds[useful_features].isna().sum(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_abs_sum"] = (
            self.my_folds[useful_features].abs().sum(axis=1)
        )
        self.my_folds[f"l{self.level_no}_f{feat_no}_sem"] = self.my_folds[
            useful_features
        ].sem(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_std"] = self.my_folds[
            useful_features
        ].std(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_mad"] = self.my_folds[
            useful_features
        ].mad(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_avg"] = self.my_folds[
            useful_features
        ].mean(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_median"] = self.my_folds[
            useful_features
        ].median(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_max"] = self.my_folds[
            useful_features
        ].max(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_min"] = self.my_folds[
            useful_features
        ].min(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_skew"] = self.my_folds[
            useful_features
        ].skew(axis=1)
        self.my_folds[f"l{self.level_no}_f{feat_no}_num_missing_std"] = (
            self.my_folds[useful_features].isna().std(axis=1).astype("float")
        )

        # test[f'l{self.level_no}_f{feat_no}_n_missing'] = test[useful_features].isna().sum(axis=1)
        self.test[f"l{self.level_no}_f{feat_no}_abs_sum"] = (
            self.test[useful_features].abs().sum(axis=1)
        )
        self.test[f"l{self.level_no}_f{feat_no}_sem"] = self.test[useful_features].sem(
            axis=1
        )
        self.test[f"l{self.level_no}_f{feat_no}_std"] = self.test[useful_features].std(
            axis=1
        )
        self.test[f"l{self.level_no}_f{feat_no}_mad"] = self.test[useful_features].mad(
            axis=1
        )
        self.test[f"l{self.level_no}_f{feat_no}_avg"] = self.test[useful_features].mean(
            axis=1
        )
        self.test[f"l{self.level_no}_f{feat_no}_median"] = self.test[
            useful_features
        ].median(axis=1)
        self.test[f"l{self.level_no}_f{feat_no}_max"] = self.test[useful_features].max(
            axis=1
        )
        self.test[f"l{self.level_no}_f{feat_no}_min"] = self.test[useful_features].min(
            axis=1
        )
        self.test[f"l{self.level_no}_f{feat_no}_skew"] = self.test[
            useful_features
        ].skew(axis=1)
        self.test[f"l{self.level_no}_f{feat_no}_num_missing_std"] = (
            self.test[useful_features].isna().std(axis=1).astype("float")
        )

        # -----------------------------dump data
        self.my_folds.to_csv(
            f"../models_{self.locker['comp_name']}/my_folds.csv", index=False
        )
        self.test.to_csv(f"../models_{self.locker['comp_name']}/test.csv", index=False)

        # -----------------------------dump current dict
        self.current_feature_no = feat_no
        self.current_dict["current_level"] = self.level_no
        self.current_dict["current_feature_no"] = self.current_feature_no
        self.save_pickle(
            f"../models_{self.locker['comp_name']}/current_dict.pkl", self.current_dict
        )
        # -----------------------------dump feature dictionary
        feat_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_dict[f"l{self.level_no}_f{feat_no}"] = [
            new_features,
            useful_features,
            feat_title,
        ]
        self.save_pickle(
            f"../models_{self.locker['comp_name']}/features_dict.pkl", feat_dict
        )
        print("New features create:- ")
        print(new_features)


if __name__ == "__main__":
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
        a = pickle.load(f)
    # ----------------------------------------------------------
    # -----------------------------------------------------------
    ft = features()
    #ft.create_statistical_features(["Age", "SibSp", "Parch"])  # ------------
    ft.display_features_generated()
    print("===================")
    ft.show_variables()
