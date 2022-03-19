import pandas as pd
from sklearn import model_selection
import os
import sys
import pickle
from collections import defaultdict


class Picker:
    def __init__(self):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.locker = self.load_pickle(f"../models_{comp_name}/locker.pkl")
        # ----------------------------------------------------------
        self.list_levels = ["1"]
        self.list_features = ["0"]
        self.list_feat_title = []
        self.feat_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/features_dict.pkl"
        )

    def show_variables(self):
        print()
        for i,(k,v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>",v)
        print()
        
    def save_pickle(self, path, to_dump):
        with open(path, "wb") as f:
            pickle.dump(to_dump, f)

    def load_pickle(self, path):
        with open(path, "rb") as f:
            o = pickle.load(f)
        return o

    def find_keys(
        self, list_levels="--|--", list_features="--|--", list_feat_title="--|--"
    ):
        # -----------------------------dump feature dictionary
        if list_levels != "--|--":
            self.list_levels = list_levels
        if list_features != "--|--":
            self.list_features = list_features
        if list_feat_title != "--|--":
            self.list_feat_title = list_feat_title
        all_keys = list(self.feat_dict.keys())

        valid_keys1 = list(self.feat_dict.keys())
        valid_keys2 = list(self.feat_dict.keys())
        valid_keys3 = list(self.feat_dict.keys())
        if self.list_levels != []:
            valid_keys1 = []
            for key in list(self.feat_dict.keys()):
                if str(key[1]) in str(self.list_levels):
                    valid_keys1.append(key)
        if self.list_features != []:
            valid_keys2 = []
            for key in list(self.feat_dict.keys()):
                if str(key[4]) in str(self.list_features):
                    valid_keys2.append(key)
        if self.list_feat_title != []:
            valid_keys3 = []
            for key, val in list(self.feat_dict.items()):
                if str(val[2]) in str(self.list_feat_title):
                    valid_keys3.append(key)
        valid_keys = set(valid_keys1).intersection(set(valid_keys2))
        valid_keys = list(valid_keys.intersection(set(valid_keys3)))
        return valid_keys

    def find_features(
        self, list_levels="--|--", list_features="--|--", list_feat_title="--|--"
    ):
        # -----------------------------dump feature dictionary
        if list_levels != "--|--":
            self.list_levels = list_levels
        if list_features != "--|--":
            self.list_features = list_features
        if list_feat_title != "--|--":
            self.list_feat_title = list_feat_title
        valid_keys = self.find_keys(
            self.list_levels, self.list_features, self.list_feat_title
        )
        valid_features = []
        for key in valid_keys:
            valid_features += self.feat_dict[key][0]
        return valid_features

    def help(self):
        # display all the feature engineering done so far
        # Key:- f"l{self.level_no}_f{feat_no}"
        # value:- [created, from , info]
        for key, value in self.feat_dict.items():
            print(key, f"{value[-1]} :-")
            print("features created:")
            print(value[0])
            print("from:")
            print(value[1])
            print("=" * 40)


if __name__ == "__main__":
    p = Picker()
    p.list_levels = ["1"]
    p.list_features = ["1", "2", "0"]
    # p.list_feat_title = ["base"]
    print(p.find_keys())
    print()
    # print(p.find_features())
