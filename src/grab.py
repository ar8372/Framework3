import os
import pandas as pd
import pickle
import sys


class Storage:
    def __init__(self):
        # read all stored files:
        # ----------------------------Keys and store it in [locker]
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
            self.locker = pickle.load(f)
        # ----------------------------current dict
        self.current_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/current_dict.pkl"
        )
        # -----------------------------features dict
        self.features_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/features_dict.pkl"
        )
        # ----------------------------base features
        self.useful_features_l1 = self.load_pickle(
            f"../models_{self.locker['comp_name']}/useful_features_l1.pkl"
        )
        # -----------------------------Table
        self.Table = self.load_pickle(f"../models_{self.locker['comp_name']}/Table.pkl")
        # ------------------------------my folds
        self.my_folds = pd.read_csv(
            f"../models_{self.locker['comp_name']}/my_folds.csv"
        )
        # ------------------------------ test
        self.test = pd.read_csv(f"../models_{self.locker['comp_name']}/test.csv")
        # ---------------------------------container
        self.names = [
            "locker",
            "current_dict",
            "features_dict",
            "useful_features_l1",
            "Table",
            "my_folds",
            "test",
        ]
        self.obj = [
            self.locker,
            self.current_dict,
            self.features_dict,
            self.useful_features_l1,
            self.Table,
            self.my_folds,
            self.test,
        ]

    def show(self, list_keys):
        for k in list_keys:
            if int(k) < len(self.names) and int(k) >= 0:
                # valid
                if self.names[k] == "Table":
                    print(f"{k}. {self.names[k]} :=======>")
                    print(self.obj[k])
                else:
                    print(f"{k}. {self.names[k]} :=======>", self.obj[k])
                print()
            else:
                print(f"{k} is not a valid key!")

    def get(self, key_no):
        if int(key_no) < len(self.names) and int(key_no) >= 0:
            # valid
            return self.obj[key_no]
        else:
            raise Exception(f"{key_no} is not a valid key!")

    def help(self):
        print()
        for i, n in enumerate(self.names):
            print(f"{i} :=======>", n)
        print()

    def save_pickle(self, path, to_dump):
        with open(path, "wb") as f:
            pickle.dump(to_dump, f)

    def load_pickle(self, path):
        with open(path, "rb") as f:
            o = pickle.load(f)
        return o


if __name__ == "__main__":
    s = Storage()
    s.help()
    s.show([0, 1, 2, 3, 4])
