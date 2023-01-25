import os
import pandas as pd
import pickle
import sys

"""
used to show stored variables:
"""


class Storage:
    def __init__(self):
        # read all stored files:
        # ----------------------------Keys and store it in [locker]
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../configs/configs-{comp_name}/locker.pkl", "rb") as f:
            self.locker = pickle.load(f)
        # ----------------------------current dict
        self.current_dict = self.load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl"
        )
        # -----------------------------features dict
        self.features_dict = self.load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        # ----------------------------base features
        self.useful_features_l1 = self.load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/useful_features_l_1.pkl"
        )
        # -----------------------------Table
        self.Table = self.load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/Table.pkl"
        )
        # ------------------------------my folds
        # self.my_folds = pd.read_csv(
        #     f"../configs/configs-{self.locker['comp_name']}/my_folds.csv"
        # )
        self.my_folds = pd.read_parquet(
            f"../input/input-{self.locker['comp_name']}/my_folds.parquet"
        )
        # ------------------------------ test
        self.test = None
        if self.locker["data_type"] == "tabular":
            # self.test = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/test.csv")
            self.test = pd.read_parquet(
                f"../input/input-{self.locker['comp_name']}/test.parquet"
            )
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
                elif k == 2:
                    # asked for feture dict
                    # for f1,f2,ft in self.obj[k]:
                    #     print(ft)
                    for i,(l,val) in enumerate(self.features_dict.items()):
                        print(l)
                        print(val[0],"-->",val[1])
                        print()
                        print()
                    #print(self.features_dict)
                    #print(self.features_dict.keys())
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

    def get_log_table(self, exp_no):
        # -------------------------------log table
        log_table = self.load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/log_exp_{exp_no}.pkl"
        )
        return log_table

    def show_log_table(self, exp_no):
        print(self.get_log_table(exp_no))

    def help(self):
        print("functions: show() get() show_log_table() get_log_table()")
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
    s.show([2])
    # s.show([0, 1, 2, 3, 4])
