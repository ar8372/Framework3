import os
import sys 
from utils import *
import pandas as pd 
import numpy as np 

class out:
    def __init__(self, exp_no=-1):
        self.exp_no = exp_no
        # initialize rest
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                self.comp_name = i
        x.close()
        self.Table = load_pickle(f"../configs/configs-{self.comp_name}/Table.pkl")
        self.locker = load_pickle(f"../configs/configs-{self.comp_name}/locker.pkl")
        self.my_folds = pd.read_csv(f"../configs/configs-{self.comp_name}/my_folds.csv")
        self.test = pd.read_csv(f"../configs/configs-{self.comp_name}/test.csv")
        self.sample = pd.read_csv(f"../configs/configs-{self.comp_name}/sample.csv")


    
    def dump(self, exp_no="--|--"):
        if exp_no != "--|--":
            self.exp_no = exp_no 
        if self.exp_no == -1:
            row_e = self.Table[self.Table.exp_no == list(self.Table.exp_no.values)[-1]]
            self.exp_no = row_e.exp_no.values[0]
        else:
            row_e = self.Table[self.Table.exp_no == self.exp_no]
        self.exp_no = row_e.exp_no.values[0]
        self.level_no = row_e.level_no.values[0]
        # found the row 
        
        try:
            # fold
            if self.locker["comp_name"] == "twistmnist": # special case
                self.sample[self.locker["target_name"]] = self.test[f"pred_l_{self.level_no}_e_{self.exp_no}"].values.astype(int) + 10
            else:
                self.sample[self.locker["target_name"]] = self.test[f"pred_l_{self.level_no}_e_{self.exp_no}"].values
            else: # use it when want hard class
                self.sample[self.locker["target_name"]] = self.test[f"pred_l_{self.level_no}_e_{self.exp_no}"].values.astype(int)
            self.sample.to_csv(f"../working/sub_exp_{self.exp_no}_fold.csv", index=False)

            # seed all 
            d1 = pd.read_csv(f"../configs/configs-{self.comp_name}/sub_seed_exp_{self.exp_no}_l_{self.level_no}_all.csv")
            if self.locker["comp_name"] == "twistmnist":
                d1[self.locker["target_name"]] = d1[self.locker["target_name"]] + 10
            else:
                d1[self.locker["target_name"]] = d1[self.locker["target_name"]]
            d1.to_csv(f"../working/sub_seed_exp_{self.exp_no}_l_{self.level_no}_all.csv", index=False)

            # seed single
            d1 = pd.read_csv(f"../configs/configs-{self.comp_name}/sub_seed_exp_{self.exp_no}_l_{self.level_no}_single.csv")
            if self.locker["comp_name"] == "twistmnist":
                d1[self.locker["target_name"]] = d1[self.locker["target_name"]] + 10 # twistmnist
            else:
                d1[self.locker["target_name"]] = d1[self.locker["target_name"]]
            d1.to_csv(f"../working/sub_seed_exp_{self.exp_no}_l_{self.level_no}_single.csv", index=False)

            print(self.sample[self.locker["target_name"]].value_counts())
            print()
            print(self.sample.head(2))
        except:
            raise Exception(f"exp_no {self.exp_no} not found!")

        




if __name__ == "__main__":
    exp_no = -1 
    o= out(-1)
    o.dump()