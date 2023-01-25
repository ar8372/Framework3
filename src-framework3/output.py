import os
import sys
from utils import *
import pandas as pd
import numpy as np


class out:
    def __init__(self,  fold_name, exp_no=-1,file_type="parquet"):
        self.exp_no = exp_no
        self.file_type = file_type
        # initialize rest
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                self.comp_name = i
        x.close()
        self.Table = load_pickle(f"../configs/configs-{self.comp_name}/Table.pkl")
        self.locker = load_pickle(f"../configs/configs-{self.comp_name}/locker.pkl")

        assert fold_name != "" 
        self.fold_name = fold_name 

        # self.my_folds = pd.read_csv(f"../configs/configs-{self.comp_name}/my_folds.csv")
        self.my_folds = pd.read_parquet(
            f"../input/input-{self.comp_name}/my_folds.parquet"
        )
        # self.test = pd.read_csv(f"../configs/configs-{self.comp_name}/test.csv")
        self.test = pd.read_parquet(f"../input/input-{self.comp_name}/test.parquet")
        # keep sample to input
        # self.sample = pd.read_csv(f"../input/input-{self.comp_name}/sample.csv")
        self.sample = pd.read_parquet(f"../input/input-{self.comp_name}/sample.parquet")

    def dump(self, exp_no="--|--", file_type="--|--"):
        if exp_no != "--|--":
            self.exp_no = exp_no
        if file_type != "--|--":
            self.file_type = file_type

        if not self.fold_name.startswith("e"):
            if self.exp_no == -1:
                row_e = self.Table[self.Table.exp_no == list(self.Table.exp_no.values)[-1]]
                self.exp_no = row_e.exp_no.values[0]
            else:
                row_e = self.Table[self.Table.exp_no == self.exp_no]
                assert self.exp_no == row_e.exp_no.values[0]  # --> assert
            self.bv = row_e.bv.values[0]  # confirming we are predcting correct experiment:
            print(f"Output of Exp No {self.exp_no}, whoose bv is {self.bv}")
            self.exp_no = row_e.exp_no.values[0]
            self.level_no = row_e.level_no.values[0]
            # found the row
            # BOTTLENECK 
            # which oof_fold_name to pull


            if self.fold_name not in list(row_e.oof_fold_name.values[0]):
                raise Exception(f"This experiment has no prediction of type {self.fold_name}!!")

            self.useful_features = [f"pred_e_{self.exp_no}_{self.fold_name}"]
            return_type = "numpy_array"
            self.optimize_on = None # just to make sure it is not called 
            xtest, ordered_list_test = bottleneck_test(self.comp_name, self.useful_features,  return_type)
            #val_idx, xtrain, xvalid, ytrain, yvalid, xtest = bottleneck_test(self.locker['comp_name'],self.useful_features, fold_name, self.optimize_on, self._state, return_type)
            #xtest.shape : (234522,1)
            #xtrain.shape: (224224,1)
            print(xtest.shape)
            print(xtest[:4,:])

            try:
                # fold
                if self.locker["comp_name"] == "twistmnist":  # special case
                    self.sample[self.locker["target_name"]] = (
                        self.test[f"pred_l_{self.level_no}_e_{self.exp_no}"].values.astype(
                            int
                        )
                        + 10
                    )
                else:
                    # self.sample[self.locker["target_name"]] = self.test[
                    #     f"pred_l_{self.level_no}_e_{self.exp_no}"
                    # ].values
                    self.sample[self.locker["target_name"]] = xtest
                # else: # use it when want hard class
                #     self.sample[self.locker["target_name"]] = self.test[f"pred_l_{self.level_no}_e_{self.exp_no}"].values.astype(int)
                print("Done here")
                if self.comp_name == 'amzcomp1':
                    self.sample = self.sample.rename(columns= {self.locker['target_name']: 'Time_taken (min)'})
                if self.file_type == "parquet":
                    self.sample.to_parquet(
                        f"../working/{self.locker['comp_name']}_sub_e_{int(self.exp_no)}_{self.fold_name}.parquet", index=False
                    )
                else:
                    self.sample.to_csv(
                        f"../working/{self.locker['comp_name']}_sub_e_{int(self.exp_no)}_{self.fold_name}.csv", index=False
                    )

                #################### Below files are already dumped in the WORKING directory
                # # seed all
                # #d1 = pd.read_csv(f"../configs/configs-{self.comp_name}/sub_seed_exp_{self.exp_no}_l_{self.level_no}_all.csv")
                # d1 = pd.read_parquet(f"../configs/configs-{self.comp_name}/sub_seed_exp_{self.exp_no}_l_{self.level_no}_all.parquet")
                # if self.locker["comp_name"] == "twistmnist":
                #     d1[self.locker["target_name"]] = d1[self.locker["target_name"]] + 10
                # else:
                #     d1[self.locker["target_name"]] = d1[self.locker["target_name"]]
                # if self.file_type == "parquet":
                #     d1.to_parquet(f"../working/sub_seed_exp_{self.exp_no}_l_{self.level_no}_all.parquet", index=False)
                # else:
                #     d1.to_csv(f"../working/sub_seed_exp_{self.exp_no}_l_{self.level_no}_all.csv", index=False)

                # # seed single
                # #d1 = pd.read_csv(f"../configs/configs-{self.comp_name}/sub_seed_exp_{self.exp_no}_l_{self.level_no}_single.csv")
                # d1 = pd.read_parquet(f"../configs/configs-{self.comp_name}/sub_seed_exp_{self.exp_no}_l_{self.level_no}_single.parquet")
                # if self.locker["comp_name"] == "twistmnist":
                #     d1[self.locker["target_name"]] = d1[self.locker["target_name"]] + 10 # twistmnist
                # else:
                #     d1[self.locker["target_name"]] = d1[self.locker["target_name"]]

                # if self.file_type == "parquet":
                #     d1.to_parquet(f"../working/sub_seed_exp_{self.exp_no}_l_{self.level_no}_single.parquet", index=False)
                # else:
                #     d1.to_csv(f"../working/sub_seed_exp_{self.exp_no}_l_{self.level_no}_single.csv", index=False)
                ##############################################################################################

                print()
                print(self.sample.head(2))
            except:
                raise Exception(f"exp_no {self.exp_no} not found with fold name : {self.fold_name}!")

        else:
            # starts with e so ensemble 
            self.useful_features = [f"{self.comp_name}_ens_{self.exp_no}"]
            # this file should be in working directory
            val = pd.read_parquet(f"../working/{self.useful_features[0]}.parquet")
            self.sample[self.locker["target_name"]] = val[self.locker["target_name"]]
            if self.comp_name == 'amzcomp1':
                self.sample = self.sample.rename(columns= {self.locker['target_name']: 'Time_taken (min)'})
            try:
                if self.file_type == "parquet":
                    self.sample.to_parquet(
                        f"../working/{self.useful_features[0]}.parquet", index=False
                    )
                else:
                    self.sample.to_csv(
                        f"../working/{self.useful_features[0]}.csv", index=False
                    )
            except:
                raise Exception("ensemble not created!!")

if __name__ == "__main__":
    # creates fold submission from a predicted experiment
    """
    file_type:
        parquet:- when working on remote ssh so that it is easy to version it
        csv:- when working on notebook so that it is easy to make submission as csv
    """
    for exp_no in [70]:
        # exp_no = (
        #     7
        # )  # -1 for last prediction i.e. last experiment so you must predict last experiment before doing output.py
        
        # need to add which fold prediction to output
        file_type = "csv"  # "csv"
        fold_name = "fold5" # need to pass some fold name can't keep it empty
        o = out( fold_name, exp_no, file_type)
        o.dump()

        # for exp_no in [0,2,5,11,19,43]:

        #     o= out(exp_no, file_type)
        #     o.dump()
