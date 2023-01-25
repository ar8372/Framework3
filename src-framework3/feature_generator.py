import pandas as pd
import numpy as np 
from scipy.stats import skew
from scipy.stats import median_abs_deviation
#from statsmodels import robust
from sklearn import model_selection
import os
import sys
import pickle
from collections import defaultdict
from utils import *
#from auto_exp import *

"""
generates new features on top of some existing featrues. 
and stores this info(title: [column_names_generated, columns_name_used_to_generate]) as a dictionary.
# This dictionary is used only to diplay it is not used to get access of features> 
# Features accessed by their name initials: pred_.. feat_...
"""


class features:
    def __init__(self):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.comp_name = comp_name 
        self.locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
        # -------------------------------------
        self.test_feat_path = f"../configs/configs-{comp_name}/test_feats/"
        self.train_feat_path = f"../configs/configs-{comp_name}/train_feats/"

        self.level_no = None
        self.current_dict = None
        #--------------------------------------------------
        self.get_feat_no()  # load level_no and current_feature_no from DISC
        # self.level_no and self.current_feature_no is update along with selfcurrent_dict
        #----------------------------------------------------
        self.useful_features = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/useful_features_l_{self.level_no}.pkl"
        )
        # from patsylearn import PatsyTransformer
        # transformer = PatsyTransformer("y ~ a + b + a^2 + b^2")
        # transformer.fit(data)
        


    def change_level(self, new_val="--|--"):
        if new_val != "--|--":
            self.level_no = new_val
        else:
            self.level_no += 1
        self.current_dict["current_level"] = self.level_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl",
            self.current_dict,
        )

    def display_features_generated(self):
        # display all the feature engineering done so far
        # Key:- f"l{self.level_no}_f{feat_no}"
        # value:- [created, from , info]
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        for key, value in self.feat_dict.items():
            print(f"Title: {key}")
            print("features created:")
            print(value[0])
            # print("from:")
            # print(value[1])
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
        # f"create_statistical_features_l_{self.level_no}_f_{self.feat_no}"
        # Standard naming convention
        # f"feat_l_{self.level_no}_f_{self.feat_no}_csf
            # feat_dict[f"exp_{self.exp_no}"] = [
            #     [f"pred_e_{self.exp_no}_{self.fold_name}"],
            #     self.useful_features
            # ]
        # feat_l_2_f_23_std 
        # self.curr
        for key, value in self.feat_dict.items():
            if key.split("_")[0]== "feat":
                # it is a feature entry
                # feat_l_{self.level_no}_f_{feat_no}_nan_count
                # old_features same and title same
                if set(value[0]) == set(old_features) and key.split("_")[-1] == feat_title:
                    raise Exception(f"This set of feature is already there. details key name: {key}") 

            
        # for key, value in self.feat_dict.items():
        #     f1, f2, ft = value
        #     if f2 == 0:
        #         # from base
        #         pass
        #     elif len(f1[0].split("_")[0]) < 5 or (
        #         f1[0].split("_")[0][0] == "l" and f1[0].split("_")[0][2] == "f"
        #     ):
        #         # originate from base so f2 can't be split
        #         f1 = ["_".join(f.split("_")[2:]) for f in f1]
        #         gen_features = ["_".join(f.split("_")[2:]) for f in gen_features]
        #     else:
        #         f2 = ["_".join(f.split("_")[2:]) for f in f2]
        #         old_features = ["_".join(f.split("_")[2:]) for f in old_features]
        #         f1 = ["_".join(f.split("_")[2:]) for f in f1]
        #         gen_features = ["_".join(f.split("_")[2:]) for f in gen_features]
        #     if f1 == gen_features and f2 == old_features and ft == feat_title:
        #         raise Exception("This feature is already present!")

    def create_statistical_features(self, useful_features="--|--"):
        fill_na_with  = -100

        if useful_features == "--|--":
            useful_features = self.useful_features
        else:
            self.useful_features = useful_features

        # Get train, test from bottleneck: 
        #------------------------------------------------------------------------
        # BOTTLENECK 
        return_type = "numpy_array"
        self.optimize_on = None # just to make sure it is not called 
        fold_name = "fold_check"
        #self._state = "seed"
        state = "seed"
        self.val_idx, self.my_folds, self.xvalid, self.ytrain, self.yvalid, ordered_list_test = bottleneck(self.locker['comp_name'],self.useful_features, fold_name, self.optimize_on, state, return_type)
        self.xvalid = None 
        self.yvalid = None 
        self.val_idx = None 
        
        self.test,ordered_list_train = bottleneck_test(self.locker['comp_name'], self.useful_features, return_type)   
        # sanity check: 
        for i,j in zip(ordered_list_test, ordered_list_train):
            if i != j:
                raise Exception(f"Features don't correspond in test - train {i},{j}")
        ordered_list_test = None 
        ordered_list_train = None     
        # self.test, self.my_folds
        #------------------------------------------------------------------------

        self.get_feat_no()  # --updated self.current_feature_no to the latest feat no
        # self.level_no, self.current_feature_no
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        # new set of feature is created so increase feat no
        feat_no = self.current_feature_no + 1
        #feat_no = 10
        feat_title = f"feat_l_{self.level_no}_f_{feat_no}_csf{fill_na_with}" # to mke it unique in the dictionary
        # ------------------------------------------
        new_features = [
            f"feat_l_{self.level_no}_f_{feat_no}_nan_count",
            f"feat_l_{self.level_no}_f_{feat_no}_num_missing_std",
            f"feat_l_{self.level_no}_f_{feat_no}_abs_sum",
            f"feat_l_{self.level_no}_f_{feat_no}_sem",
            f"feat_l_{self.level_no}_f_{feat_no}_std",
            f"feat_l_{self.level_no}_f_{feat_no}_medianad",
            f"feat_l_{self.level_no}_f_{feat_no}_meanad",
            f"feat_l_{self.level_no}_f_{feat_no}_avg",
            f"feat_l_{self.level_no}_f_{feat_no}_median",
            f"feat_l_{self.level_no}_f_{feat_no}_max",
            f"feat_l_{self.level_no}_f_{feat_no}_min",
            f"feat_l_{self.level_no}_f_{feat_no}_skew",
        ]
        # --------------------------Duplicacy check
        self.isRepetition(
            new_features, useful_features, feat_title
        )  # check for duplicate process


        print(self.train_feat_path)
        print(self.test_feat_path)
        print()
        #-------------------------------------------------
        # nan_count
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            val1= np.isnan(self.test).sum(axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_nan_count.pkl", val1)
            val2= np.isnan(self.my_folds).sum(axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_nan_count.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_nan_count")
        #--------------------------------------------------------------
        # num_missing_std 
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            val1= np.isnan(self.test).std(axis=1).astype("float")
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_num_missing_std.pkl", val1)
            val2= np.isnan(self.my_folds).std(axis=1).astype("float")
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_num_missing_std.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_num_missing_std")

        # below all are affected by nan 
        # -------------------------------------------------
        # So first fill nan
        self.my_folds[np.isnan(self.my_folds)] = fill_na_with
        self.test[np.isnan(self.test)] = fill_na_with
        # sanity check 
        assert np.isnan(self.my_folds).sum() ==0 
        assert np.isnan(self.test).sum() ==0 
        # quite different value is added
        #-----------------------------------------------------
        # abs_sum : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            val1= np.abs(self.test).sum(axis=1)
            print(val1.shape, "abs_sum")
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_abs_sum.pkl", val1)
            val2= np.abs(self.my_folds).sum(axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_abs_sum.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_abs_sum")
        # sem : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            val1= np.std(self.test, axis=1)/np.sqrt(self.test.shape[1])
            print(val1.shape, "val1")
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_sem.pkl", val1)
            val2= np.std(self.my_folds, axis=1)/np.sqrt(self.my_folds.shape[1]) 
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_sem.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_sem")
        # std : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            val1= np.std(self.test, axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_std.pkl", val1)
            val2= np.std(self.my_folds, axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_std.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_std")
        # medianad : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            # need to reshpae since medina creates 1D array and thus can't be broadcasted to 2D
            #val1= np.median(np.absolute(self.test - np.median(self.test, axis=1)), axis=1).reshape(-1)
            #val1 = self.test.mad(axis=1) #robust.mad(self.test, axis=1)
            val1 = median_abs_deviation(self.test, axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_medianad.pkl", val1)
            #val2= np.median(np.absolute(self.my_folds - np.median(self.my_folds, axis=1)), axis=1).reshape(-1)
            #val2 = self.my_folds.mad(axis=1) #robust.mad(self.my_folds, axis=1)
            val2 = median_abs_deviation(self.my_folds, axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_medianad.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_medianad")
        # meanad : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            # need to reshpae since medina creates 1D array and thus can't be broadcasted to 2D
            #val1= np.median(np.absolute(self.test - np.median(self.test, axis=1)), axis=1).reshape(-1)
            val1 = np.mean(np.abs(self.test - np.mean(self.test, axis=1).reshape(-1,1)), axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_meanad.pkl", val1)
            #val2= np.median(np.absolute(self.my_folds - np.median(self.my_folds, axis=1)), axis=1).reshape(-1)
            val2 = np.mean(np.abs(self.my_folds - np.mean(self.my_folds, axis=1).reshape(-1,1)), axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_meanad.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_meanad")
        # avg : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            # need to reshpae since medina creates 1D array and thus can't be broadcasted to 2D
            val1= np.mean(self.test, axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_avg.pkl", val1)
            val2= np.mean(self.my_folds, axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_avg.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_avg")
        # median : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            # need to reshpae since medina creates 1D array and thus can't be broadcasted to 2D
            val1= np.median(self.test, axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_median.pkl", val1)
            val2= np.median(self.my_folds, axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_median.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_median")
        # max : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            # need to reshpae since medina creates 1D array and thus can't be broadcasted to 2D
            val1= np.max(self.test, axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_max.pkl", val1)
            val2= np.max(self.my_folds, axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_max.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_max")
        # min : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            # need to reshpae since medina creates 1D array and thus can't be broadcasted to 2D
            val1= np.min(self.test, axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_min.pkl", val1)
            val2= np.min(self.my_folds, axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_min.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_min")
        # skew : nan affected
        try:
            # for dataframe: self.test.isnull().sum(axis=1)
            # need to reshpae since medina creates 1D array and thus can't be broadcasted to 2D
            val1= np.min(self.test, axis=1)
            save_pickle(self.test_feat_path+ f"test_feat_l_{self.level_no}_f_{feat_no}_skew.pkl", val1)
            val2= np.min(self.my_folds, axis=1)
            save_pickle(self.train_feat_path+ f"train_feat_l_{self.level_no}_f_{feat_no}_skew.pkl", val2)
            print(val2)
            print(val1)
            print()
        except:
            raise Exception(f"Couldn't create feat_l_{self.level_no}_f_{feat_no}_skew")
            
        del val1, val2 
        gc.collect()

        print(new_features)
        print()
        v = input("Are you sure you want to add these features?: [y/n]")
        if v.lower() == 'n':
            raise Exception("Process terminated.")

 

        # -----------------------------dump current dict
        self.current_feature_no = feat_no
        self.current_dict["current_level"] = self.level_no
        self.current_dict["current_feature_no"] = self.current_feature_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl",
            self.current_dict,
        )
        # -----------------------------dump feature dictionary
        feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_dict[feat_title] = [new_features, useful_features]
        # feat_dict[f"l_{self.level_no}_f_{feat_no}"] = [
        #     new_features,
        #     useful_features,
        #     feat_title,
        # ]
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl",
            feat_dict,
        )
        print("New features created:- ")
        print(new_features)

    def create_unique_characters(self, useful_features="--|--"):
        raise Exception("Don't enter")
        feat_title = "unique_characters"

        self.my_folds = pd.read_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv"
        )
        self.test = pd.read_csv(
            f"../configs/configs-{self.locker['comp_name']}/test.csv"
        )
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
            new_features.append(f"ch{i}")
        new_features.append(f"unique_characters")
        # -------------------------------------------------
        self.isRepetition(
            new_features, useful_features, feat_title
        )  # check for duplicate process
        # -------------------------------------------------
        for df in [self.test, self.my_folds]:
            for i in range(10):
                df[f"ch{i}"] = df.f_27.str.get(i).apply(ord) - ord("A")
            df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))

        # -----------------------------dump data
        self.my_folds.to_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv", index=False
        )
        self.test.to_csv(
            f"../configs/configs-{self.locker['comp_name']}/test.csv", index=False
        )

        # -----------------------------dump current dict
        self.current_feature_no = feat_no
        self.current_dict["current_level"] = self.level_no
        self.current_dict["current_feature_no"] = self.current_feature_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl",
            self.current_dict,
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
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl",
            feat_dict,
        )
        print("New features create:- ")
        print(new_features)

    def create_interaction_features(self, useful_features="--|--"):
        raise Exception("Don't enter")
        feat_title = "interaction_features"

        self.my_folds = pd.read_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv"
        )
        self.test = pd.read_csv(
            f"../configs/configs-{self.locker['comp_name']}/test.csv"
        )
        if useful_features == "--|--":
            useful_features = self.useful_features
        else:
            self.useful_features = useful_features 

        self.get_feat_no()  # --updated self.current_feature_no to the latest feat no
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_no = self.current_feature_no + 1
        # ------------------------------------------
        # From https://www.kaggle.com/ambrosm/tpsmay22-eda-which-makes-sense
        new_features = ["i_02_21", "i_05_22", "i_00_01_26"]
        # -------------------------------------------------
        self.isRepetition(
            new_features, useful_features, feat_title
        )  # check for duplicate process
        # -------------------------------------------------
        for df in [self.test, self.my_folds]:
            df["i_02_21"] = (df.f_21 + df.f_02 > 5.2).astype(int) - (
                df.f_21 + df.f_02 < -5.3
            ).astype(int)
            df["i_05_22"] = (df.f_22 + df.f_05 > 5.1).astype(int) - (
                df.f_22 + df.f_05 < -5.4
            ).astype(int)
            i_00_01_26 = df.f_00 + df.f_01 + df.f_26
            df["i_00_01_26"] = (i_00_01_26 > 5.0).astype(int) - (
                i_00_01_26 < -5.0
            ).astype(int)

        # -----------------------------dump data
        self.my_folds.to_csv(
            f"../configs/configs-{self.locker['comp_name']}/my_folds.csv", index=False
        )
        self.test.to_csv(
            f"../configs/configs-{self.locker['comp_name']}/test.csv", index=False
        )

        # -----------------------------dump current dict
        self.current_feature_no = feat_no
        self.current_dict["current_level"] = self.level_no
        self.current_dict["current_feature_no"] = self.current_feature_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl",
            self.current_dict,
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
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl",
            feat_dict,
        )
        print("New features create:- ")
        print(new_features)

    def create_polynomial_features(self,title, useful_features="--|--"):

        if useful_features == "--|--":
            useful_features = self.useful_features
        else:
            self.useful_features = useful_features

        # Get train, test from bottleneck: 
        #------------------------------------------------------------------------
        # BOTTLENECK 
        return_type = "numpy_array"
        self.optimize_on = None # just to make sure it is not called 
        fold_name = "fold_check"
        #self._state = "seed"
        state = "seed"
        self.val_idx, self.xtrain, self.xvalid, self.ytrain, self.yvalid, ordered_list_train = bottleneck(self.locker['comp_name'],self.useful_features, fold_name, self.optimize_on, state, return_type)
        self.xvalid = None 
        self.yvalid = None 
        self.val_idx = None 
        print(self.xtrain.shape)
        self.xtrain = pd.DataFrame(self.xtrain, columns = useful_features)
        print(self.xtrain.iloc[:10,:5])
        self.test,ordered_list_test = bottleneck_test(self.locker['comp_name'], self.useful_features, return_type) 
        print(self.test.shape) 
        self.test = pd.DataFrame(self.test, columns = useful_features)
        # sanity check: 
        for i,j in zip(ordered_list_test, ordered_list_train):
            if i != j:
                raise Exception(f"Features don't correspond in test - train {i},{j}")
        useful_features = ordered_list_test # just to make sure order
        ordered_list_test = None 
        ordered_list_train = None     
        # self.test, self.my_folds
        #------------------------------------------------------------------------
        # This updated input folder 
        #----------------------------------------------
        # places where feature are updated 
        #1> useful_features_l_1 
        #2> feature_dict (here base is never used so can skip)
        #3> input_dict 
        self.input_dict = load_pickle(
            f"../input/input-{self.comp_name}/input_dict.pkl"
        )
        print("input dict before")
        print(self.input_dict.keys())
        useful_features_l_1 = load_pickle(f"../configs/configs-{self.comp_name}/useful_features_l_1.pkl")
        print("Total features before", len(useful_features_l_1))
        # ------------------------------------------
        self.xtrain = self.xtrain.values 
        self.test = self.test.values 
        train_dummy = np.array([], dtype=np.int8).reshape(self.xtrain.shape[0],0)
        test_dummy = np.array([], dtype=np.int8).reshape(self.test.shape[0],0)
        no_features = len(useful_features)
        generated_features = []
        for i in range(no_features):
            f = useful_features[i]
            generated_features += [f"{f}*{useful_features[i]}" for i in range(no_features)]
            train_dummy = np.concatenate((train_dummy, self.xtrain* self.xtrain[:,i].reshape(-1,1)), axis=1)
            test_dummy = np.concatenate((test_dummy, self.test* self.test[:,i].reshape(-1,1)), axis=1)
        print("After")
        print(train_dummy.shape)
        print(test_dummy.shape)
        print(len(generated_features))
        print(generated_features[:6])
        # poly  = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        # t= poly.fit_transform(self.xtrain)
        # print(t.shape)
        # print(t.iloc[:10,:5])
        #
        # PUSH FEATURES 

        self.input_dict[f"{title}_interact"] = generated_features
        
        print("input dict after ")
        print(self.input_dict.keys())
        useful_features_l_1 += generated_features 
        # sanity check remove duplicate if called twice 
        #assert len(useful_features_l_1) == len(list(set(useful_features_l_1)))
        useful_features_l_1 = list(set(useful_features_l_1))
        print("Total features after", len(useful_features_l_1))
        print(useful_features_l_1[:4])
        v = input("Do you want to create these features? y/Y or else: ")
        if v.lower() == "y":
            self.xtrain = pd.DataFrame(train_dummy , columns = generated_features)
            self.test = pd.DataFrame(test_dummy , columns = generated_features)
            self.xtrain.to_parquet(f"../input/input-{self.comp_name}/train_{title}_interact.parquet")
            self.test.to_parquet(f"../input/input-{self.comp_name}/test_{title}_interact.parquet")
            save_pickle(f"../input/input-{self.comp_name}/input_dict.pkl", self.input_dict)
            save_pickle(f"../configs/configs-{self.comp_name}/useful_features_l_1.pkl",useful_features_l_1)
            print("Updated!")

            print()
            print(f"Title: {title}_interact")
            print()
            print(generated_features)
        else:
            print("Aborted!")
    

    def pull_input(self, source_comp_name, source_feat_name):
        """
        source_comp_name = "amex4"
        source_feat_name = "last_mean_diff"
        """
        train_dummy = pd.read_parquet(f"../input/input-{source_comp_name}/train_{source_feat_name}.parquet")
        test_dummy = pd.read_parquet(f"../input/input-{source_comp_name}/test_{source_feat_name}.parquet")

        generated_features = list(train_dummy.columns)
        #----------------------------------------------
        # places where feature are updated 
        #1> useful_features_l_1 
        #2> feature_dict (here base is never used so can skip)
        #3> input_dict 
        self.input_dict = load_pickle(
            f"../input/input-{self.comp_name}/input_dict.pkl"
        )
        print("input dict before")
        print(self.input_dict.keys())
        useful_features_l_1 = load_pickle(f"../configs/configs-{self.comp_name}/useful_features_l_1.pkl")
        print("Total features before", len(useful_features_l_1))
        print("Total Generated features", len(generated_features))
        # ------------------------------------------
        # PUSH FEATURES 

        self.input_dict[source_feat_name] = generated_features
        print("input dict after ")
        print(self.input_dict.keys())
        useful_features_l_1 += generated_features 
        # sanity check remove duplicate if called twice 
        assert len(useful_features_l_1) == len(list(set(useful_features_l_1)))
        useful_features_l_1 = list(set(useful_features_l_1))
        print("Total features after", len(useful_features_l_1))
        print(useful_features_l_1[:4])
        v = input("Do you want to create these features? y/Y or else: ")
        if v.lower() == "y":
            self.xtrain = pd.DataFrame(train_dummy , columns = generated_features)
            self.test = pd.DataFrame(test_dummy , columns = generated_features)
            self.xtrain.to_parquet(f"../input/input-{self.comp_name}/train_{source_feat_name}.parquet")
            self.test.to_parquet(f"../input/input-{self.comp_name}/test_{source_feat_name}.parquet")
            save_pickle(f"../input/input-{self.comp_name}/input_dict.pkl", self.input_dict)
            save_pickle(f"../configs/configs-{self.comp_name}/useful_features_l_1.pkl",useful_features_l_1)
            print("Updated!")

            print()
            print(generated_features)
        else:
            print("Aborted!")

from settings import *          
       

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

    ## Statistical features
    useful_features = amzcomp1_settings.feature_dict['ver2']
    #ft.create_statistical_features( useful_features)  # ------------
    # ft.create_unique_characters()
    # ft.create_interaction_features()

    # Interaction/polynomial features
    #useful_features = amzcomp1_settings().filtered_features['filter6']
    title = 'ver2'
    print("Feature to interact")
    print(useful_features)
    #input()
    ft.create_polynomial_features(title, useful_features)


    #ft.create_statistical_features(useful_features)
    #ft.display_features_generated()

    # title = '--'
    # amex = amex4_settings()
    # useful_features = amex.feature_dict2[title]
    # print(useful_features)
    # ft.create_polynomial_features(title,useful_features)


    # source_comp_name = "amexdummy"
    # source_feat_name = 'date'
    # ft.pull_input(source_comp_name=source_comp_name, source_feat_name=source_feat_name)
    # print("===================")
    # # ft.show_variables()


    # useful_features = getaroom_settings().feature_dict['base']
    # title  = "base"
    # print(useful_features)
    # ft.create_polynomial_features(title,useful_features)
