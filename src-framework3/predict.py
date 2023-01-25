from optuna_search import OptunaOptimizer
from utils import *
from custom_models import *
from custom_classes import *
from utils import *
import os
import gc
import sys
import pandas as pd
import numpy as np
from scipy import stats
import ast  # for literal
#pd.set_option("display.max_columns", None) # for time
"""
Inference file
"""
import global_variables

class predictor(OptunaOptimizer):
    def __init__(self, exp_no):
        self.exp_no = exp_no
        # initialize rest
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                self.comp_name = i
        x.close()
        self.Table = load_pickle(f"../configs/configs-{self.comp_name}/Table.pkl")
        self.locker = load_pickle(f"../configs/configs-{self.comp_name}/locker.pkl")
        
        if self.exp_no == -1:
            row_e = self.Table[self.Table.exp_no == list(self.Table.exp_no.values)[-1]]
            self.exp_no = row_e.exp_no.values[0]
        else:
            row_e = self.Table[self.Table.exp_no == self.exp_no]
        global_variables.exp_no = self.exp_no # setting for making dir in lgb
        self.model_name = row_e.model_name.values[0]
        self.params = row_e.bp.values[0]
        self.bv = row_e.bv.values[0]  # confirming we are predcting correct experiment:
        print(f"Predicting Exp No {self.exp_no}, whoose bv is {self.bv}")
        if self.model_name == "lgr":
            del self.params["c"]
        self._random_state = row_e.random_state.values[0]
        self.with_gpu = row_e.with_gpu.values[0]
        self.features_list = row_e.features_list.values[0]
        self.prep_list = row_e.prep_list.values[0]
        self.metrics_name = row_e.metrics_name.values[0]
        self.level_no = row_e.level_no.values[0]
        self.useful_features = row_e.features_list.values[0]
        self.aug_type = row_e.aug_type.values[0]
        self._dataset = row_e._dataset.values[0]
        self.use_cutmix = row_e.use_cutmix.values[0]

        super().__init__(
            model_name=self.model_name,
            comp_type=self.locker["comp_type"],
            metrics_name=self.metrics_name,
            prep_list=self.prep_list,
            with_gpu=self.with_gpu,
            aug_type=self.aug_type,
            _dataset=self._dataset,
            use_cutmix=self.use_cutmix,
        )
        # When we call super() It is like calling their init 
        # so all the default initialization of parent class is made here
        # So we must manually change it here after doing super() 
        # if we did it before calling super() it will be overwritten by parent init
        # Overrite exp_no of OptunaOptimizer since it takes exp_no from current_dict 
        self.exp_no = exp_no 
        if self.exp_no == -1:
            row_e = self.Table[self.Table.exp_no == list(self.Table.exp_no.values)[-1]]
            self.exp_no = row_e.exp_no.values[0]
        # fold specific work do here 
        # only looking at feature importance while optimizing
        self.calculate_feature_importance = False
        self.calculate_permutation_feature_importance = False 
        global_variables.counter = 0 # since we can't get trial no from optuna we are maintaining our own
        # run() can be called for each new exp so we initialize here rather than at init
        self.feature_importance_table = None # store the feature importances each trial wise

        # We can set manually on which fold_name to create oof predictions
        self.fold_name  = row_e.opt_fold_name.values[0] # 'fold3'



        # --- sanity check [new_feat, old_feat, feat_title]
        # ---------------
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        #new_features = [f"pred_e_{self.exp_no}_{self.exp_no}"]
        #useful_features = self.useful_features



    def run_folds(self, fold_name="--|--" ):

        check_memory_usage("run folds started", self, 0)
        ######################################
        #         Memory uage                #
        ######################################
        tracemalloc.start()

        if fold_name not in ["", "--|--"]: # empty 
            self.fold_name = fold_name
        self._state = "fold"


        """
        IF WE DON'T DO SANITY CHECK THEN OLD PREDICTION WILL BE REPLACED WITH NEW PREDICTION
        FOR THE FEATURE_DICT, It will not create new key but replace old key and the value
        Use it not in init because new fold_name is made in run_folds function
        """
        self.isRepetition() 

        # image_path = f'../input/input-{self.locker["comp_name"]}/' + "train_img/"
        # my_folds = pd.read_csv(f"../configs/configs-{self.comp_name}/my_folds.csv")
#        # my_folds = pd.read_parquet(
        #     f"../input/input-{self.comp_name}/my_folds.parquet"
        # )
        # test = pd.read_csv(f"../configs/configs-{self.comp_name}/test.csv")
        scores = []
        oof_prediction = {}
        test_predictions = []

        print("Running folds:")
        # BOTTLENECK
        return_type = "numpy_array" # "numpy_array" # "tensor"

        # don't delete ordered_list test because it is used later by obj function for the sanity check 
        print("Before Bottleneck:", len(self.useful_features))
        self.xtest , self.ordered_list_test = bottleneck_test(self.locker['comp_name'], self.useful_features, return_type)
        print("After Bottleneck:", len(self.ordered_list_test))

        if self.model_name.startswith("k"):
            self.xtest[np.isnan(self.xtest)] = 0
        if self.model_name == "xgb": # for lgb no need
            # if we are using feature importance it is better to convert it to dataframe 
            if self.calculate_feature_importance:   
                self.xtest = pd.DataFrame(self.xtest, columns = self.ordered_list_test)

            self.xtest = xgb.DMatrix(self.xtest)
        for fold in range(self.locker['fold_dict'][self.fold_name]): #self.locker["no_folds"]):
            self.optimize_on = [fold]  # setting on which to optimize 
            #set it also as list It is much more flexible like this
            # select data: xtrain xvalid etc
            self.run(
                self.useful_features
            )  # don't delete variables of self here
            # since used here
            scores.append(self.obj("--|--"))

            if self.locker["comp_type"] == "multi_label":
                # 3 to 1 so elongate
                self.test_preds = coln_3_1(self.test_preds)
                self.valid_preds = coln_3_1(self.valid_preds)
            else:
                self.valid_preds = self.valid_preds[0]
                self.test_preds = self.test_preds[0]

            oof_prediction.update(dict(zip(self.val_idx, self.valid_preds)))  # oof
            test_predictions.append(self.test_preds)

            ######################################
            #         gc collect                 #
            ######################################
            del self.xtrain, self.xvalid, self.val_idx, self.ytrain, self.yvalid #, self.test don't delete xtest since called only once
            # just see the importance of deleting self.test 
            # one thing we can try is instead of reading multiple times just read once from outside
            _ = gc.collect()
            check_memory_usage(f"fold #{fold}", self, 0)
            print()

        #------------------------------
        # save feature importance
        if self.calculate_feature_importance or self.calculate_permutation_feature_importance:
            # need to calculate feature importance 
            save_pickle(f"../configs/configs-{self.locker['comp_name']}/feature_importance/feature_importance_e_{self.exp_no}_fold.pkl",self.feature_importance_table)
            del self.feature_importance_table
            gc.collect()

        del self.xtest 
        gc.collect() # delete xtest now

        # CHECK MEMORY USAGE JUST AFTER ALL FOLDS
        check_memory_usage("AFTER_ALL_FOLDS", self, 1)

        """
        Note: for multi_label we can't save oof in my_folds.csv since multiple target columns are flattened to create submission file.
        """
        # save oof predictions
        temp_valid_predictions = pd.DataFrame.from_dict(
            oof_prediction, orient="index"
        ).reset_index()
        temp_valid_predictions.columns = [
            f"{self.locker['id_name']}",
            f"{self.locker['target_name']}"
        ]
        # if regression problem then rank it
        if self.locker["comp_type"] in [
            "regression",
            "2class",
        ] and self.metrics_name in [
            "auc",
            "auc_tf",
        ]:  # auc takes rank s
            #print("checking")
            # quite sensitive point 
            temp_valid_predictions[f"{self.locker['target_name']}"]= stats.rankdata(temp_valid_predictions.loc[:,f"{self.locker['target_name']}"])

            final_test_predictions = [stats.rankdata(f) for f in test_predictions]

            del test_predictions 
            gc.collect()
        else:
            # temp_valid_predictions already defined
            final_test_predictions = test_predictions.copy()
            
            del test_predictions 
            gc.collect()

        # save oof predictions
        if self.locker["comp_type"] != "multi_label":
            # SPLIT SUBMISSIONS
            #-> a = a.sort_values(by=['customer_ID'])
            #temp_valid_predictions = np.array(temp_valid_predictions.sort_values(by=[self.locker['id_name']])["prediction"].values)
            temp_valid_predictions = np.array(temp_valid_predictions.sort_values(by=[self.locker['id_name']])[f"{self.locker['target_name']}"].values)
            
            # temp_valid_predictions = temp_valid_predictions.set_index('customer_ID')
            # temp_valid_predictions = temp_valid_predictions.sort_index().reset_index().values

            # pkl # it is 1D array
            save_pickle(f"../configs/configs-{self.locker['comp_name']}/oof_preds/oof_pred_e_{self.exp_no}_{self.fold_name}.pkl", temp_valid_predictions)

            # clear memory 
            del temp_valid_predictions 
            gc.collect()
        else:
            input("We have not found way to save multi-label problem yet!!")

        # SPLIT PREDICTIONS
        # save test predictions
        # mode is good for classification problem but not for regression problem
        if self.locker["comp_type"] in ["regression", "2class"]:
            # so we will use regression methods [ for now using 0.2]
            final_test_predictions = [
                np.sum([0.2 * i for i in f], axis=0) for f in [final_test_predictions]
            ][0]
            #final_test_predictions = [np.mean(f, axis=0) for f in [final_test_predictions]][0]
        else:
            final_test_predictions = stats.mode(np.column_stack(final_test_predictions), axis=1)[0]
        
        print("Test preds")
        print(final_test_predictions)
        save_pickle(f"../configs/configs-{self.locker['comp_name']}/test_preds/test_pred_e_{self.exp_no}_{self.fold_name}.pkl", np.array(final_test_predictions))
        
        del final_test_predictions
        gc.collect()

        # ---------------
        # feature name should be unique enough not to match with base features
        #new_features = [f"pred_e_{self.exp_no}_{self.fold_name}"]
        #useful_features = self.useful_features
        # -----------------------------update current dict
        self.current_dict["current_feature_no"] = (
            self.current_dict["current_feature_no"] + 1
        )
        feat_no = self.current_dict["current_feature_no"]
        level_no = self.current_dict["current_level"]
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl",
            self.current_dict,
        )
        # -----------------------------dump feature dictionary
        feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        #--
        # sanity check is already done that's why we are dumping
        is_it_repetition = False
        if f"exp_{self.exp_no}" in list(feat_dict.keys()):
            # experiment already predicted so append feature
            if f"pred_e_{self.exp_no}_{self.fold_name}" in list(feat_dict[f"exp_{self.exp_no}"][0]):
                #this feature already present no need to put again either raise error or let it overwrite
                print(f"This feature pred_e_{self.exp_no}_{self.fold_name} is already present no need to put again either raise error or let it overwrite")
                s = input("Type Y/y to overwrite or Type N/n to raise error.")
                if s.upper() == "N":
                    raise Exception("Repeating feture") 
                is_it_repetition = True
                print("Overwritting")
            else:
                feat_dict[f"exp_{self.exp_no}"][0] += [f"pred_e_{self.exp_no}_{self.fold_name}"]

        else: 
            # first time so add
            feat_dict[f"exp_{self.exp_no}"] = [
                [f"pred_e_{self.exp_no}_{self.fold_name}"],
                self.useful_features
            ]
        #---
        # feat_dict[f"l_{level_no}_f_{feat_no}"] = [
        #     new_features,
        #     useful_features,
        #     f"exp_{self.exp_no}",
        # ]
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl",
            feat_dict,
        )
        # -----------------------
        print("New features create:- ")
        print([f"pred_e_{self.exp_no}_{self.fold_name}"])
        # -----------------------------
        print("scores: ")
        print(scores)

        
        # ---- update table
        if not is_it_repetition:
            self.Table.loc[self.Table.exp_no == self.exp_no, "fold_mean"].values[0] += [np.mean(scores)]
            self.Table.loc[self.Table.exp_no == self.exp_no, "fold_std"].values[0] += [np.std(scores)]
            self.Table.loc[self.Table.exp_no == self.exp_no, "oof_fold_name"].values[0] += [self.fold_name]

        # pblb to be updated mannually
        # ---------------- dump table
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/Table.pkl", self.Table
        )

        gc.collect()
        check_memory_usage("run folds stop", self)
        tracemalloc.stop()

    def isRepetition(self):
        title_new = f"exp_{self.exp_no}"
        if title_new in list(self.feat_dict.keys()):
            # So this experiment is done but is that fold_name is used 
            if f"pred_e_{self.exp_no}_{self.fold_name}" in self.feat_dict[title_new][0]:
                # pred_e_10_fold10, pred_e_10_fold5
                raise Exception(f"This feature : pred_e_{self.exp_no}_{self.fold_name}, with title : {title_new} is already present in my_folds!")
        gc.collect()


if __name__ == "__main__":
    #p = predictor(exp_no=-1)  # last exp
    #p = predictor(exp_no=236)  # last exp

    # more the no of folds bigger training size in each fold
    # fold_name = "fold5" #fold5" #"fold5" # "fold3" , "fold5", "fold10" , "fold20", ""
    # p.run_folds(fold_name)
    # del p
    # p = predictor(exp_no=3)  # exp_4
    #p.run_folds()

    for exp_no in [-1]:
        fold_name = "fold5"
        p = predictor(exp_no = exp_no)
        p.run_folds(fold_name)
        del p 
        gc.collect()

