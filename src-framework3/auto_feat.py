from optuna_search import OptunaOptimizer
from feature_generator import features
from feature_picker import Picker
import os
import sys
import gc
import pickle
import pandas as pd
import tracemalloc

# from custom_models import UModel
# from custom_models import *
from utils import *

from settings import *

from experiment import Agent
from show_importance import Importance


if __name__ == "__main__": 
    while True:
    
        """
        # Feature selection process Starts
        """
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")
        last_exp_no = -1
        if last_exp_no == -1:
            row_e = Table[Table.exp_no == list(Table.exp_no.values)[-1]]
            last_exp_no = row_e.exp_no.values[0]
        else:
            row_e = Table[Table.exp_no == last_exp_no]
        # we get exp no of the latest experiment 

        auto_feat_dict = load_json(f"../configs/configs-{comp_name}/auto_feat.json")
        print(last_exp_no)
        useful_features = auto_feat_dict[str(last_exp_no)]

        #useful_features += amzcomp1_settings().feature_dict["ver2"]
        useful_features = list(set(useful_features))
        """
        # Feature selection process Ends
        """
        # ==========================================================
        model_name = "xgbr"  # -----s--->
        """
            [
                "lgr",
                "lir",
                "xgb",
                "xgbc",
                "xgbr",
                "cbc",
                "cbr",
                "mlpc", 
                "rg", 
                "ls",
                "knnc", 
                "dtc", 
                "adbc", 
                "gbmc" ,
                "gbmr,
                "hgbc", 
                "lgb", 
                "lgbmc", 
                "lgbmr", 
                "rfc" ,
                "rfr",
        # --------------->["k1", "k2", "k3", "tez1", "tez2", "p1" ,"pretrained"]
        """
        comp_type = (
            "2class"  # -------->["regression", "2class","multi_class", "multi_label"]
        )
        metrics_name = "amzcomp1_metrics"  # --------->["getaroom_metrics", "amex_metric","amex_metric_mod", "accuracy","f1","recall","precision", "auc", "logloss","auc_tf","mae","mse","rmse","msle","rmsle","r2"]
        n_trials = 20 #30  # ------------> no of times to run optuna
        prep_list = [
            "Sd",
        ]  # ------> ["SiMe", "SiMd",~ "SiMo", "Mi", "Ro", "Sd", "Lg"] <= _prep_list
        prep_list = []
        fold_name = "fold5"  # ['fold3', 'fold5', 'fold10', 'fold20']
        optimize_on = [random.choice(range(5))] #[4]  # fold on which optimize # 0,1,2,4
        with_gpu = True

        aug_type = "aug2"  # "aug1", "aug2", "aug3", "aug4"
        _dataset = "DigitRecognizerDataset"  # "BengaliDataset", "ImageDataset", "DigitRecognizerDataset", "DigitRecognizerDatasetTez2"
        use_cutmix = False

        # CALLBACKS
        # lgbmClassifiers callback: 
        # https://lightgbm.readthedocs.io/en/latest/Python-API.html#callbacks
        """
        ############################ change learning rate
        custom schedulers
        :=>cosine_decay , exponential_decay, simple_decay 
        built in scheduler
        :=>"ReduceLROnPlateau" # EarlyStopping, 

        ############################# save model
        :=>chk_pt : whether to use checkpoint or not

        ############################### stop training
        :=>terminate_on_NaN 
        >Callbacks monitor a particular varaibable and stops exectuion when it crosses fixed value , 
        >early_top also monitors a particular value but it stops when it stops improving it has some patience 
        builtin callback
        :=>early_stopping 
        custom callback 
        :=>myCallback1
        """
        # don't use early_stopping with cyclic decay lr because model gets good and bad periodically and it doesn't mean we should terminate.
        # "swa", "cosine_decay", "exponential_decay", "simple_decay", "ReduceLROnPlateau", "chk_pt", "terminate_on_NaN", "early_stopping", "myCallback1"
        callbacks_list = ["terminate_on_NaN"] # ["exponential_decay", "terminate_on_NaN"] # [swa,early_stop]
        # -----------------------------------------------------------
        note = "ragnar"
        e = Agent(
            useful_features=useful_features,
            model_name=model_name,
            comp_type=comp_type,
            metrics_name=metrics_name,
            n_trials=n_trials,
            prep_list=prep_list,
            fold_name = fold_name,
            optimize_on=optimize_on,
            with_gpu=with_gpu,
            aug_type=aug_type,
            _dataset=_dataset,
            use_cutmix=use_cutmix,
            callbacks_list=callbacks_list,
            note=note,
        )
        print("=" * 40)
        print("Useful_features:", useful_features)

        e.run()
        del e
        # -------------------------------------------------------------
        # exp_list = ["1"]  # ----------------> [1,2,3,4]
        # e.show(exp_list)

        """
        {'learning_rate': 0.010821262164314453, 'max_depth': 16, 'min_child_weight': 5, 'subsample': 0.4521783648128741, 'n_estimators': 500, 'objective': 'reg:squarederror', 'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': 'gpu_predictor'}
        
        """

        # Make prediction also 
        os.system(f"python predict.py")


        # Let's save auto features 
        

        direction = "minimize"

        technique = "bagging" # "weighted_mean" , "best" , "mean", "top50", "bagging"

        last_exp_no += 1
        f = Importance(exp_no=last_exp_no)
        # helps when doing weighted mean

    
        base_features = None

        type_importance = "fold" #"fold", "opt"
        pick = None # pick top 2 trials out of 5
        top = None
        threshold = 1
        #f.show(technique= technique, top=top, threshold=threshold, direction=direction, pick = pick, type_importance= type_importance, base_features=base_features)
        

        useful_features = f.give(technique= technique, top=top, threshold=threshold, direction=direction, pick = pick, type_importance= type_importance, base_features=base_features)
        auto_feat_dict[str(last_exp_no)] = useful_features
        save_json(f"../configs/configs-{comp_name}/auto_feat.json", auto_feat_dict)


        #raise Exception('stop')
