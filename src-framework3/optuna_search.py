from metrics import ClassificationMetrics
from metrics import RegressionMetrics
from metrics import *
from collections import defaultdict
import pickle
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from scipy import stats
import gc
import psutil
import seaborn as sns
import tracemalloc
import pathlib
from settings import * 

"""
################
"""
# import albumentations
import numpy as np
import pandas as pd

# import timm
import torch
import torch.nn as nn
from sklearn import metrics, model_selection
from torch.utils.data import Dataset, DataLoader
#########
#Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

########################
# use it only when using tez2 i.e latest version
# from tez import Tez, TezConfig
# from tez.callbacks import EarlyStopping
# from tez.utils import seed_everything
# use this will pip install tez
# from tez import Tez, TezConfig
# from tez.callbacks import EarlyStopping
# from tez.utils import seed_everything
import global_variables

"""
"""
sns.set()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor

# https://stackoverflow.com/questions/47152610/what-is-the-difference-between-xgb-train-and-xgb-xgbregressor-or-xgb-xgbclassif
import xgboost as xgb # when calling the low level api

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from optuna.integration import LightGBMPruningCallback

# get skewed features to impute median instead of mean
from scipy.stats import skew
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBRegressor, XGBRFRegressor
import itertools
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor, log_evaluation
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Dense,
    Dropout,
    LSTM,
    Input,
    Activation,
    concatenate,
    Bidirectional,
)
from keras import optimizers
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    LSTM,
)
from keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
    LearningRateScheduler,
)
import warnings

# Filter up and down
np.random.seed(1337)  # for reproducibility
warnings.filterwarnings("ignore")

from metrics import ClassificationMetrics, RegressionMetrics

# tez ----------------------------
import os
import albumentations as A
import pandas as pd
import numpy as np
import tez
from tez.datasets import ImageDataset
from tez.callbacks import EarlyStopping
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn import metrics, model_selection, preprocessing
import timm

from sklearn.model_selection import KFold

# ignoring warnings
import warnings

warnings.simplefilter("ignore")
import os, cv2, json
from PIL import Image

import random

import tez
from tez.datasets import ImageDataset
from tez.callbacks import EarlyStopping


from custom_models import *
from custom_classes import *
from utils import *

# ------------------------------
# keras image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------
"""
self._state = fold, opt, seed
"""
##----------------
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from torchcontrib.optim import SWA
import torch.nn as nn
from model_dispatcher import MODEL_DISPATCHER


class OptunaOptimizer:
    def __init__(
        self,
        model_name="lgr",
        comp_type="2class",
        metrics_name="accuracy",
        n_trials=2,  # 50,
        fold_name = "fold3",
        optimize_on=[0],
        prep_list=[],
        with_gpu=False,
        save_models=True,
        aug_type="aug2",
        _dataset="ImageDataset",
        use_cutmix=True,
        callbacks_list=[],
    ):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.comp_name = comp_name # put it here for the mkdir of lgb callback
        self.locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
        self.current_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl"
        )
        self.exp_no = self.current_dict["current_exp_no"] # set it as default and will be changed by predict.py and seed_it.py 

        self.calculate_feature_importance = False # later integrate it (No need just keep saving that's all or maybe)
        self.calculate_permutation_feature_importance = False 

        self.save_models = save_models
        self._trial_score = None
        self._history = None
        self.use_cutmix = use_cutmix
        self.callbacks_list = callbacks_list
        self.all_callbacks = [
            "cosine_decay",
            "exponential_decay",
            "simple_decay",
            "ReduceLROnPlateau",
            "chk_pt",
            "terminate_on_NaN",
            "early_stopping",
            "myCallback1",
        ]
        self.comp_list = ["regression", "2class", "multi_class", "multi_label"]
        self.metrics_list = [
            "accuracy",
            "f1",
            "recall",
            "precision",
            "auc",
            "logloss",
            "auc_tf",
            "mae",
            "mse",
            "rmse",
            "msle",
            "rmsle",
            "r2",
            "amex_metric",
            "amex_metric_mod"
            "amex_metric_lgb_base",
            "getaroom_metrics",
            "amzcomp1_metrics",
        ]
        self.model_list = [
            "lgr",
            "lir",
            "xgb",
            "xgbc",
            "xgbr",
            "cbc",
            "cbr",
            "mlpc",
            "mlpr",
            "rg",
            "ls",
            "knnc",
            "dtc",
            "adbc",
            "gbmc",
            "gbmr",
            "hgbc",
            "lgb",
            "lgbmc",
            "lgbmr",
            "rfc",
            "rfr",
            "tabnetc", 
            "tabnetr", 
            "k1",
            "k2",
            "k3",
            "k4",
            "tez1",
            "tez2",
            "p1",
            "pretrained",
        ]
        self._prep_list = ["SiMe", "SiMd", "SiMo", "Mi", "Ro", "Sd", "Lg"]
        self.metric_list = ["amzcomp1_metrics","getaroom_metrics", "amex_metric","amex_metric_mod", "amex_metric_lgb_base","accuracy","f1","recall","precision", "auc", "logloss","auc_tf","mae","mse","rmse","msle","rmsle","r2"]
        self.prep_list = prep_list
        self.comp_type = comp_type
        self.metrics_name = metrics_name
        self.with_gpu = with_gpu
        self.aug_type = aug_type
        self._dataset = _dataset
        self._log_table = None  # will track experiments
        self._state = "opt"  # ["opt","fold", "seed"]
        # in start we want to find best params then we will loop
        if self.metrics_name in [
            "accuracy",
            "f1",
            "recall",
            "precision",
            "auc",
            "auc_tf",
            "r2",
            "amex_metric",
            "amex_metric_mod",
            "amex_metric_lgb_base",
            "getaroom_metrics",
            "amzcomp1_metrics",
        ]:
            self._aim = "maximize"
        else:
            self._aim = "minimize"
        self.n_trials = n_trials
        self.best_params = None
        self.best_value = None
        self.model_name = model_name
        self.fold_name = fold_name 
        self.optimize_on = optimize_on
        self.sanity_check()

    def show_variables(self):
        print()
        for i, (k, v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>", v)
            gc.collect()
        print()

    def sanity_check(self):
        if self.comp_type not in self.comp_list:
            raise Exception(f"{self.comp_type} not in the list {self.comp_list}")
        if self.metrics_name not in self.metrics_list:
            raise Exception(f"{self.metrics_name} not in the list {self.metrics_list}")
        if self.model_name not in self.model_list:
            raise Exception(f"{self.model_name} not in the list {self.model_list}")
        if self.fold_name not in list(self.locker['fold_dict'].keys()):
            raise Exception(
                f"{self.fold_name} not in {list(self.locker['fold_dict'].keys())}"
            )
        for f in self.optimize_on:
            if f >= self.locker['fold_dict'][self.fold_name]: # self.locker["no_folds"]:
                raise Exception(
                    f"{self.optimize_on} out of range {self.locker['fold_dict'][self.fold_name]}"
                )
        for p in self.prep_list:
            if p not in list(self._prep_list):
                raise Exception(f"{p} is invalid preprocessing type!")
            gc.collect()
        for c in self.callbacks_list:
            if c not in self.all_callbacks:
                raise Exception(f"{c} is invalid callback type!")
            gc.collect()

    def help(self):
        print("comp_type:=> ", [comp for i, comp in enumerate(self.comp_list)])
        print("metrics_name:=>", [mt for i, mt in enumerate(self.metrics_list)])
        print()
        models = [
            "LogisticRegression",
            "LinearRegression",
            "XGBClassifier",
            "XGBRegressor",
        ]
        print("model_name:=>")
        for a, b in list(
            zip(self.model_list, models)
        ):  # ,[mt for i,mt in enumerate(self.model_list)])
            print(f"{a}:=> {b}")
            gc.collect()
        print()
        preps = [
            "SimpleImputer_mean",
            "SimpleImputer_median",
            "SimpleImputer_mode",
            "RobustScaler",
            "StandardScaler",
            "LogarithmicScaler",
        ]
        print("preprocess_names:=>")
        for a, b in list(
            zip(self._prep_list, preps)
        ):  # ,[mt for i,mt in enumerate(self.model_list)])
            print(f"{a}:=> {b}")
        print("preprocess_names:=>", [p for i, p in enumerate(self._prep_list)])
        ## preprocess
        # Si: SimpleImputer SiMe SiMd SiMo
        # Mi: MinMaxScaler
        # Ro: RobustScaler
        # Sd: StandardScaler
        # Lg: Logarithmic Scaler

    def generate_random_no(self):
        if self._state == "opt":
            comp_random_state = self.locker["random_state"]
            total_no_folds = self.locker['fold_dict'][self.fold_name] #self.locker["no_folds"]
            # fold_on = self.optimize_on
            metric_no = self.metrics_list.index(self.metrics_name)
            comp_type_no = self.comp_list.index(self.comp_type)
            model_no = self.model_list.index(self.model_name)

            prep_no = 0
            for p in self._prep_list:
                if p == "Lg":
                    prep_no += 10
                else:
                    prep_no += self._prep_list.index(p)
            # round_on
            # level_on
            #
            seed = (
                comp_random_state
                + total_no_folds * 2
                + metric_no * 3
                #+ self.optimize_on * 4
            )
            for i,o in enumerate(self.optimize_on):
                seed =  + o * (4+i)
            # seed = (
            #     comp_random_state
            #     + total_no_folds * 2
            #     + metric_no * 3
            #     + self.optimize_on * 4
            # )
            seed += int(
                comp_type_no * 5
                + model_no * 6
                + prep_no * 7
                + self.current_dict["current_level"]
            )  # + round_on * 4 + level_on * 5

            if self.callbacks_list != []:
                for c in self.callbacks_list:
                    seed += self.all_callbacks.index(c)
            seed = int(seed)
        else:
            seed = int(self._random_state) # it should be int type

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(
            seed
        )  # f"The truth value of a {type(self).__name__} is ambiguous. "
        return seed  # np.random.randint(3, 1000) # it should return 5

    def get_params(self, trial):
        model_name = self.model_name

        if model_name == "lgr":
            params = {
                "class_weight": trial.suggest_categorical(
                    "class_weight",
                    [
                        "balanced",
                        None,
                        {
                            1: 1,
                            0: (
                                sum(list(self.ytrain == 0))
                                / sum(list(self.ytrain == 1))
                            ),
                        },
                    ],
                ),
                "penalty": trial.suggest_categorical(
                    "penalty", ["l2"]
                ),  # ['l1','l2']),
                "C": trial.suggest_float("c", 0.01, 1000),
            }
            return params

        if model_name == "lir":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "subsample": trial.suggest_discrete_uniform(
                    "subsample", 0.6, 1.0, 0.05
                ),
                "n_estimators": trial.suggest_int("n_estimators", 1000, 10000, 100),
                "learning_rate": trial.suggest_discrete_uniform("learning_rate", 0.01, 0.1, 0.01),
                "reg_alpha": trial.suggest_int("reg_alpha", 1, 50),
                "reg_lambda": trial.suggest_int("reg_lambda", 5, 100),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            }
            return params

        if model_name == "xgb":
            # for regression
            # https://xgboost.readthedocs.io/en/stable/python/python_intro.html#setting-parameters
            # https://www.kaggle.com/code/sudalairajkumar/xgb-starter-in-python
            # https://www.kaggle.com/general/197091
            params = {
                "objective": trial.suggest_categorical("objective", ['reg:squarederror']),
                #"eta": trial.suggest_categorical("eta", ['train']),
                "learning_rate": trial.suggest_float("learning_rate", 0,0.2),
                "max_depth": trial.suggest_categorical("max_depth", [6]),
                "silent": trial.suggest_categorical("silent", [1]),
                #"num_class": trial.suggest_categorical("num_class", [3]),
                "eval_metric": trial.suggest_categorical("eval_metric", ['rmse']),
                "min_child_weight": trial.suggest_categorical("min_child_weight", [1]),
                "subsample": trial.suggest_categorical("subsample", [0.7]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7]),
                "booster": trial.suggest_categorical("booster", ['gbtree']),

            }

            # diff b/w .train() and .fit()
            # https://stackoverflow.com/questions/47152610/what-is-the-difference-between-xgb-train-and-xgb-xgbregressor-or-xgb-xgbclassif 
            # .fit() is a wrapper over .train()

            # use evaluation metrics 
            # https://stackoverflow.com/questions/60231559/how-to-set-eval-metrics-for-xgboost-train
            
            # xgb_parms = {
            #     'max_depth':4,
            #     'learning_rate':0.05,
            #     'subsample':0.8,
            #     'colsample_bytree':0.6,
            #     'eval_metric':'logloss',
            #     'objective':'binary:logistic',
            #     'tree_method':'gpu_hist',
            #     'predictor':'gpu_predictor',
            #     'random_state':SEED
            # }
            
            ## https://www.kaggle.com/code/karandora/xgboost-optuna Implement Features also from this notebook
            # param = {
            #     'booster':'gbtree',
            #     'tree_method':'gpu_hist',
            #     "objective": "binary:logistic",
            #     'lambda': trial.suggest_loguniform(
            #         'lambda', 1e-3, 10.0
            #     ),
            #     'alpha': trial.suggest_loguniform(
            #         'alpha', 1e-3, 10.0
            #     ),
            #     'colsample_bytree': trial.suggest_float(
            #         'colsample_bytree', 0.5,1,step=0.1
            #     ),
            #     'subsample': trial.suggest_float(
            #         'subsample', 0.5,1,step=0.1
            #     ),
            #     'learning_rate': trial.suggest_float(
            #         'learning_rate', 0.001,0.05,step=0.001
            #     ),
            #     'n_estimators': trial.suggest_int(
            #         "n_estimators", 80,1000,10
            #     ),
            #     'max_depth': trial.suggest_int(
            #         'max_depth', 2,10,1
            #     ),
            #     'random_state': 99,
            #     'min_child_weight': trial.suggest_int(
            #         'min_child_weight', 1,256,1
            #     ),
            # }

            # # https://www.kaggle.com/code/sietseschrder/xgboost-starter-0-793
            # params = {
            #     "max_depth": trial.suggest_int("max_depth", 2, 10), # 6 --> 10
            #     "subsample": trial.suggest_discrete_uniform(
            #         "subsample", 0.6, 1.0, 0.05
            #     ),
            #     #"n_estimators": trial.suggest_int("n_estimators", 10, 100, 10),
            #     "learning_rate": trial.suggest_discrete_uniform("learning_rate", 0.001, 0.1, 0.005), # "eta"
            #     #"reg_alpha": trial.suggest_int("reg_alpha", 1, 10),
            #     #"reg_lambda": trial.suggest_int("reg_lambda", 5, 20),
            #     #"min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
            #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            #     "objective": trial.suggest_categorical(
            #         "objective", ["binary:logistic"]
            #     )
                
            # }

            # big file
            # params = {
            #     "max_depth": trial.suggest_int("max_depth", 2, 4),
            #     "subsample": trial.suggest_discrete_uniform(
            #         "subsample", 0.6, 1.0, 0.05
            #     ),
            #     "n_estimators": trial.suggest_int("n_estimators", 10, 100, 10),
            #     "eta": trial.suggest_discrete_uniform("eta", 0.01, 0.1, 0.01),
            #     "reg_alpha": trial.suggest_int("reg_alpha", 1, 10),
            #     "reg_lambda": trial.suggest_int("reg_lambda", 5, 20),
            #     "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
            #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            # }

            # --> base Good for small dataset
            # params = {
            #     "max_depth": trial.suggest_int("max_depth", 2, 15),
            #     "subsample": trial.suggest_discrete_uniform(
            #         "subsample", 0.6, 1.0, 0.05
            #     ),
            #     "n_estimators": trial.suggest_int("n_estimators", 1000, 10000, 100),
            #     "eta": trial.suggest_discrete_uniform("eta", 0.01, 0.1, 0.01),
            #     "reg_alpha": trial.suggest_int("reg_alpha", 1, 50),
            #     "reg_lambda": trial.suggest_int("reg_lambda", 5, 100),
            #     "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
            #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            # }

            if self.with_gpu == True:
                params.update(
                    {
                        "tree_method": trial.suggest_categorical(
                            "tree_method", ["gpu_hist"]
                        ),
                        "gpu_id": trial.suggest_categorical("gpu_id", [0]),
                        "predictor": trial.suggest_categorical(
                            "predictor", ["gpu_predictor"]
                        ),
                    }
                )
            return params

        if model_name == "xgbc":
            # diff b/w .train() and .fit()
            # https://stackoverflow.com/questions/47152610/what-is-the-difference-between-xgb-train-and-xgb-xgbregressor-or-xgb-xgbclassif 
            # .fit() is a wrapper over .train()

            # use evaluation metrics 
            # https://stackoverflow.com/questions/60231559/how-to-set-eval-metrics-for-xgboost-train
            
            # xgb_parms = {
            #     'max_depth':4,
            #     'learning_rate':0.05,
            #     'subsample':0.8,
            #     'colsample_bytree':0.6,
            #     'eval_metric':'logloss',
            #     'objective':'binary:logistic',
            #     'tree_method':'gpu_hist',
            #     'predictor':'gpu_predictor',
            #     'random_state':SEED
            # }
            
            ## https://www.kaggle.com/code/karandora/xgboost-optuna Implement Features also from this notebook
            # param = {
            #     'booster':'gbtree',
            #     'tree_method':'gpu_hist',
            #     "objective": "binary:logistic",
            #     'lambda': trial.suggest_loguniform(
            #         'lambda', 1e-3, 10.0
            #     ),
            #     'alpha': trial.suggest_loguniform(
            #         'alpha', 1e-3, 10.0
            #     ),
            #     'colsample_bytree': trial.suggest_float(
            #         'colsample_bytree', 0.5,1,step=0.1
            #     ),
            #     'subsample': trial.suggest_float(
            #         'subsample', 0.5,1,step=0.1
            #     ),
            #     'learning_rate': trial.suggest_float(
            #         'learning_rate', 0.001,0.05,step=0.001
            #     ),
            #     'n_estimators': trial.suggest_int(
            #         "n_estimators", 80,1000,10
            #     ),
            #     'max_depth': trial.suggest_int(
            #         'max_depth', 2,10,1
            #     ),
            #     'random_state': 99,
            #     'min_child_weight': trial.suggest_int(
            #         'min_child_weight', 1,256,1
            #     ),
            # }

            # # https://www.kaggle.com/code/sietseschrder/xgboost-starter-0-793
            # params = {
            #     "max_depth": trial.suggest_int("max_depth", 2, 6),
            #     "subsample": trial.suggest_discrete_uniform(
            #         "subsample", 0.6, 1.0, 0.05
            #     ),
            #     #"n_estimators": trial.suggest_int("n_estimators", 10, 100, 10),
            #     "learning_rate": trial.suggest_discrete_uniform("learning_rate", 0.01, 0.1, 0.01),
            #     #"reg_alpha": trial.suggest_int("reg_alpha", 1, 10),
            #     #"reg_lambda": trial.suggest_int("reg_lambda", 5, 20),
            #     #"min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
            #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            #     "objective": trial.suggest_categorical(
            #         "objective", ["binary:logistic"]
            #     ),
            #     # booster implemented from below discussion chris's comment
            #     # https://www.kaggle.com/competitions/amex-default-prediction/discussion/333953 
            #     # Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
            #     #"booster": trial.suggest_categorical("booster", ["dart", "gbtree", "gblinear"])
            #     "booster": trial.suggest_categorical("booster", [ "gbtree", "gblinear"])

                
            # }

            # big file
            # params = {
            #     "max_depth": trial.suggest_int("max_depth", 2, 4),
            #     "subsample": trial.suggest_discrete_uniform(
            #         "subsample", 0.6, 1.0, 0.05
            #     ),
            #     "n_estimators": trial.suggest_int("n_estimators", 10, 100, 10),
            #     "eta": trial.suggest_discrete_uniform("eta", 0.01, 0.1, 0.01),
            #     "reg_alpha": trial.suggest_int("reg_alpha", 1, 10),
            #     "reg_lambda": trial.suggest_int("reg_lambda", 5, 20),
            #     "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
            #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            # }


            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "subsample": trial.suggest_discrete_uniform(
                    "subsample", 0.6, 1.0, 0.05
                ),
                "n_estimators": trial.suggest_int("n_estimators", 1000, 10000, 100),
                "learning_rate": trial.suggest_discrete_uniform("learning_rate", 0.01, 0.1, 0.01),
                "reg_alpha": trial.suggest_int("reg_alpha", 1, 50),
                "reg_lambda": trial.suggest_int("reg_lambda", 5, 100),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                "objective": trial.suggest_categorical(
                    "objective", ["binary:logistic"]
                ),
                # booster implemented from below discussion chris's comment
                # https://www.kaggle.com/competitions/amex-default-prediction/discussion/333953 
                # Which booster to use. Can be gbtree, gblinear or dart; gbtree and dart use tree based models while gblinear uses linear functions.
                #"booster": trial.suggest_categorical("booster", ["dart", "gbtree", "gblinear"])
                "booster": trial.suggest_categorical("booster", ["dart", "gbtree", "gblinear"])
            }


            # --> base Good for small dataset
            # params = {
            #     "max_depth": trial.suggest_int("max_depth", 2, 15),
            #     "subsample": trial.suggest_discrete_uniform(
            #         "subsample", 0.6, 1.0, 0.05
            #     ),
            #     "n_estimators": trial.suggest_int("n_estimators", 1000, 10000, 100),
            #     "eta": trial.suggest_discrete_uniform("eta", 0.01, 0.1, 0.01),
            #     "reg_alpha": trial.suggest_int("reg_alpha", 1, 50),
            #     "reg_lambda": trial.suggest_int("reg_lambda", 5, 100),
            #     "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
            #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            # }

            if self.with_gpu == True:
                params.update(
                    {
                        "tree_method": trial.suggest_categorical(
                            "tree_method", ["gpu_hist"]
                        ),
                        "gpu_id": trial.suggest_categorical("gpu_id", [0]),
                        "predictor": trial.suggest_categorical(
                            "predictor", ["gpu_predictor"]
                        ),
                    }
                )
            return params

        if model_name == "xgbr":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                "max_depth": trial.suggest_int("max_depth", 3,20), #[3, 5, 7, 10]
                "min_child_weight": trial.suggest_categorical(
                    "min_child_weight", [1, 3, 5]
                ),
                "subsample": trial.suggest_float("subsample", 0.01, 0.5),
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", [100, 200, 300, 400, 500, 1000, 1200, 1500]
                ),
                "objective": trial.suggest_categorical(
                    "objective", ["reg:squarederror"]
                ),
            }
            if self.with_gpu == True:
                params.update(
                    {
                        "tree_method": trial.suggest_categorical(
                            "tree_method", ["gpu_hist"]
                        ),
                        "gpu_id": trial.suggest_categorical("gpu_id", [0]),
                        "predictor": trial.suggest_categorical(
                            "predictor", ["gpu_predictor"]
                        ),
                    }
                )
            return params

        if model_name == "cbc":
            # CatBoostClassifier(iterations=1000, random_state=22, nan_mode='Min')
            # https://www.kaggle.com/code/bhavikardeshna/catboost-gradient-boosting-ensemble-learning/notebook
            # params = {
            #     "iterations": trial.suggest_int("iterations", 300, 1200),
            #     'nan_mode' : trial.suggest_categorical('nan_mode', ["Min"])
            # }


            # main
            params = {
                "iterations": trial.suggest_int("iterations", 300, 1200),
                "objective": trial.suggest_categorical(
                    "objective", ["Logloss", "CrossEntropy"]
                ),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bernoulli"] #["Bayesian", "Bernoulli", "MVS"]
                ),
                # 'MVS' for CPU and only can be used in CPU. For GPU it is 'Bernoulli'
                # https://www.kaggle.com/competitions/amex-default-prediction/discussion/328606#1809347
                "od_wait": trial.suggest_int("od_wait", 500, 2000),
                "learning_rate": trial.suggest_uniform("learning_rate", 0.02, 1),
                "reg_lambda": trial.suggest_uniform("reg_lambda", 1e-5, 100),
                "random_strength": trial.suggest_uniform("random_strength", 10, 50),
                "depth": trial.suggest_int("depth", 1, 15),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
                "leaf_estimation_iterations": trial.suggest_int(
                    "leaf_estimation_iterations", 1, 15
                ),
                "verbose": False,
            }
            if self.with_gpu == True:
                params.update(
                    {
                        "task_type": trial.suggest_categorical("task_type", ["GPU"]),
                        "devices": trial.suggest_categorical("devices", ["0"]),
                    }
                )
            return params

        if model_name == "cbr":

            # main
            params = {
                "iterations": trial.suggest_int("iterations", 300, 1000),
                "objective": trial.suggest_categorical(
                    "objective", ["RMSE"] #, "MultiRMSE", "SurvivalAft", "MAE", "Quantile", "LogLinQuantile", "Poisson", "MAPE", "Lq"]
                ),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bernoulli"] #["Bayesian", "Bernoulli", "MVS"]
                ),
                # 'MVS' for CPU and only can be used in CPU. For GPU it is 'Bernoulli'
                # https://www.kaggle.com/competitions/amex-default-prediction/discussion/328606#1809347
                "od_wait": trial.suggest_int("od_wait", 500, 600), #900), # 2000
                "learning_rate": trial.suggest_uniform("learning_rate", 0.02, 1),
                "reg_lambda": trial.suggest_uniform("reg_lambda", 1e-5, 30), #50),
                "random_strength": trial.suggest_uniform("random_strength", 10, 30),
                "depth": trial.suggest_int("depth", 1, 10), #15),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
                "leaf_estimation_iterations": trial.suggest_int(
                    "leaf_estimation_iterations", 1, 15
                ),
                "verbose": False,
            }
            if self.with_gpu == True:
                params.update(
                    {
                        "task_type": trial.suggest_categorical("task_type", ["GPU"]),
                        "devices": trial.suggest_categorical("devices", ["0"]),
                    }
                )
            return params

        if model_name == "mlpc":
            params = {
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["constant", "invscaling", "adaptive"]
                ),
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    [(5, 10, 5), (20, 10), (10, 20), (50, 50), (100, 100)],
                ),
                "alpha": trial.suggest_categorical(
                    "alpha", [0.3, 0.1, 0.01, 0.001, 0.0001]
                ),
                "activation": trial.suggest_categorical(
                    "activation", ["logistic", "relu", "tanh"]
                ),
                # "solver": trial.suggest_categorical("solver",['lbfgs'])
            }
            return params

        if model_name == "mlpr":
            params = {
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["constant", "invscaling", "adaptive"]
                ),
                "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    [(100, 100), (200,100,80),(800,40)],
                ),
                "alpha": trial.suggest_categorical(
                    "alpha", [0.3, 0.1, 0.01, 0.001, 0.0001]
                ),
                "activation": trial.suggest_categorical(
                    "activation", ["relu", "tanh", "identity"]
                ),
                # "solver": trial.suggest_categorical("solver",['lbfgs'])
            }
            # params = {
            #     "learning_rate": trial.suggest_categorical(
            #         "learning_rate", ["constant", "invscaling", "adaptive"]
            #     ),
            #     "hidden_layer_sizes": trial.suggest_categorical(
            #         "hidden_layer_sizes",
            #         [(5, 10, 5), (20, 10), (10, 20), (50, 50), (100, 100)],
            #     ),
            #     "alpha": trial.suggest_categorical(
            #         "alpha", [0.3, 0.1, 0.01, 0.001, 0.0001]
            #     ),
            #     "activation": trial.suggest_categorical(
            #         "activation", ["relu", "tanh", "identity"]
            #     ),
            #     # "solver": trial.suggest_categorical("solver",['lbfgs'])
            # }
            return params

        if model_name == "rg":
            params = {
                "alpha": trial.suggest_categorical(
                    "alpha", list(np.linspace(1, 100, 100))
                ),
                "solver": trial.suggest_categorical(
                    "solver",
                    ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                ),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True]),
            }
            return params

        if model_name == "ls":
            params = {
                "alpha": trial.suggest_categorical(
                    "alpha", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ),
                "selection": trial.suggest_categorical(
                    "selection", ["cyclic", "random"]
                ),
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True]),
            }
            return params

        if model_name == "knnc":
            params = {
                "leaf_size": trial.suggest_categorical(
                    "leaf_size", [5, 10, 15, 20, 25, 30, 35, 40, 45]
                ),
                "n_neighbors": trial.suggest_categorical(
                    "n_neighbors", [3, 4, 5, 6, 7, 8, 9, 10]
                ),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
                ),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
            }
            return params

        if model_name == "dtc":
            params = {
                "class_weight": trial.suggest_categorical(
                    "class_weight",
                    [
                        "balanced",
                        None,
                        {1: 1, 0: (sum(list(ytrain == 0)) / sum(list(ytrain == 1)))},
                    ],
                ),
                "criterion": trial.suggest_categorical(
                    "criterion", ["entropy", "gini"]
                ),
                "max_depth": trial.suggest_categorical("max_depth", [None, 5, 20, 70]),
                "min_samples_leaf": trial.suggest_categorical(
                    "min_samples_leaf", [5, 10, 15, 20, 25]
                ),
                "min_samples_split": trial.suggest_categorical(
                    "min_samples_split", [2, 10, 20]
                ),
            }
            return params

        if model_name == "adbc":
            params = {
                # "device_type": trial.suggest_categorical("device_type", ['gpu']),
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", [10, 100, 200, 500]
                ),  # ,1000,10000
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["SAMME", "SAMME.R"]
                ),
            }
            return params

        if model_name == "gbmc":
            params = {
                # "device_type": trial.suggest_categorical("device_type", ['gpu']),
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", [10, 100, 200, 500]
                ),  # ,1000,10000
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "loss": trial.suggest_categorical("loss", ["deviance", "exponential"]),
                "criterion": trial.suggest_categorical(
                    "criterion", ["friedman_mse", "mse", "mae"]
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["auto", "sqrt", "log2"]
                ),
                "min_samples_split": trial.suggest_float("min_samples_split", 0.1, 0.5),
                "min_samples_leaf": trial.suggest_float("min_samples_split", 0.1, 0.5),
                "subsample": trial.suggest_categorical(
                    "subsample", [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0]
                ),
            }
            return params

        if model_name == "gbmr":
            params = {
                # "device_type": trial.suggest_categorical("device_type", ['gpu']),
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", [10, 100, 200, 500]
                ),  # ,1000,10000
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "loss": trial.suggest_categorical("loss", ['squared_error']), #, 'absolute_error', 'huber', 'quantile']),
                "criterion": trial.suggest_categorical(
                    "criterion", ["friedman_mse", "squared_error", "mse"]
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["auto", "sqrt", "log2"]
                ),
                "min_samples_split": trial.suggest_float("min_samples_split", 0.1, 0.5),
                "min_samples_leaf": trial.suggest_float("min_samples_split", 0.1, 0.5),
                "subsample": trial.suggest_categorical(
                    "subsample", [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0]
                ),
            }
            return params

        if model_name == "hgbc":
            params = {
                "l2_regularization": trial.suggest_loguniform(
                    "l2_regularization", 1e-10, 10.0
                ),
                "early_stopping": trial.suggest_categorical(
                    "early_stopping", ["False"]
                ),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
                "max_iter": trial.suggest_categorical(
                    "max_iter", [i for i in range(1000, 12000, 100)]
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 30),
                "max_bins": trial.suggest_int("max_bins", 100, 255),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100000),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 80),
            }
            return params

        if model_name == "xgbdd":
            params = {
                "objective": trial.suggest_categorical("objective", ['regression']),
                #"eta": trial.suggest_categorical("eta", ['train']),
                "learning_rate": trial.suggest_float("learning_rate", 0,0.2),
                "max_depth": trial.suggest_categorical("max_depth", [6]),
                "silent": trial.suggest_categorical("silent", [1]),
                "num_class": trial.suggest_categorical("num_class", [3]),
                "eval_metric": trial.suggest_categorical("eval_metric", ['rmse']),
                "min_child_weight": trial.suggest_categorical("min_child_weight", [1]),
                "subsample": trial.suggest_categorical("subsample", [0.7]),
                "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.7]),

            }
            # param = {}
            # param['objective'] = 'multi:softprob'
            # param['eta'] = 0.1
            # param['max_depth'] = 6
            # param['silent'] = 1
            # param['num_class'] = 3
            # param['eval_metric'] = "mlogloss"
            # param['min_child_weight'] = 1
            # param['subsample'] = 0.7
            # param['colsample_bytree'] = 0.7
            # param['seed'] = seed_val

            return params

        if model_name == "lgb":
            # dart: https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7963
            # params = {
            #         'objective': 'binary',
            #         'metric': "binary_logloss",
            #         'boosting': 'dart',
            #         'seed': CFG.seed,
            #         'num_leaves': 100,
            #         'learning_rate': 0.01,
            #         'feature_fraction': 0.20,
            #         'bagging_freq': 10,
            #         'bagging_fraction': 0.50,
            #         'n_jobs': -1,
            #         'lambda_l2': 2,
            #         'min_data_in_leaf': 40
            #     }
            """
            params = {
                "objective": trial.suggest_categorical("objective", ["binary"]),
                "metric": trial.suggest_categorical("metric", ["binary_log_loss"]),
                "boosting": trial.suggest_categorical("boosting", ["dart"]),
                "num_leaves": trial.suggest_int("num_leaves", 80, 100, step=5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
                "bagging_freq": trial.suggest_categorical("bagging_freq", [1,10, 20]),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
                "lambda_l2": trial.suggest_int("lambda_l2", 0, 10, step=2),
                "min_data_in_leaf": trial.suggest_int( "min_data_in_leaf", 20, 100, step=10),
            }
            """

            # https://www.kaggle.com/competitions/amex-default-prediction/discussion/332575
            # params = {
            #         'objective': 'binary',
            #         'metric': "amex_metric",
            # }

            # set metric as same as we set in feval: "amex_metric"
            # params = {
            #     "objective": trial.suggest_categorical("objective", ["binary"]),
            #     "metric": trial.suggest_categorical("metric", ["amex_metric"]),
            #     "boosting": trial.suggest_categorical("boosting", ["dart"]),
            #     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
            # }   
            """

            params_lgbm = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'learning_rate': 0.01,
                    'objective': 'regression',
                    'metric': 'None',
                    'max_depth': -1,
                    'n_jobs': -1,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.7,
                    'lambda_l2': 1,
                    'verbose': -1
                    #'bagging_freq': 5
            }

            """
            params = {
                "task": trial.suggest_categorical("task", ['train']),
                "boosting_type": trial.suggest_categorical("boosting_type", ['gbdt', 'dart']),
                "learning_rate": trial.suggest_float("learning_rate", 0,0.2),
                "objective": trial.suggest_categorical("objective", ['regression']),
                "metric": trial.suggest_categorical("metric", ['None']),
                "max_depth": trial.suggest_categorical("max_depth", [-1]),
                "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
                #"feature_fraction": trial.suggest_categorical("feature_fraction", [0.7]),
                "feature_fraction": trial.suggest_categorical("feature_fraction", [0.7]),
                #"bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.7]),
                "bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.7]),
                "lambda_l2": trial.suggest_categorical("lambda_l2", [1]),
                "verbose": trial.suggest_categorical("verbose", [-1]),
                #"bagging_freq": trial.suggest_categorical("bagging_freq", [5]),

                # "objective": trial.suggest_categorical("objective", ["regression"]),#["binary"]),
                # "metric": trial.suggest_categorical("metric", [None]),#["binary_logloss"]),
                # "boosting": trial.suggest_categorical("boosting", ["dart"]),
                # "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.011),
                # "seed": trial.suggest_categorical("seed", [241]),
                # "num_leaves": trial.suggest_categorical("num_leaves", [95,101,105]),
                # "feature_fraction": trial.suggest_float("feature_fraction", 0.15,0.22),
                # "bagging_freq": trial.suggest_categorical("bagging_freq" , [9,11,12]),
                # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.45,0.52),
                # "n_jobs": trial.suggest_categorical("n_jobs" , [-1]),
                # "lambda_l2": trial.suggest_categorical("lambda_l2" , [1,2,3]),
                # "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf" , [35,41,45]),
            }   

            # params = {
            #     "objective": trial.suggest_categorical("objective", ["regression"]),#["binary"]),
            #     "metric": trial.suggest_categorical("metric", [None]),#["binary_logloss"]),
            #     "boosting": trial.suggest_categorical("boosting", ["dart"]),
            #     "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.011),
            #     "seed": trial.suggest_categorical("seed", [241]),
            #     "num_leaves": trial.suggest_categorical("num_leaves", [95,101,105]),
            #     "feature_fraction": trial.suggest_float("feature_fraction", 0.15,0.22),
            #     "bagging_freq": trial.suggest_categorical("bagging_freq" , [9,11,12]),
            #     "bagging_fraction": trial.suggest_float("bagging_fraction", 0.45,0.52),
            #     "n_jobs": trial.suggest_categorical("n_jobs" , [-1]),
            #     "lambda_l2": trial.suggest_categorical("lambda_l2" , [1,2,3]),
            #     "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf" , [35,41,45]),
            # }   
            if self.with_gpu == True:
                params.update(
                    {
                        "device_type": trial.suggest_categorical(
                            "device_type", ["gpu"]
                        ),
                    }
                )       
            return params 
        
        if model_name == "lgbmc":
            # amex amrosm 
            # link: https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart
            params = {
                "n_estimators": trial.suggest_categorical(
                    "n_estimators", [i for i in range(1100, 1300, 100)]
                ),  # 10000
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 80, 100, step=5),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1,0.2),
                "max_bins": trial.suggest_int("max_bins", 500,520, step=5),
                # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
                "boosting_type": trial.suggest_categorical("boosting_type", ['dart'])
                #"max_depth": trial.suggest_int("max_depth", 3, 12),
                # "min_data_in_leaf": trial.suggest_int(
                #     "min_data_in_leaf", 200, 10000, step=100
                # ),
                #"lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
                #"lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
                #"min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
                # "bagging_fraction": trial.suggest_float(
                #     "bagging_fraction"lgbmc, 0.2, 0.95, step=0.1
                # ),
                #"bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
                # "feature_fraction": trial.suggest_float(
                #     "feature_fraction", 0.2, 0.95, step=0.1
                # ),
            }

            # main
            # params = {
            #     "n_estimators": trial.suggest_categorical(
            #         "n_estimators", [i for i in range(1000, 10000, 100)]
            #     ),  # 10000
            #     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            #     "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
            #     "max_depth": trial.suggest_int("max_depth", 3, 12),
            #     "min_data_in_leaf": trial.suggest_int(
            #         "min_data_in_leaf", 200, 10000, step=100
            #     ),
            #     "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
            #     "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            #     "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
            #     "bagging_fraction": trial.suggest_float(
            #         "bagging_fraction", 0.2, 0.95, step=0.1
            #     ),
            #     "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
            #     "feature_fraction": trial.suggest_float(
            #         "feature_fraction", 0.2, 0.95, step=0.1
            #     ),
            # }
            ## gpu build
            if self.with_gpu == True:
                params.update(
                    {
                        "device_type": trial.suggest_categorical(
                            "device_type", ["gpu"]
                        ),
                    }
                )
            return params

        if model_name == "lgbmr":
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_data_in_leaf": trial.suggest_int(
                    "min_data_in_leaf", 200, 10000, step=100
                ),
                "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
                "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.2, 0.95, step=0.1
                ),
                "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", 0.2, 0.95, step=0.1
                ),
                "objective": trial.suggest_categorical("objective", ["regression"]),
            }
            if self.with_gpu == True:
                params.update(
                    {
                        "device_type": trial.suggest_categorical(
                            "device_type", ["gpu"]
                        ),
                    }
                )
            return params

        if model_name == "rfc":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 4, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 150),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 60),
            }
            return params

        if model_name == "rfr":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 5, 50),
                "max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 150),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 60),
                # "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                # "max_depth": trial.suggest_int("max_depth", 4, 50),
                # "min_samples_split": trial.suggest_int("min_samples_split", 2, 150),
                # "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 60),
            }
            return params

        if model_name == "tabnetc":
            # https://www.kaggle.com/code/wangqihanginthesky/baseline-tabnet

            # params = {
            #     "gamma": trial.suggest_uniform("gamma", 0, 3)
            # }
            params = {
                # cat_idxs=cat_idxs,
                # cat_emb_dim=1,
                "n_d": trial.suggest_categorical("n_d", [16]),
                "n_a": trial.suggest_categorical("n_a", [16]),
                "n_steps": trial.suggest_categorical("n_steps", [2]),
                "gamma": trial.suggest_categorical("gamma", [1.4690246460970766]),
                "n_independent": trial.suggest_categorical("n_independent", [9]),
                "n_shared": trial.suggest_categorical("n_shared", [4]),
                "lambda_sparse": trial.suggest_categorical("lambda_sparse", [0]),
                "optimizer_fn": trial.suggest_categorical("optimizer_fn", [Adam]),
                "optimizer_params": trial.suggest_categorical("optimizer_params", [dict(lr = (0.024907164557092944))]),
                "mask_type": trial.suggest_categorical("mask_type", ["entmax"]),
                "scheduler_params": trial.suggest_categorical("scheduler_params", [dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False)]),
                "scheduler_fn": trial.suggest_categorical("scheduler_fn", [CosineAnnealingWarmRestarts]),
                "seed": trial.suggest_categorical("seed", [42]),
                "verbose": trial.suggest_categorical("verbose", [10]),
                # n_d = 16,
                # n_a = 16,
                # n_steps = 2,
                # gamma = 1.4690246460970766,
                # n_independent = 9,
                # n_shared = 4,
                # lambda_sparse = 0,
                # optimizer_fn = Adam,
                # optimizer_params = dict(lr = (0.024907164557092944)),
                # mask_type = "entmax",
                # scheduler_params = dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
                # scheduler_fn = CosineAnnealingWarmRestarts,
                # seed = 42,
                # verbose = 10, 
            } 
            
            return params 
    
        if model_name == "tabnetr":
            # https://www.kaggle.com/code/wangqihanginthesky/baseline-tabnet

            # params = {
            #     "gamma": trial.suggest_uniform("gamma", 0, 3)
            # }
            params = {
                # cat_idxs=cat_idxs,
                # cat_emb_dim=1,
                "n_d": trial.suggest_categorical("n_d", [16]),
                "n_a": trial.suggest_categorical("n_a", [16]),
                "n_steps": trial.suggest_categorical("n_steps", [2]),
                #"n_steps": trial.suggest_categorical("n_steps", [2, 4, 7]),
                "gamma": trial.suggest_categorical("gamma", [1.4690246460970766]),
                #"gamma": trial.suggest_uniform("gamma", 0.1, 2),
                "n_independent": trial.suggest_categorical("n_independent", [9]),
                "n_shared": trial.suggest_categorical("n_shared", [4]),
                "lambda_sparse": trial.suggest_categorical("lambda_sparse", [0]),
                "optimizer_fn": trial.suggest_categorical("optimizer_fn", [Adam]),
                "optimizer_params": trial.suggest_categorical("optimizer_params", [dict(lr = (0.024907164557092944))]),
                "mask_type": trial.suggest_categorical("mask_type", ["entmax"]),
                "scheduler_params": trial.suggest_categorical("scheduler_params", [dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False)]),
                "scheduler_fn": trial.suggest_categorical("scheduler_fn", [CosineAnnealingWarmRestarts]),
                "seed": trial.suggest_categorical("seed", [42]),
                "verbose": trial.suggest_categorical("verbose", [10]),
                # n_d = 16,
                # n_a = 16,
                # n_steps = 2,
                # gamma = 1.4690246460970766,
                # n_independent = 9,
                # n_shared = 4,
                # lambda_sparse = 0,
                # optimizer_fn = Adam,
                # optimizer_params = dict(lr = (0.024907164557092944)),
                # mask_type = "entmax",
                # scheduler_params = dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
                # scheduler_fn = CosineAnnealingWarmRestarts,
                # seed = 42,
                # verbose = 10, 
            }
            
            return params 

        if model_name == "k1":
            # params = {
            #     "epochs": trial.suggest_int("epochs", 10, 55, step=5, log=False),  # 5,55
            #     "batchsize": trial.suggest_int("batchsize", 8, 40, step=16, log=False),
            #     "learning_rate": trial.suggest_uniform("learning_rate", 0, 3),
            #     "o": trial.suggest_categorical("o", [True, False]),
            # }
            params = {
                "epochs": trial.suggest_int("epochs", 2, 55, step=5, log=False),  # 5,55
                "batchsize": trial.suggest_int("batchsize", 8, 40, step=16, log=False),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 3),
                "o": trial.suggest_categorical("o", [True, False]),
            }
            return params

        if model_name == "k2":
            params = {
                "epochs": trial.suggest_int("epochs", 5, 55, step=5, log=False),  # 5,55
                "batchsize": trial.suggest_int("batchsize", 8, 40, step=16, log=False),
                "learning_rate": trial.suggest_uniform("learning_rate", 0, 3),
                "prime": trial.suggest_categorical(
                    "prime", [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
                ),
                "drop_val": trial.suggest_uniform("drop", 0.01, 0.9),
                "o": trial.suggest_categorical("o", [True, False]),
            }
            return params

        if model_name == "k3":
            no_hidden_layers = trial.suggest_int("no_hidden_layers", 1, 30, step=1)
            dropout_placeholder = [
                trial.suggest_uniform(f"drop_rate_{i+1}", 0, 0.99)
                for i in range(no_hidden_layers)
            ]
            units_placeholder = [
                trial.suggest_int(
                    f"units_val_{j+1}",
                    int((self.xtrain.shape[1] + 1) ** (1 / 3)),
                    int((self.xtrain.shape[1] + 1) ** 3),
                    step=1,
                    log=False,
                )
                for j in range(no_hidden_layers)
            ]
            batch_norm_placeholder = [
                trial.suggest_categorical(f"batch_norm_val_{i+1}", [0, 1])
                for i in range(no_hidden_layers)
            ]
            activation_placeholder = [
                trial.suggest_categorical(
                    f"activation_string_{i+1}", ["relu", "sigmoid", "tanh"]
                )
                for i in range(no_hidden_layers)
            ]

            epochs = trial.suggest_int("epochs", 2, 3, step=5, log=False) # 5,55
            batchsize = trial.suggest_int("batchsize", 8, 40, step=16, log=False)
            learning_rate = trial.suggest_uniform("learning_rate", 0, 3)
            # epochs = trial.suggest_int("epochs", 5, 100, step=5, log=False)  # 5,55
            # batchsize = trial.suggest_int("batchsize", 8, 40, step=16, log=False)
            # learning_rate = trial.suggest_uniform("learning_rate", 0, 3)
            o = trial.suggest_categorical("o", [True, False])
            params = {
                "no_hidden_layers": no_hidden_layers,
                "dropout_placeholder": dropout_placeholder,
                "units_placeholder": units_placeholder,
                "batch_norm_placeholder": batch_norm_placeholder,
                "activation_placeholder": activation_placeholder,
                "epochs": epochs,
                "batchsize": batchsize,
                "learning_rate": learning_rate,
                "o": o,
            }
            return params

        if model_name == "tez1":
            # -batch_size = 16
            # -epochs = 5
            # =====seed = 42
            # ===target_size = 28
            # -learning_rate = 0.002

            params = {
                "batch_size": trial.suggest_categorical(
                    "batch_size", [16]
                ),  # ,32,64, 128,256, 512]),
                # "epochs": trial.suggest_int(
                #     "epochs", 1, 2, step=1, log=False
                # ),  # 55, step=5, log=False),  # 5,55
                "epochs": trial.suggest_categorical("epochs", [5]),
                "learning_rate": trial.suggest_uniform("learning_rate", 2, 8),
                "patience": trial.suggest_categorical("patience", [5]),
            }
            return params

        if model_name == "tez2":
            # -batch_size = 16
            # -epochs = 5
            # =====seed = 42
            # ===target_size = 28
            # -learning_rate = 0.002

            params = {
                "batch_size": trial.suggest_categorical(
                    "batch_size", [16]
                ),  # ,32,64, 128,256, 512]),
                # "epochs": trial.suggest_int(
                #     "epochs", 1, 2, step=1, log=False
                # ),  # 55, step=5, log=False),  # 5,55
                "epochs": trial.suggest_categorical("epochs", [1]),
                "learning_rate": trial.suggest_uniform("learning_rate", 2, 8),
                "patience": trial.suggest_categorical("patience", [5]),
            }
            return params

        if model_name == "p1":
            # -batch_size = 16
            # -epochs = 5
            # =====seed = 42
            # ===target_size = 28
            # -learning_rate = 0.002

            params = {
                "batch_size": trial.suggest_categorical(
                    "batch_size", [16, 32, 128, 512]
                ),  # ,32,64, 128,256, 512]),
                "epochs": trial.suggest_int(
                    "epochs", 20, 55, step=10, log=False
                ),  # 55, step=5, log=False),  # 5,55
                # "epochs": trial.suggest_categorical("epochs", [1]),
                "learning_rate": trial.suggest_uniform("learning_rate", 1, 8),
                "patience": trial.suggest_categorical("patience", [3, 5]),
                "momentum": trial.suggest_uniform("momentum", 0.2, 0.9),
            }

            # Demo
            # params = {
            #     "batch_size": trial.suggest_categorical(
            #         "batch_size", [16, 32, 128, 512]
            #     ),  # ,32,64, 128,256, 512]),
            #     "epochs": trial.suggest_int(
            #         "epochs", 1,3, step=1, log=False
            #     ),  # 55, step=5, log=False),  # 5,55
            #     # "epochs": trial.suggest_categorical("epochs", [1]),
            #     "learning_rate": trial.suggest_uniform("learning_rate", 1, 8),
            #     "patience": trial.suggest_categorical("patience", [3, 5]),
            #     "momentum": trial.suggest_uniform("momentum", 0.2, 0.9),
            # }
            return params

        if model_name == "pretrained":
            # -batch_size = 16
            # -epochs = 5
            # =====seed = 42
            # ===target_size = 28
            # -learning_rate = 0.002

            # params = {
            #     "batch_size": trial.suggest_categorical(
            #         "batch_size", [16, 32, 128, 512]
            #     ),  # ,32,64, 128,256, 512]),
            #     "epochs": trial.suggest_int(
            #         "epochs", 12,25, step=5, log=False
            #     ),  # 55, step=5, log=False),  # 5,55
            #     #"epochs": trial.suggest_categorical("epochs", [1]),
            #     "learning_rate": trial.suggest_uniform("learning_rate", 1, 8),
            #     "patience": trial.suggest_categorical("patience", [3,5]),
            #     "momentum": trial.suggest_uniform("momentum", 0.2, 0.9)
            # }

            # Demo
            params = {
                "batch_size": trial.suggest_categorical(
                    "batch_size", [16, 32, 128, 512]
                ),  # ,32,64, 128,256, 512]),
                "epochs": trial.suggest_int(
                    "epochs", 1, 2, step=1, log=False
                ),  # 55, step=5, log=False),  # 5,55
                # "epochs": trial.suggest_categorical("epochs", [1]),
                "learning_rate": trial.suggest_uniform("learning_rate", 1, 8),
                "patience": trial.suggest_categorical("patience", [3, 5]),
                "momentum": trial.suggest_uniform("momentum", 0.2, 0.9),
            }

            print("finding params")
            print(params)
            return params

        if model_name == "keras":  # demo
            self.Table = pd.DataFrame(
                columns=[
                    "val_score",
                    "lr_modified",
                    "learning_rate",
                    "epochs",
                    "batch_size",
                    "no_hidden_layers",
                    "dropout_placeholder",
                    "units_placeholder",
                    "batch_norm_placeholder",
                    " activation_placeholder",
                ]
            )
            return params

    def get_model(self, params):

        # ["lgr","lir","xgb","xgbc","xgbr"]
        model_name = self.model_name
        self._random_state = self.generate_random_no()
        if model_name == "lgr":
            return LogisticRegression(**params, random_state=self._random_state)
        if model_name == "lir":
            return LinearRegression() #**params, random_state=self._random_state)
        if model_name == "xgb":
            return "No Model Yet"
        if model_name == "xgbc":
            return XGBClassifier(**params, random_state=self._random_state)
        if model_name == "xgbr":
            return XGBRegressor(**params, random_state=self._random_state)
        if model_name == "mlpc":
            return MLPClassifier(**params, random_state=self._random_state)
        if model_name == "mlpr":
            return MLPRegressor(**params, random_state=self._random_state)
        if model_name == "rg":
            return Ridge(**params, random_state=self._random_state)
        if model_name == "ls":
            return Lasso(**params, random_state=self._random_state)
        if model_name == "knnc":
            return KNeighborsClassifier(**params)
        if model_name == "cbc":
            # User defined loss functions, metrics and callbacks are not supported for GPU
            return CatBoostClassifier(**params, random_state=self._random_state) #, eval_metric = CustomMetric_cbc())
        
        if model_name == "cbr":
            # User defined loss functions, metrics and callbacks are not supported for GPU
            return CatBoostRegressor(**params, random_state=self._random_state) #, eval_metric = CustomMetric_cbc())
        

        if model_name == "dtc":
            return DecisionTreeClassifier(**params, random_state=self._random_state)
        if model_name == "adbc":
            return AdaBoostClassifier(**params, random_state=self._random_state)
        if model_name == "gbmc":
            #return HistGradientBoostingClassifier(**params, random_state=self._random_state)
            return GradientBoostingClassifier(**params, random_state=self._random_state)
        if model_name == "gbmr":
            #return HistGradientBoostingRegressor(**params, random_state=self._random_state)
            return GradientBoostingRegressor(**params, random_state=self._random_state)

        if model_name == "hgbc":
            return HistGradientBoostingClassifier(
                **params, random_state=self._random_state
            )
        if model_name == "lgb":
            return "No model yet"
        if model_name == "lgbmc":
            return LGBMClassifier(**params, random_state=self._random_state)
        if model_name == "lgbmr":
            return LGBMRegressor(**params, random_state=self._random_state)
        if model_name == "rfc":
            return RandomForestClassifier(**params, random_state=self._random_state)
        if model_name == "rfr":
            return RandomForestRegressor(**params, random_state=self._random_state)   
        if model_name == "tabnetc":
            return TabNetClassifier(**params)
        if model_name == "tabnetr":
            return TabNetRegressor(**params)     
        if model_name == "k1":
            return self._k1(params, random_state=self._random_state)
        if model_name == "k2":
            return self._k2(params, random_state=self._random_state)
        if model_name == "k3":
            return self._k3(params, random_state=self._random_state)
        if model_name == "k4":
            return self._k4(params, random_state=self._random_state)
        if model_name == "tez1":
            return self._tez1(params, random_state=self._random_state)
        if model_name == "tez2":
            return self._tez2(params, random_state=self._random_state)
        if model_name == "p1":  # pytorch1
            # basic pytorch model
            return self._p1(params=params)
        if model_name == "pretrained":  # pytorch
            return self._pretrained(params=params)
        else:
            raise Exception(f"{model_name} is invalid!")

    def _pretrained(self, params):
        self.learning_rate = 10 ** (-1 * params["learning_rate"])
        # model = pretrained_models(len(self.filtered_features))
        model = MODEL_DISPATCHER["resnet34"](
            pretrained=False
        )  # supports all image size
        model.to("cuda")
        # train_loader
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=params["batch_size"],
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=params["batch_size"],
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=params["batch_size"],
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=10 ** (-1 * params["learning_rate"]),
            momentum=params["momentum"],  # 0.9,
        )
        # base_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        # optimizer = SWA(base_opt, swa_start=10, swa_freq=2, swa_lr=0.0005)

        # in torch some scheduler step after every epoch,
        # some after every batch
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=params["patience"],
            verbose=True,
            mode="max",
            threshold=1e-4,
        )
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        return trainer_p1(
            model,
            self.train_loader,
            self.valid_loader,
            optimizer,
            scheduler,
            self.use_cutmix,
        )

    def _p1(self, params=0, random_state=0):
        self.learning_rate = 10 ** (-1 * params["learning_rate"])
        model = p1_model()
        model.to("cuda")
        # train_loader
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=params["batch_size"],
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=params["batch_size"],
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=params["batch_size"],
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=10 ** (-1 * params["learning_rate"]),
            momentum=params["momentum"],  # 0.9,
        )
        # base_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        # optimizer = SWA(base_opt, swa_start=10, swa_freq=2, swa_lr=0.0005)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=params["patience"],
            verbose=True,
            mode="max",
            threshold=1e-4,
        )
        return trainer_p1(
            model,
            self.train_loader,
            self.valid_loader,
            optimizer,
            scheduler,
            self.use_cutmix,
        )

    def _tez1(self, params, random_state):
        """
        self.train_image_paths
        self.valid_image_paths
        self.train_dataset
        selt.valid_dataset
        """
        print("params of tez1")
        print(params)
        print("=" * 40)
        batch_size = params["batch_size"]
        epochs = params["epochs"]
        learning_rate = 10 ** (
            -1 * params["learning_rate"]
        )  # params["learning_rate"]  #

        img_size = 256  # nothing to do with model used for naming output files
        model_name = "resnet50"
        seed = random_state
        target_size = len(set(self.ytrain))
        n_train_steps = int(len(self.train_dataset) / batch_size * epochs)

        model = UModel(
            model_name=model_name,
            num_classes=target_size,
            learning_rate=learning_rate,
            n_train_steps=n_train_steps,
        )

        return model

    def _tez2(self, params, random_state):
        """
        self.train_image_paths
        self.valid_image_paths
        self.train_dataset
        selt.valid_dataset
        """
        print("params of tez2")
        print(params)
        print("=" * 40)
        batch_size = params["batch_size"]
        epochs = params["epochs"]
        learning_rate = 10 ** (
            -1 * params["learning_rate"]
        )  # params["learning_rate"]  #

        img_size = 256  # nothing to do with model used for naming output files
        model_name = "resnet50"
        seed = random_state
        target_size = len(set(self.ytrain))
        # n_train_steps = int(len(self.train_image_paths) / batch_size * epochs)
        n_train_steps = int(len(self.train_dataset) / batch_size / 1 * epochs)
        model = DigitRecognizerModel(
            model_name="resnet50",
            num_classes=target_size,
            learning_rate=learning_rate,
            n_train_steps=n_train_steps,
        )
        model = Tez(model)
        return model

    def _k1(self, params, random_state):
        # simple model
        model = keras.Sequential()
        model.add(BatchNormalization())
        model.add(Dense(132, activation="relu"))

        model.add(BatchNormalization())
        model.add(Dense(32, activation="relu"))

        model.add(BatchNormalization())
        model.add(Dense(32, activation="relu"))

        model.add(BatchNormalization())
        model.add(Dense(32, activation="relu"))
        
        model.add(BatchNormalization())
        model.add(Dense(32, activation="relu"))

        # model.add(Dense(32, activation="relu"))
        # #model.add(Dropout(p=0.2))
        # model.add(Dense(32, activation="relu"))
        # model.add(BatchNormalization())
        # model.add(Dense(32, activation="relu"))
        # #model.add(Dropout(p=0.2))
        # model.add(BatchNormalization())
        # model.add(Dense(32, activation="relu"))
        # model.add(BatchNormalization())
        # model.add(Dense(16, activation="relu"))

        model.add(BatchNormalization())
        """
        model = keras.Sequential()
        model.add(BatchNormalization())
        model.add(Dense(32, activation="relu"))

        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(16, activation="relu"))

        model.add(BatchNormalization())
        """
        # adam is used when we don't have custom lr_scheduler
        # any(x in ['b', 'd', 'foo', 'bar'] for x in ['a', 'b']): True
        # all(x in ['b', 'd', 'foo', 'bar'] for x in ['a', 'b']): False
        if any(
            x in self.callbacks_list
            for x in ["simple_decay", "cosine_decay", "exponential_decay"]
        ):
            # time for custom lr
            lr_start = 1e-2 # 0.01
            lr_start = 10 ** (
                -1 * params["learning_rate"]
            )  # optimize starting point using optuna
            global_variables.lr_start = lr_start
            opt = Adamax(learning_rate=lr_start)
        else:
            adam = tf.keras.optimizers.Adam(
                learning_rate=10 ** (-1 * params["learning_rate"])
            )
            opt = adam

        # PREDICT: gives probability always so in case of metrics which takes hard class do (argmax)
        # For #class more than 2 output label has multiple node:
        # confusion remains with 2class problem as it can have both one node or 2 node in end
        if self.comp_type == "regression":
            model.add(Dense(1, activation="linear"))  # /None  linear tanh
            # model.compile(
            #     loss=self.metrics_name,
            #     optimizer=Adamax(learning_rate=lr_start),  # adam,
            #     metrics=[tf.keras.metrics.MeanSquaredError()],
            # )
            model.compile(optimizer=opt, loss = 'mae', metrics=['mean_squared_error'])
        elif self.comp_type == "2class":
            if len(self.ytrain.shape) == 1:
                model.add(Dense(1, activation="sigmoid"))  #: binary_crossentropy

                #--> https://www.kaggle.com/code/cdeotte/tensorflow-gru-starter-0-790
                #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
                loss = tf.keras.losses.BinaryCrossentropy()
                model.compile(loss=loss, optimizer = opt)

                # model.compile(
                #     optimizer=opt, metrics= [amex_metric_tensorflow], #["accuracy"],["accuracy"]
                #     loss=tf.keras.losses.BinaryCrossentropy(),
                # ) #loss="binary_crossentropy",
                print("New metrics implemented1")
            else:
                # binary with one hot y no more binary problem so it is like multi class := don't use this case use above one instead
                model.add(Dense(self.ytrain.shape[1], activation="softmax"))
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=opt,
                    metrics= [amex_metric], #["accuracy"],
                )
                print("New metrics implemented")
        elif self.comp_type == "multi_class":
            # https://medium.com/deep-learning-with-keras/which-activation-loss-functions-in-multi-class-clasification-4cd599e4e61f
            if len(self.ytrain.shape) == 1:  # sparse  since true prediction is 1D
                # op1>
                # act:None
                # loss:keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                # metrics:metrics=[keras.metrics.SparseCategoricalAccuracy()]
                # op2>
                # act: softmax
                # loss: sparse_categorical_crossentropy
                # metrics: keras.metrics.SparseCategoricalAccuracy()
                if params["o"]:
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                    metrics = [keras.metrics.SparseCategoricalAccuracy()]
                    act = None
                else:
                    loss = keras.losses.SparseCategoricalCrossentropy()
                    metrics = [keras.metrics.SparseCategoricalAccuracy()]
                    act = "softmax"
                model.add(Dense(self.ytrain.shape[1], activation=act))
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
            else:  # ytrain is not 1D
                # op1>
                # act:None
                # loss:keras.losses.CategoricalCrossentropy(from_logits=True)
                # metrics:metrics=[keras.metrics.CategoricalAccuracy()]
                # op2>
                # act: softmax
                # loss: categorical_crossentropy
                # metrics: keras.metrics.CategoricalAccuracy()
                if params["o"]:
                    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
                    metrics = [keras.metrics.CategoricalAccuracy()]
                    act = None
                else:
                    loss = keras.losses.CategoricalCrossentropy()
                    metrics = [keras.metrics.CategoricalAccuracy()]
                    act = "softmax"
                model.add(Dense(self.ytrain.shape[1], activation="softmax"))
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
        elif self.comp_type == "multi_label":
            model.add(Dense(self.ytrain.shape[1], activation="sigmoid"))

            model.compile(
                loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )

        return model

    def _k2(self, params, random_state):
        # cone model
        model = Sequential()
        model.add(BatchNormalization())
        no_cols = self.xtrain.shape[1]
        model.add(Dense(2 * no_cols, activation="relu"))

        while no_cols > 2 * 10:
            model.add(Dropout(params["drop_val"]))
            model.add(Dense(no_cols, activation="relu"))
            model.add(Dense(no_cols, activation="relu"))
            model.add(BatchNormalization())
            no_cols = int(no_cols // params["prime"])

        adam = tf.keras.optimizers.Adam(
            learning_rate=10 ** (-1 * params["learning_rate"])
        )
        # PREDICT: gives probability always so in case of metrics which takes hard class do (argmax)
        # For #class more than 2 output label has multiple node:
        # confusion remains with 2class problem as it can have both one node or 2 node in end
        if self.comp_type == "regression":
            model.add(Dense(1, activation="relu"))  # /None  linear tanh
            model.compile(
                loss=self.metrics_name,
                optimizer=adam,
                metrics=[tf.keras.metrics.MeanSquaredError()],
            )
        elif self.comp_type == "2class":
            if len(self.ytrain.shape) == 1:
                model.add(Dense(1, activation="sigmoid"))  #: binary_crossentropy

                #--> https://www.kaggle.com/code/cdeotte/tensorflow-gru-starter-0-790
                #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
                loss = tf.keras.losses.BinaryCrossentropy()
                model.compile(loss=loss, optimizer = opt)

                # model.compile(
                #     optimizer=opt, metrics= [amex_metric_tensorflow], #["accuracy"],["accuracy"]
                #     loss=tf.keras.losses.BinaryCrossentropy(),
                # ) #loss="binary_crossentropy",
                print("New metrics implemented1")
            else:
                # binary with one hot y no more binary problem so it is like multi class := don't use this case use above one instead
                model.add(Dense(self.ytrain.shape[1], activation="softmax"))
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=opt,
                    metrics= [amex_metric], #["accuracy"],
                )
                print("New metrics implemented")
        elif self.comp_type == "multi_class":
            # https://medium.com/deep-learning-with-keras/which-activation-loss-functions-in-multi-class-clasification-4cd599e4e61f
            if len(self.ytrain.shape) == 1:  # sparse  since true prediction is 1D
                # op1>
                # act:None
                # loss:keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                # metrics:metrics=[keras.metrics.SparseCategoricalAccuracy()]
                # op2>
                # act: softmax
                # loss: sparse_categorical_crossentropy
                # metrics: keras.metrics.SparseCategoricalAccuracy()
                if params["o"]:
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                    metrics = [keras.metrics.SparseCategoricalAccuracy()]
                    act = None
                else:
                    loss = keras.losses.SparseCategoricalCrossentropy()
                    metrics = [keras.metrics.SparseCategoricalAccuracy()]
                    act = "softmax"
                model.add(Dense(self.ytrain.shape[1], activation=act))
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
            else:  # ytrain is not 1D
                # op1>
                # act:None
                # loss:keras.losses.CategoricalCrossentropy(from_logits=True)
                # metrics:metrics=[keras.metrics.CategoricalAccuracy()]
                # op2>
                # act: softmax
                # loss: categorical_crossentropy
                # metrics: keras.metrics.CategoricalAccuracy()
                if params["o"]:
                    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
                    metrics = [keras.metrics.CategoricalAccuracy()]
                    act = None
                else:
                    loss = keras.losses.CategoricalCrossentropy()
                    metrics = [keras.metrics.CategoricalAccuracy()]
                    act = "softmax"
                model.add(Dense(self.ytrain.shape[1], activation="softmax"))
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
        elif self.comp_type == "multi_label":
            model.add(Dense(self.ytrain.shape[1], activation="sigmoid"))

            model.compile(
                loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )

        return model

    def _k3(self, params, random_state):
        # cone model
        model = Sequential()
        model.add(BatchNormalization())

        for i in range(params["no_hidden_layers"]):
            if params["batch_norm_placeholder"][i] == 1:
                model.add(BatchNormalization())
            model.add(
                Dense(
                    units=params["units_placeholder"][i],
                    activation=params["activation_placeholder"][i],
                )
            )
            model.add(Dropout(params["dropout_placeholder"][i]))
            gc.collect()

        adam = tf.keras.optimizers.Adam(
            learning_rate=10 ** (-1 * params["learning_rate"])
        )
        # PREDICT: gives probability always so in case of metrics which takes hard class do (argmax)
        # For #class more than 2 output label has multiple node:
        # confusion remains with 2class problem as it can have both one node or 2 node in end
        if self.comp_type == "regression":
            model.add(Dense(1, activation="relu"))  # /None  linear tanh
            model.compile(
                loss=self.metrics_name,
                optimizer=adam,
                metrics=[tf.keras.metrics.MeanSquaredError()],
            )
        elif self.comp_type == "2class":
            if len(self.ytrain.shape) == 1:
                model.add(Dense(1, activation="sigmoid"))  #: binary_crossentropy

                #--> https://www.kaggle.com/code/cdeotte/tensorflow-gru-starter-0-790
                #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
                loss = tf.keras.losses.BinaryCrossentropy()
                model.compile(loss=loss, optimizer = opt)

                # model.compile(
                #     optimizer=opt, metrics= [amex_metric_tensorflow], #["accuracy"],["accuracy"]
                #     loss=tf.keras.losses.BinaryCrossentropy(),
                # ) #loss="binary_crossentropy",
                print("New metrics implemented1")
            else:
                # binary with one hot y no more binary problem so it is like multi class := don't use this case use above one instead
                model.add(Dense(self.ytrain.shape[1], activation="softmax"))
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=opt,
                    metrics= [amex_metric], #["accuracy"],
                )
                print("New metrics implemented")
        elif self.comp_type == "multi_class":
            # https://medium.com/deep-learning-with-keras/which-activation-loss-functions-in-multi-class-clasification-4cd599e4e61f
            if len(self.ytrain.shape) == 1:  # sparse  since true prediction is 1D
                # op1>
                # act:None
                # loss:keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                # metrics:metrics=[keras.metrics.SparseCategoricalAccuracy()]
                # op2>
                # act: softmax
                # loss: sparse_categorical_crossentropy
                # metrics: keras.metrics.SparseCategoricalAccuracy()
                if params["o"]:
                    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                    metrics = [keras.metrics.SparseCategoricalAccuracy()]
                    act = None
                else:
                    loss = keras.losses.SparseCategoricalCrossentropy()
                    metrics = [keras.metrics.SparseCategoricalAccuracy()]
                    act = "softmax"
                model.add(Dense(self.ytrain.shape[1], activation=act))
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
            else:  # ytrain is not 1D
                # op1>
                # act:None
                # loss:keras.losses.CategoricalCrossentropy(from_logits=True)
                # metrics:metrics=[keras.metrics.CategoricalAccuracy()]
                # op2>
                # act: softmax
                # loss: categorical_crossentropy
                # metrics: keras.metrics.CategoricalAccuracy()
                if params["o"]:
                    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
                    metrics = [keras.metrics.CategoricalAccuracy()]
                    act = None
                else:
                    loss = keras.losses.CategoricalCrossentropy()
                    metrics = [keras.metrics.CategoricalAccuracy()]
                    act = "softmax"
                model.add(Dense(self.ytrain.shape[1], activation="softmax"))
                model.compile(loss=loss, optimizer=opt, metrics=metrics)
        elif self.comp_type == "multi_label":
            model.add(Dense(self.ytrain.shape[1], activation="sigmoid"))

            model.compile(
                loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
            )

        return model

    # DLastStark
    # https://www.kaggle.com/code/dlaststark/tps-may22-what-tf-again
    def _k4(self, params, random_state):

        x_input = Input(shape=(len(self.useful_features),))

        xi = Dense(units=384, activation="swish", kernel_initializer="lecun_normal")(
            x_input
        )
        xi = BatchNormalization()(xi)
        xi = Dropout(rate=0.25)(xi)

        x = Reshape((16, 24))(xi)

        x = Conv1D(
            filters=48,
            activation="swish",
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer="lecun_normal",
        )(x)
        x = BatchNormalization()(x)

        x1 = Conv1D(
            filters=96,
            activation="swish",
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer="lecun_normal",
        )(x)
        x1 = BatchNormalization()(x1)

        x2 = Conv1D(
            filters=96,
            activation="swish",
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer="lecun_normal",
        )(x1)
        x2 = BatchNormalization()(x2)

        x2 = Conv1D(
            filters=96,
            activation="swish",
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer="lecun_normal",
        )(x2)
        x2 = BatchNormalization()(x2)

        x = Add()([x1, x2])

        x = Conv1D(
            filters=96,
            activation="swish",
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer="lecun_normal",
        )(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Add()([x, xi])

        x = Dense(units=192, activation="swish", kernel_initializer="lecun_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.3)(x)

        x = Dense(units=96, activation="swish", kernel_initializer="lecun_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.2)(x)

        x_output = Dense(
            units=1, activation="sigmoid", kernel_initializer="lecun_normal"
        )(x)

        model = Model(inputs=x_input, outputs=x_output, name="TPS_May22_TF_Model")

        model.compile(
            optimizer=Adamax(learning_rate=lr_start),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )

        return model

    def save_logs(self, params):
        print("Saving Log Table")
        if self._log_table is None:
            # not initialized
            self._log_table = pd.DataFrame(
                columns=["trial_score"] + list(params.keys()) + ["keras_history"]
            )
        self._log_table.loc[self._log_table.shape[0], :] = (
            [self._trial_score] + list(params.values()) + [self._history]
        )

    def get_callbacks(self, params, verbose):
        ###############################################################################
        #                         Tree based models
        ############################################################################
        # lgb/lgbmc
        # https://lightgbm.readthedocs.io/en/latest/Python-API.html#callbacks 
        """
        early_stopping(stopping_rounds[, ...])      Create a callback that activates early stopping.
        log_evaluation([period, show_stdv])         Create a callback that logs the evaluation results.
        record_evaluation(eval_result)              Create a callback that records the evaluation history into eval_result.
        reset_parameter(**kwargs)                   Create a callback that resets the parameter after the first iteration.
        """



        ###############################################################################
        #                         KERAS
        ############################################################################
        # DLastStark
        # https://www.kaggle.com/code/dlaststark/tps-may22-what-tf-again
        # ambrosm
        # https://www.kaggle.com/code/ambrosm/amex-keras-quickstart-1-training/notebook
        # lr_start = 1e-2 set equal to lr # It is set while defining model K1
        global_variables.lr_end = 1e-5  # 1e-4
        global_variables.epochs = params["epochs"]

        callbacks = [TerminateOnNaN()]  # always keep TerminateOnNan()
        if "chk_pt" in self.callbacks_list:
            # https://keras.io/api/callbacks/model_checkpoint/
            # chk_point = ModelCheckpoint(
            #     filepath=f"../models/models-{self.locker['comp_name']}/",  #  "./",  # to work on this part
            #     save_weights_only=True,
            #     monitor="val_accuracy",
            #     mode="max",
            #     save_best_only=True,
            # )
            # https://www.kaggle.com/code/dlaststark/tps-may22-what-tf-again
            chk_point = ModelCheckpoint(
                f"../models/models-{self.locker['comp_name']}/model_exp_{self.current_dict['current_exp_no']}.h5",
                monitor="val_auc",
                verbose=verbose,
                save_best_only=True,
                mode="max",
            )
            callbacks.append(chk_pt)
        if "ReduceLROnPlateau" in self.callbacks_list:
            reduce_lr = ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=verbose,
            )
            callbacks.append(reduce_lr)
        if "early_stopping" in self.callbacks_list:
            stop = EarlyStopping(
                monitor="accuracy", mode="max", patience=50
            )  # , verbose=1)
            callbacks.append(stop)

        if "cosine_decay" in self.callbacks_list:
            lr = LearningRateScheduler(cosine_decay, verbose=verbose)
            callbacks.append(lr)
        if "exponential_decay" in self.callbacks_list:
            print("entered exponential_decay")
            lr = LearningRateScheduler(exponential_decay, verbose=verbose)
            callbacks.append(lr)
        if "simple_decay" in self.callbacks_list:
            lr = LearningRateScheduler(simple_decay, verbose=verbose)
            callbacks.append(lr)
        if "swa" in self.callbacks_list:
            # https://www.kaggle.com/competitions/google-quest-challenge/discussion/119371
            # https://pypi.org/project/keras-swa/
            """
            # 'manual' , 'constant' or 'cyclic'
             The default schedule is 'manual', 
            allowing the learning rate to be controlled by an external learning rate scheduler or the optimizer. 
                start_epoch - Starting epoch for SWA.
                lr_schedule - Learning rate schedule. 'manual' , 'constant' or 'cyclic'.
                swa_lr - Learning rate used when averaging weights.
                swa_lr2 - Upper bound of learning rate for the cyclic schedule.
                swa_freq - Frequency of weight averagining. Used with cyclic schedules.
                batch_size - Batch size model is being trained with (only when using batch normalization).
                verbose - Verbosity mode, 0 or 1.
            """
            from swa.keras import SWA
            #Define when to start SWA
            start_epoch = 5
            # define swa callback
            swa = SWA(start_epoch=start_epoch, 
                    lr_schedule='constant', #cyclic 'manual' , 'constant' or 'cyclic'
                    swa_lr=1e-5,
                    #batch_size = 256 ,
                    #swa_lr2=9e-5,
                    # swa_freq = 10,
                    verbose=1)
        return callbacks


    def obj(self, trial):
        if self._state == "seed" or self._state == "fold":
            params = self.params
        else:
            params = self.get_params(trial)
            print("Current params:")
            print(params)
        model = self.get_model(params)

        if self._state == "seed":
            # There must be some other better ways
            self.xvalid = self.xtrain 
            self.yvalid = self.ytrain 
        # fit xtrain
        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        # lgb/lgbmc
        # https://lightgbm.readthedocs.io/en/latest/Python-API.html#callbacks 
        """
        early_stopping(stopping_rounds[, ...])      Create a callback that activates early stopping.
        log_evaluation([period, show_stdv])         Create a callback that logs the evaluation results.
        record_evaluation(eval_result)              Create a callback that records the evaluation history into eval_result.
        reset_parameter(**kwargs)                   Create a callback that resets the parameter after the first iteration.
        """

        if self.model_name == "lgb": # done but add callback
            # dart 

            lgb_train = lgb.Dataset(self.xtrain, self.ytrain ) #, categorical_feature = cat_features)
            lgb_valid = lgb.Dataset(self.xvalid, self.yvalid ) #, categorical_feature = cat_features)

            """
            model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 10500,
            valid_sets = [lgb_train, lgb_valid],
            early_stopping_rounds = 100,
            verbose_eval = 500,
            feval = lgb_amex_metric
            )
            # Save best model
            joblib.dump(model, f"../models/models-{self.locker['comp_name']}/model_exp_{self.current_dict['current_exp_no']}.pkl")
            """

            # https://www.kaggle.com/competitions/amex-default-prediction/discussion/332575
            global_variables.fold = self.optimize_on 
            global_variables._state = self._state
            # sometimes we may want to predict an old experiment so don't use current exp no, set self.exp_no
            # counter stores no of times objective function is run
            path = f"../models/models-{self.comp_name}/callback_logs/lgb_models_e_{self.exp_no}_f_{global_variables.counter}_{global_variables._state}/"
            mkdir_from_path(path)

            n_rounds = 5000#5000
            model = lgb.train(params, 
                            train_set = lgb_train, 
                            num_boost_round= n_rounds, 
                            valid_sets = [lgb_train,lgb_valid],
                            feval=feval_RMSPE,
                            verbose_eval= 250,
                            early_stopping_rounds=500
                            )

            # model = lgb.train(
            #     params = params,
            #     train_set = lgb_train,
            #     num_boost_round = 105, #10500, # 0, #10500
            #     valid_sets = [lgb_train,lgb_valid],
            #     #early_stopping_rounds = 1500, #100
            #     verbose_eval = 50,
            #     #feval = amzcomp1_metrics,
            #     #callbacks=[save_model1()],
            #     #callbacks=[save_model2(models_folder=pathlib.Path(path), fold_id=global_variables.counter, min_score_to_save=0.78, every_k=50)]
            # )

        elif self.model_name in ["lgbmr"]:  # , "lgbmc"]:
            model.fit( self.xtrain, self.ytrain, eval_set=[(self.xvalid, self.yvalid)],
                eval_metric="auc",
                early_stopping_rounds=1000,
                callbacks=[
                    LightGBMPruningCallback(trial, "auc")
                ],  # there is trial which creates issue when called from seed_it
                verbose=0,
            )
        elif self.model_name in ["lgbmc"]: # done
            # https://www.kaggle.com/code/ambrosm/amex-lightgbm-quickstart
            model.fit( 
                self.xtrain, 
                self.ytrain, 
                eval_set = [(self.xvalid, self.yvalid)],  
                # custom metrics: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
                eval_metric=[lgbmc_amex_metric], 
                callbacks= [log_evaluation(100)] 
                # save_model() can't be called from lgbmc but can be from lgb
                )
        elif self.model_name == "xgb": # not working
        
            # https://www.kaggle.com/competitions/amex-default-prediction/discussion/332575
            # global_variables.fold = self.optimize_on 
            # global_variables._state = self._state
            # global_variables.exp_no = self.current_dict['current_exp_no']
            # path = f"../models/models-{self.comp_name}/lgb_models_e_{global_variables.exp_no}_f_{global_variables.counter}_{global_variables._state}/"
            # mkdir_from_path(path)

            # https://www.kaggle.com/code/cdeotte/xgboost-starter-0-793
            # https://www.kaggle.com/code/sietseschrder/xgboost-starter-0-793
            # of chris deotte and one other person but not worked yet
            # it is a low level xgboost 
            # TRAIN, VALID, TEST FOR FOLD K
            # pass xtrain, ytrain as df
            # Xy_train = IterLoadForDMatrix(np.concatenate((self.xtrain, self.ytrain.reshape(-1,1)), axis=1))
            # dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256)
            # dvalid = xgb.DMatrix(data=self.xvalid, label=self.yvalid)
            
            # if we are using feature importance it is better to convert it to dataframe 
            if self.calculate_feature_importance:   
                self.xtrain = pd.DataFrame(self.xtrain, columns = self.useful_features)
                self.xvalid = pd.DataFrame(self.xvalid, columns = self.useful_features)
                self.ytrain = pd.DataFrame(self.ytrain, columns = [self.locker["target_name"]])
                self.yvalid = pd.DataFrame(self.yvalid, columns = [self.locker["target_name"]])




            # https://www.kaggle.com/code/thedevastator/ensemble-lightgbm-catboost-xgboost
            dtrain = xgb.DMatrix(data = self.xtrain, label = self.ytrain)
            dvalid = xgb.DMatrix(data = self.xvalid, label = self.yvalid)
            # TRAIN MODEL FOLD K
            watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
            model = xgb.train(params, 
                        dtrain=dtrain,
                        evals= watchlist, # it is same [(dtrain,'train'),(dvalid,'valid')],
                        custom_metric= xgboost_amex_metric_mod1, #getaroom_metrics
                        maximize = True,       # be very careful of this maximizes amex_metric
                        # num_boost_round= 9999, #3000, #9999, #9999,
                        # early_stopping_rounds= 5000, # 1000, #5000, #1000,
                        # verbose_eval= 500, #100,

                        num_boost_round= 6000, #3000, #9999, #9999,
                        early_stopping_rounds= 1000, # 1000, #5000, #1000,
                        verbose_eval= 500, #100,

                        #callbacks=[save_model2(models_folder=pathlib.Path(path), fold_id=global_variables.counter, min_score_to_save=0.78, every_k=50)]
                        #callbacks = [save_model()]
                        ) 
            #model.save_model(f"../models/models-{self.locker['comp_name']}/model_exp_{self.current_dict['current_exp_no']}.xgb")
            
            # https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost

            # print('best ntree_limit:', model.best_ntree_limit)
            # print('best score:', model.best_score)
            # self.best_ntree_limit_value = model.best_ntree_limit

            # GET FEATURE IMPORTANCE FOR FOLD K
            # will work only when we pass a dataframe
            if self.calculate_feature_importance:   
                dd = model.get_score(importance_type='weight')
                temp = pd.DataFrame({'feature': list(dd.keys()) ,f'importance_{global_variables.counter}': list(dd.values())}) # dd.keys() replace by self.ordered_list_test
                if self.feature_importance_table is None:
                    # first time
                    self.feature_importance_table = pd.DataFrame({'feature':dd.keys(),f'importance_{global_variables.counter}':dd.values()})
                else:
                    self.feature_importance_table = pd.merge(self.feature_importance_table, temp, on="feature",how="left")
                
                del dd , temp 
                gc.collect()
                print(self.feature_importance_table)

        elif self.model_name == "cbc":
            # for catboost params,eval_metric are set while creating model actually it is done for every wrapper function.
            # for low level like lgb we call it from .train()
            model.fit(
                self.xtrain,
                self.ytrain,
                eval_set = [(self.xvalid, self.yvalid)],
                verbose=0,
                
                #eval_metric= [lgb_amex_metric], no eval metric for catboost, actually below true
                # we can call it when instantiating 
                # https://stackoverflow.com/questions/65462220/how-to-create-custom-eval-metric-for-catboost

                # https://www.kaggle.com/code/thedevastator/ensemble-lightgbm-catboost-xgboost
            )
        elif self.model_name == "cbr":
            # for catboost params,eval_metric are set while creating model actually it is done for every wrapper function.
            # for low level like lgb we call it from .train()
            model.fit(
                self.xtrain,
                self.ytrain,
                eval_set = [(self.xvalid, self.yvalid)],
                verbose=0,
            )

        elif self.model_name in ["tabnetr", "tabnetc", "k1", "k2", "k3", "k4"]:  # keras
            # ------------------------- general for all keras model
            # lr ranges from 1--> 0.001
            # in cosine_decay : start 0.01 --> end 0.0001
            # in exponential_decay : start 0.01 --> end 0.00001

            # [ when we use lr scheduler]
            # -> we will use lr as the start while keeping end fixed to be either 0.0001 or 0.00001

            # [ when we don't use lr scheduler]
            # -> we will use lr as constant , in this way we don't have to make much changes.
            # Althouh it is better to always use lr_scheduler
            verbose = 0
            if self.model_name not in ["tabnetc", "tabnetr"]:
                callbacks = self.get_callbacks(params, verbose)

            # if self.model_name == "k4" or self.model_name == "k1": # for now we are passing "k1" as well later make it as param [callbacks]
            #     # DLastStark model
            #     callbacks = [lr, chk_point, TerminateOnNaN()]
            # else:
            #     callbacks = [ checkpoint, reduce_lr], #[stop, checkpoint, reduce_lr],

            # ----------------------
            if self.model_name in ["tabnetc", "tabnetr"]:
                        model.fit(
                        self.xtrain, self.ytrain,
                        eval_set=[(self.xvalid, self.yvalid)],
                        max_epochs = 30,
                        # patience = 50,
                        batch_size = 64, 
                        virtual_batch_size = 128, #128*20,
                        num_workers = 4,
                        drop_last = False,

                        )
            elif self.locker["data_type"] == "tabular":
                history = model.fit(
                    x=self.xtrain,
                    y=self.ytrain,
                    validation_data=(
                        self.xvalid,
                        self.yvalid,
                    ),  # validation_data will override validation_split
                    batch_size=params["batchsize"],
                    epochs=params["epochs"],
                    shuffle=True,
                    validation_split=0.15,
                    callbacks=callbacks,
                )
                if self.model_name == "k4":
                    model = load_model(
                        f"../models/models-{self.locker['comp_name']}/model_exp_{self.current_dict['current_exp_no']}.h5",
                    )
            if self.locker["data_type"] == "image":
                history = model.fit(
                    self.train_dataset,
                    steps_per_epoch=np.ceil(
                        len(self.xtrain) / self.params["batchsize"]
                    ),
                    epochs=self.params["epochs"],
                    verbose=2,
                    validatioin_data=self.valid_dataset,
                    validation_step=8,
                    callbacks=[stop, checkpoint, reduce_lr],
                )
                model.evaluate_generator(
                    generator=self.valid_dataset, steps=1
                )  # 1 to make it perfectly divisible

            if self.model_name not in ["tabnetc", "tabnetr"]:
                self._history = history.history
        elif self.model_name in ["tez1", "tez2", "p1", "pretrained"]:  # pytorch
            model_path_es = f"../models/models-{self.locker['comp_name']}/model_exp_{self.current_dict['current_exp_no'] + 1}_es"  # 'model_es_s' + str(CFG.img_size) + '_f' +str(fold) + '.bin',
            model_path_s = f"../models/models-{self.locker['comp_name']}/model_exp_{self.current_dict['current_exp_no'] + 1}_s"
            if self._state == "seed":
                model_path_es = model_path_es + f"_seed_{self._random_state}"
                model_path_s = model_path_s + f"_seed_{self._random_state}"
            stop = EarlyStopping(
                monitor="valid_loss",
                model_path=model_path_es,
                patience=params["patience"],
                mode="min",
            )
            es = EarlyStopping(
                monitor="valid_accuracy",
                model_path=model_path_es,
                patience=10,
                mode="max",
                save_weights_only=True,
            )

            if self.model_name == "tez1":
                history = model.fit(
                    self.train_dataset,  # self.train_dataset
                    valid_dataset=self.valid_dataset,  # self.valid_dataset
                    train_bs=params["batch_size"],
                    valid_bs=16,
                    device="cuda",
                    epochs=params["epochs"],
                    callbacks=[stop],
                    fp16=True,
                )
                # self._history = history.history
                model.save(
                    model_path_s,
                )
            elif self.model_name == "tez2":
                config = TezConfig(
                    training_batch_size=params["batch_size"],
                    validation_batch_size=2 * params["batch_size"],
                    test_batch_size=2 * params["batch_size"],
                    gradient_accumulation_steps=1,
                    epochs=params["epochs"],
                    step_scheduler_after="epoch",
                    step_scheduler_metric="valid_accuracy",
                    fp16=True,
                )

                model.fit(
                    self.train_dataset,
                    valid_dataset=self.valid_dataset,
                    callbacks=[es],
                    config=config,
                )
            elif self.model_name in ["p1", "pretrained"]:
                model.fit(n_iter=params["epochs"])

            # self._history = history.history
            model.save(
                model_path_s,
            )

        else:  # tabular
            model.fit(self.xtrain, self.ytrain)
        
        # for now calculate here feature_permutation importance for xgb
        """
        Can be modified to test only the new features, to make it fast
        """
        # col1, col2, col3, col4
        if self.calculate_permutation_feature_importance:
            prefix_name = "perm_importance"
            perm = {}
            #dvalid = xgb.DMatrix(data = self.xvalid, label = self.yvalid)
            #self.valid_preds = [model.predict(self.xvalid)]
            if self.model_name.startswith("k"):
                self.valid_preds = [model.predict(self.xvalid).flatten()] # NN
            else:
                self.valid_preds = [model.predict(self.xvalid)] 

            #baseline= amex_metric_mod(self.yvalid, self.valid_preds[0][:])

            # score valid predictions
            if self.metrics_name in [
                "auc",
                "accuracy",
                "f1",
                "recall",
                "precision",
                "logloss",
                "auc_tf",
                "amex_metric",
                "amex_metric_mod",
                "amex_metric_lgb_base",
            ]:
                # Classification
                cl = ClassificationMetrics()
                if self.locker["comp_type"] == "multi_label":
                    s1 = cl(self.metrics_name, self.yvalid[:, 0], self.valid_preds[0][:])
                    s2 = cl(self.metrics_name, self.yvalid[:, 1], self.valid_preds[1][:])
                    s3 = cl(self.metrics_name, self.yvalid[:, 2], self.valid_preds[2][:])
                    baseline = (s1 + s2 + s3) / 3
                elif self.metrics_name in ["auc", "log_loss", "amex_metric"]:
                    # then y proba can't be None
                    # sanity check : convert it to numpy array

                    baseline = cl(
                        self.metrics_name,
                        np.array(self.yvalid),
                        "y_pred_dummy",
                        np.array(self.valid_preds[0]),
                    )
                elif self.metrics_name == "amex_metric_mod":
                    baseline= amex_metric_mod(self.yvalid, self.valid_preds[0][:])
                elif self.metric_name == "amex_metric_lgb_base":
                    baseline = amex_metric_lgb_base(self.yvalid, self.yvalid_preds[0][:])
                else:
                    baseline = cl(self.metrics_name, self.yvalid, self.valid_preds[0][:])
            elif self.metrics_name in ["mae", "mse", "rmse", "msle", "rmsle", "r2"]:
                # Regression
                rg = RegressionMetrics()
                baseline = rg(self.metrics_name, self.yvalid, self.valid_preds)
            elif self.metrics_name == "getaroom_metrics":
                baseline = getaroom_metrics(self.yvalid, self.valid_preds[0])
            elif self.metrics_name == "amzcomp1_metrics":
                baseline = amzcomp1_metrics(self.yvalid, self.valid_preds[0])


            #care_feat = amex4_settings().feature_dict2["date"]
            # care_feat = getaroom_settings().feature_dict['base']
            # care_feat += getaroom_settings().feature_dict['base_interact']
            # care_feat = list(set(care_feat)) # don't know why it duplicated 

            care_feat = list(set(self.useful_features))

            print("Total no of features:", len(care_feat), len(set(care_feat)))
            print(f"Expected Time: {len(care_feat)*2/60} minutes" )

            index_list = [self.useful_features.index(i) for i in care_feat]
            
            self.xvalid1 = self.xvalid.copy()
            for i,f_name in zip(index_list, care_feat): #range(self.xvalid.shape[1]): # no of features 
                #print(f"perm: {i}")
                value = self.xvalid1[:,i].copy()
                # permute 10 times 
                score = []
                for j in range(3):
                    self.xvalid1[:,i] = np.random.permutation(self.xvalid1[:,i].copy())
                    #dvalid = xgb.DMatrix(data = self.xvalid, label = self.yvalid)
                    if self.model_name.startswith("k"):
                        self.valid_preds = [model.predict(self.xvalid).flatten()] # NN
                    else:
                        self.valid_preds = [model.predict(self.xvalid1)] 

                    # score valid predictions
                    if self.metrics_name in [
                        "auc",
                        "accuracy",
                        "f1",
                        "recall",
                        "precision",
                        "logloss",
                        "auc_tf",
                        "amex_metric",
                        "amex_metric_mod",
                        "amex_metric_lgb_base",
                    ]:
                        # Classification
                        cl = ClassificationMetrics()
                        if self.locker["comp_type"] == "multi_label":
                            s1 = cl(self.metrics_name, self.yvalid[:, 0], self.valid_preds[0][:])
                            s2 = cl(self.metrics_name, self.yvalid[:, 1], self.valid_preds[1][:])
                            s3 = cl(self.metrics_name, self.yvalid[:, 2], self.valid_preds[2][:])
                            baseline1 = (s1 + s2 + s3) / 3
                        elif self.metrics_name in ["auc", "log_loss", "amex_metric"]:
                            # then y proba can't be None
                            # sanity check : convert it to numpy array

                            baseline1 = cl(
                                self.metrics_name,
                                np.array(self.yvalid),
                                "y_pred_dummy",
                                np.array(self.valid_preds[0]),
                            )
                        elif self.metrics_name == "amex_metric_mod":
                            baseline1= amex_metric_mod(self.yvalid, self.valid_preds[0][:])
                        elif self.metric_name == "amex_metric_lgb_base":
                            baseline1 = amex_metric_lgb_base(self.yvalid, self.yvalid_preds[0][:])
                        else:
                            baseline1 = cl(self.metrics_name, self.yvalid, self.valid_preds[0][:])
                    elif self.metrics_name in ["mae", "mse", "rmse", "msle", "rmsle", "r2"]:
                        # Regression
                        rg = RegressionMetrics()
                        baseline1 = rg(self.metrics_name, self.yvalid, self.valid_preds)
                    elif self.metrics_name == "getaroom_metrics":
                        baseline1 = getaroom_metrics(self.yvalid, self.valid_preds[0])
                    elif self.metrics_name == "amzcomp1_metrics":
                        baseline1 = amzcomp1_metrics(self.yvalid, self.valid_preds[0])

                    score.append(baseline1)
                    #score.append( amex_metric_mod(self.yvalid, self.valid_preds[0][:]) )
                #perm[self.useful_features[i]] = np.mean(score) - baseline
                perm[f_name] = np.mean(score) - baseline

                # reset back
                self.xvalid1[:,i] = value 

                gc.collect()
            temp = pd.DataFrame({'feature': list(perm.keys()) ,f'{prefix_name}_{global_variables.counter}': list(perm.values())}) # dd.keys() replace by self.ordered_list_test
            if self.feature_importance_table is None:
                # first time
                self.feature_importance_table = pd.DataFrame({'feature':perm.keys(),f'{prefix_name}_{global_variables.counter}':perm.values()})
            else:
                self.feature_importance_table = pd.merge(self.feature_importance_table, temp, on="feature",how="left")
            
            del perm , temp , self.xvalid1
            gc.collect()
            print(self.feature_importance_table)


        """
        Make prediction    [keras datagen pytorch dataset ] 
        tabular xvalid yvalid
        # because of multi-class evearything is a list even for 1d it is list of single element i.e. 1d array

        """
        metrics_name = self.metrics_name
        if self.locker["data_type"] in ["image_path", "image_df", "image_folder"]:
            # storage for oof and submission

            # produce predictions - oof
            if self.model_name in ["tabnetr", "tabnetc", "k1", "k2", "k3"]:
                # keras image
                self.valid_dataset.reset()
                temp_preds = model.predict_generator(
                    self.valid_dataset, steps=STEP_SIZE_TEST, verbose=1
                )
            elif self.model_name in ["tez1", "tez2"]:
                valid_preds = model.predict(
                    self.valid_dataset, batch_size=16, n_jobs=-1
                )
                temp_preds = None
                for p in valid_preds:
                    if temp_preds is None:
                        temp_preds = p
                    else:
                        temp_preds = np.vstack((temp_preds, p))
                gc.collect()
            elif self.model_name in ["p1", "pretrained"]:
                valid_preds = model.predict(self.valid_loader)
                # valid_preds = valid_preds.to("cpu")
                if self.locker["comp_type"] == "multi_label":
                    temp_preds = [None, None, None]
                else:
                    temp_preds = [None]
                for i, n in enumerate(temp_preds):
                    valid_preds[i] = valid_preds[i].to("cpu")
                    for p in valid_preds[i]:
                        if temp_preds[i] is None:
                            temp_preds[i] = p
                        else:
                            temp_preds[i] = np.vstack((temp_preds[i], p))
                        gc.collect()
                    gc.collect()

            # for now done only for pretrained part
            self.valid_preds = [
                np.argmax(temp_pred, axis=1) for temp_pred in temp_preds
            ]
            if self.locker["comp_type"] == "multi_label":
                print("Cal valid preds")
                print(self.valid_preds[0][:3])
                print(self.valid_preds[1][:3])
                print(self.valid_preds[2][:3])

            if (
                self._state == "seed" or self._state == "fold"
            ):  # so create test prediction
                # produce predictions - test data

                if self.model_name in ["tabnetr", "tabnetc", "k1", "k2", "k3"]:
                    self.valid_dataset = model.predict_generator(
                        self.test_dataset, steps=STEP_SIZE_TEST, verbose=1
                    )
                elif self.model_name in ["tez1", "tez2"]:
                    test_preds = model.predict(
                        self.test_dataset, batch_size=params["batch_size"], n_jobs=-1
                    )
                    temp_preds = None
                    for p in test_preds:
                        if temp_preds is None:
                            temp_preds = p
                        else:
                            temp_preds = np.vstack((temp_preds, p))
                        gc.collect()

                elif self.model_name in ["p1", "pretrained"]:
                    test_preds = model.predict(self.test_loader)
                    # test_preds = test_preds.to("cpu")
                    if self.locker["comp_type"] == "multi_label":
                        temp_preds = [None, None, None]
                    else:
                        temp_preds = [None]
                    for i, n in enumerate(temp_preds):
                        test_preds[i] = test_preds[i].to("cpu")
                        for p in test_preds[i]:
                            if temp_preds[i] is None:
                                temp_preds[i] = p
                            else:
                                temp_preds[i] = np.vstack((temp_preds[i], p))
                            gc.collect()
                        gc.collect()

                    # temp_preds = None
                    # for p in test_preds:
                    #     if temp_preds is None:
                    #         temp_preds = p
                    #     else:
                    #         temp_preds = np.vstack((temp_preds, p))

                # self.test_preds = temp_preds.argmax(axis=1)
                self.test_preds = [temp_pred.argmax(axis=1) for temp_pred in temp_preds]
                if self.locker["comp_type"] == "multi_label":
                    print("Cal test preds")
                    print(self.test_preds[0][:3])
                    print(self.test_preds[1][:3])
                    print(self.test_preds[2][:3])

        elif self.locker["data_type"] == "tabular":
            if self.comp_type in [
                "2class",
                "multi_class",
                "multi_label",
            ] and self.model_name in [
                "xgb", # let's say we use it as a classifier
                "xgbc",
                "cbc",
                "mlpc",
                "knnc",
                "dtc",
                "adbc",
                "gbmc",
                "hgbc",
                "lgbmc",
                "rfc",
            ]:  # self.model_name not in ["xgbr","lgr","lir", "lgbmr"]:
                if self.model_name == "lgbmc":
                    #self.valid_preds = model.predict_proba(self.xvalid, raw_score=True) #[:, 1]
                    self.valid_preds = model.predict_proba(self.xvalid)[:, 1]
                elif self.model_name in ["xgb"]: # lgb normal
                    self.valid_preds = model.predict(dvalid ) #, iteration_range = (0, self.best_ntree_limit_value))
                else:
                    self.valid_preds = model.predict_proba(self.xvalid)[:, 1]
            elif self.model_name.startswith("k")or self.model_name in ["tabnetr", "tabnetc"]:
                self.valid_preds = model.predict(self.xvalid).flatten() # NN
                print("k")
                print(self.xvalid.shape, self.valid_preds.shape)
                print(self.valid_preds)
            else:
                self.valid_preds = model.predict(self.xvalid) # ML
            self.valid_preds = [
                self.valid_preds
            ]  # list of predictions maintain to sink with multilabel

            if (
                self._state == "seed" or self._state == "fold"
            ):  # so create test prediction
                # produce predictions - test data

                if self.locker["comp_type"] == "multi_label":
                    temp_preds = [None, None, None]
                else:
                    temp_preds = [None]
                # special case
                if self.comp_type in [
                    "2class",
                    "multi_class",
                    "multi_label",
                ] and self.model_name in [
                    "xgbc",
                    "cbc",
                    "mlpc",
                    "knnc",
                    "dtc",
                    "adbc",
                    "gbmc",
                    "hgbc",
                    "lgbmc",
                    "rfc",
                ]:  # self.model_name not in ["xgbr","lgr","lir", "lgbmr"]:
                    # shape should be 1D (15232,)
                    if self.model_name == "xgb":
                        temp_preds[0] = model.predict(xgb.DMatrix(data=self.xtest) ) # ,  iteration_range = (0, self.best_ntree_limit_value)) # Yes this was causing issue
                    else:
                        temp_preds[0] = model.predict_proba(self.xtest)[:, 1]
                    
                else:
                    temp_preds[0] = model.predict(self.xtest)

                # else:
                #     temp_preds[0] = model.predict(self.xtest)
                self.test_preds = temp_preds

                del temp_preds
                gc.collect()

        else:
            raise Exception(f"metrics not set yet of type {self.locker['data_type']}")

        # score valid predictions
        if self.metrics_name in [
            "auc",
            "accuracy",
            "f1",
            "recall",
            "precision",
            "logloss",
            "auc_tf",
            "amex_metric",
            "amex_metric_mod",
            "amex_metric_lgb_base",
        ]:
            # Classification
            cl = ClassificationMetrics()
            if self.locker["comp_type"] == "multi_label":
                s1 = cl(self.metrics_name, self.yvalid[:, 0], self.valid_preds[0][:])
                s2 = cl(self.metrics_name, self.yvalid[:, 1], self.valid_preds[1][:])
                s3 = cl(self.metrics_name, self.yvalid[:, 2], self.valid_preds[2][:])
                score = (s1 + s2 + s3) / 3
            elif self.metrics_name in ["auc", "log_loss", "amex_metric"]:
                # then y proba can't be None
                # sanity check : convert it to numpy array

                score = cl(
                    self.metrics_name,
                    np.array(self.yvalid),
                    "y_pred_dummy",
                    np.array(self.valid_preds[0]),
                )
            elif self.metrics_name == "amex_metric_mod":
                score= amex_metric_mod(self.yvalid, self.valid_preds[0][:])
            elif self.metric_name == "amex_metric_lgb_base":
                score = amex_metric_lgb_base(self.yvalid, self.yvalid_preds[0][:])
            else:
                print(self.metrics_name)
                print(self.yvalid[:5])
                print(self.valid_preds[:5])
                print()
                score = cl(self.metrics_name, self.yvalid, self.valid_preds[0][:])
        elif self.metrics_name in ["mae", "mse", "rmse", "msle", "rmsle", "r2"]:
            # Regression
            rg = RegressionMetrics()
            score = rg(self.metrics_name, self.yvalid, self.valid_preds)
        elif self.metrics_name == "getaroom_metrics":
            #print(self.yvalid.shape, len(self.valid_preds[0]), "this si sit")
            print(self.yvalid[:4], self.valid_preds[0][:5])
            score = getaroom_metrics(self.yvalid, self.valid_preds[0])
        elif self.metrics_name == "amzcomp1_metrics":
            #print(self.yvalid.shape, len(self.valid_preds[0]), "this si sit")
            print(self.yvalid[:4], self.valid_preds[0][:5])
            score = amzcomp1_metrics(self.yvalid, self.valid_preds[0])


        if self._state == "opt":
            # Let's save these values
            self._trial_score = score  # save it to save in log_table because params don't contain our metrics score
            self.save_logs(params)

        check_memory_usage("Obj", self, 0)

        #trial no, It records no of times objective function is called
        global_variables.counter += 1
        return score

    def run(
        self,
        useful_features,
        with_gpu="--|--",
        prep_list="--|--",
        optimize_on="--|--",
    ):
        """
        Run is used to call Optuna trials
        Run is also used to initialize variables while making prediction,
        so we call run --> then we call obj while Predicting
        """
        if with_gpu != "--|--":
            self.with_gpu = with_gpu
        if optimize_on != "--|--":
            self.optimize_on = optimize_on
        if prep_list != "--|--":
            self.prep_list = prep_list
        self.useful_features = useful_features  # ["pixel"]
        """
        ######################################
        #         Memory uage                #
        ######################################
        """
        print(f"Optimize on fold name {self.fold_name} and fold no: {self.optimize_on}")

        # BOTTLENECK
        return_type = "numpy_array" # "numpy_array" # "tensor"
        # xtest is not needed in optimiztion
        # xtest is needed in predict but only once so called from outside
        # slow
        # for i,f in enumerate(self.optimize_on):
        #     if i==0:
        #         # first time
        #         self.val_idx, self.xtrain, self.xvalid, self.ytrain, self.yvalid, self.ordered_list_train = bottleneck(self.locker['comp_name'],self.useful_features, self.fold_name, f, self._state, return_type)
        #     else:
        #         # second time 
        #         val_idx, xtrain, xvalid, ytrain, yvalid, self.ordered_list_train = bottleneck(self.locker['comp_name'],self.useful_features, self.fold_name, f, self._state, return_type)
        #         if val_idx is not None:
        #             self.val_idx += val_idx 
        #         else:
        #             self.valid_idx = None
        #         if xtrain is not None:
        #             self.xtrain = np.concatenate([self.xtrain, xtrain], axis=0)
        #         else:
        #             self.xtrain = None 
        #         if xvalid is not None:
        #             self.xvalid = np.concatenate([self.xvalid, xvalid], axis=0)
        #         else:
        #             self.xvalid = None 
        #         if ytrain is not None:
        #             self.ytrain = np.concatenate([self.ytrain, ytrain], axis=0)
        #         else:
        #             self.ytrain = None 
        #         if yvalid is not None:
        #             self.yvalid = np.concatenate([self.yvalid, yvalid], axis=0)
        #         else:
        #             self.yvalid = None 
                
        #         del val_idx, xtrain, xvalid, ytrain, yvalid
        #         gc.collect()

        # optimize on fold no [2] mean optimizing on [0,1,3,4]
        self.val_idx, self.xtrain, self.xvalid, self.ytrain, self.yvalid, self.ordered_list_train = bottleneck(self.locker['comp_name'],self.useful_features, self.fold_name, self.optimize_on, self._state, return_type)       
        if self.model_name  == "tabnetr":
            self.ytrain = self.ytrain.reshape(-1,1)
            self.yvalid = self.yvalid.reshape(-1,1)
        print("printing xtrain")
        print(self.xtrain)
        print()
        print("printing xvalid")
        print(self.xvalid)
        print("-"*40)
        if self.model_name.startswith("k") or self.model_name in ["tabnetr", "tabnetc", 'gbmr','gbmc']:
            if self._state == "fold":
                # has train , valid , test 
                # fill with na in case of NN
                self.xtrain[np.isnan(self.xtrain)] = 0
                self.xvalid[np.isnan(self.xvalid)] = 0
                self.xtest[np.isnan(self.xtest)] = 0
                #print("self.xtrain.shape, self.ytrain.shape, self.xvalid.shape, self.yvalid.shape, len(self.val_idx), self.xtest.shape")
                #print(self.xtrain.shape, self.ytrain.shape, self.xvalid.shape, self.yvalid.shape, len(self.val_idx), self.xtest.shape)
                pass
            elif self._state == "opt":
                self.xtrain[np.isnan(self.xtrain)] = 0
                self.xvalid[np.isnan(self.xvalid)] = 0
                #print("self.xtrain.shape, self.ytrain.shape, self.xvalid.shape, self.yvalid.shape, len(self.val_idx)")
                #print(self.xtrain.shape, self.ytrain.shape, self.xvalid.shape, self.yvalid.shape, len(self.val_idx))
                pass
            else:
                raise Exception("Can't enter run function if _state != opt/fold")

        if self._state != 'opt':  # "fold", "_seed" then we would have xtest and corresponding self.ordered_list_test 
            # sanity check: 
            for c,(i,j) in enumerate(zip(self.ordered_list_test, self.ordered_list_train)):
                if i != j:
                    print()
                    print("--+"*40)
                    print(self.ordered_list_test)
                    print()
                    print(self.ordered_list_train)
                    raise Exception(f"Feature no {c} don't correspond in test - train {i},{j}")
        # be sure 
        self.useful_features = self.ordered_list_train 
        print("No of Features:",len(self.useful_features))

        if self.model_name in ["lgr"]:
            # fill missing values 
            for item in [self.xtrain, self.xvalid]:
                if item is not None:
                    item[ np.isnan(item)] = -999 #item.fillna(-999, inplace=True)

        #self.val_idx, self.xtrain, self.xvalid, self.ytrain, self.yvalid, self.xtest = bottleneck(self.locker['comp_name'],self.useful_features, self.fold_name, self.optimize_on, self._state, return_type)
        # print(len(self.val_idx))
        # print(self.xtrain.shape)
        # print(self.xvalid.shape)
        # print(self.ytrain.shape)
        # print(self.yvalid.shape)
        #print(self.xtest.shape)
        """
        image_df : image is stored in dataframe 
        image_path: image path is there in dataframe 
        image_folder: there are image folders

        returns 
        self.valid_dataset 
        self.train_dataset
        """
        # => albumations augmentations
        # tez1
        if self.aug_type == "aug1":
            self.train_aug = A.Compose(
                [
                    A.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        max_pixel_value=255.0,
                        p=1.0,
                    )
                ],
                p=1.0,
            )
            self.valid_aug = A.Compose(
                [
                    A.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5],
                        max_pixel_value=255.0,
                        p=1.0,
                    )
                ],
                p=1.0,
            )
        # tez2
        elif self.aug_type == "aug2":
            self.train_aug = A.Compose(
                [
                    albumentations.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                ],
                p=1.0,
            )
            self.valid_aug = A.Compose(
                [
                    albumentations.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    ),
                ],
                p=1.0,
            )
        # kaggle tv
        elif self.aug_type == "aug3":
            self.train_aug = A.Compose([Rotate(20), ToTensor()])
            self.valid_aug = A.Compose([ToTensor()])
        # abhishek bengaliai video
        elif self.aug_type == "aug4":
            # valid
            self.valid_aug = albumentations.Compose(
                [
                    albumentations.Resize(128, 128, always_apply=True),
                    albumentations.Normalize(
                        (0.485, 0.456, 0.406), (9, 0.224, 0.225), always_apply=True
                    ),
                ]
            )
            # train
            self.train_aug = albumentations.Compose(
                [
                    albumentations.Resize(128, 128, always_apply=True),
                    albumentations.ShiftScaleRotate(
                        shift_limit=0.0625, scale_limit=5, p=0.9
                    ),
                    albumentations.Normalize(
                        (0.485, 0.456, 0.406), (9, 0.224, 0.225), always_apply=True
                    ),
                ]
            )

        # self.sample = pd.read_csv(
        #     f"../input/input-{self.locker['comp_name']}/" + "sample.csv"
        # )
        # self.test = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/" + "test.csv")
        # Do we need self.test Now because we are not mapping to id using whole putting whole as array
        # we definately not need it in optimization
        # self.test = pd.read_parquet(
        #     f"../input/input-{self.locker['comp_name']}/" + "test.parquet"
        # )
        # self.test[self.locker["target_name"]] = 0.0
        # self.sample = self.test.copy() # temp No need now It only increases memory usage use inside "image_df"
        # => datasets
        if self.locker["data_type"] == "image_path":
            image_path = f"../input/input-{self.locker['comp_name']}/" + "train_images/"
            test_path = f"../input/input-{self.locker['comp_name']}/" + "test_images/"
            self.sample = self.test.copy()[
                [self.locker["id_name"], self.locker["target_name"]]
            ]
            if self.model_name in ["tez1", "tez2", "pretrained", "p1"]:
                # now implemented for pytorch

                # use pytorch
                self.train_image_paths = [
                    os.path.join(image_path, str(x))
                    for x in self.xtrain[self.locker["id_name"]].values
                ]
                self.valid_image_paths = [
                    os.path.join(image_path, str(x))
                    for x in self.xvalid[self.locker["id_name"]].values
                ]

                # new

                # ------------------  prep test dataset
                # self.test_image_paths = [
                #     os.path.join(
                #         test_path, str(x)
                #     )  # f"../input/input-{self.locker['comp_name']}/" + "test_img/" + x
                #     for x in self.sample[self.locker["id_name"]].values
                # ]
                # fake targets

                # correctly defince sample
                new_list = []  # do this to maintain order
                for i in self.sample[self.locker["id_name"]]:
                    if i not in new_list:
                        new_list.append(i)
                    gc.collect()

                self.sample = pd.DataFrame(new_list, columns=[self.locker["id_name"]])
                self.sample[self.locker["target_name"]] = 0

                self.test_image_paths = [
                    os.path.join(
                        test_path, str(x)
                    )  # f"../input/input-{self.locker['comp_name']}/" + "test_img/" + x
                    for x in self.sample[self.locker["id_name"]].values
                ]

                self.test_targets = self.sample[
                    self.locker["target_name"]
                ].values  # dfx_te.digit_sum.values
                # ==========================================>

                if self._dataset in [
                    "BengaliDataset",
                ]:
                    print("Entered here", self._dataset)
                    # BengaliDataset
                    self.train_dataset = BengaliDataset(  # train_dataset
                        image_paths=self.train_image_paths,
                        targets=self.ytrain,
                        img_height=128,
                        img_width=128,
                        transform=self.train_aug,
                    )

                    self.valid_dataset = BengaliDataset(  # valid_dataset
                        image_paths=self.valid_image_paths,
                        targets=self.yvalid,
                        img_height=128,
                        img_width=128,
                        transform=self.valid_aug,
                    )
                    self.test_dataset = BengaliDataset(
                        image_paths=self.test_image_paths,
                        targets=self.test_targets,
                        img_height=128,
                        img_width=128,
                        transform=self.valid_aug,
                    )
                    print(self.train_dataset)
                    print(self.valid_dataset)
                    print(self.test_dataset)
                else:
                    # imageDataset
                    self.train_dataset = ImageDataset(  # train_dataset
                        image_paths=self.train_image_paths,
                        targets=self.ytrain,
                        augmentations=self.train_aug,
                    )

                    self.valid_dataset = ImageDataset(  # valid_dataset
                        image_paths=self.valid_image_paths,
                        targets=self.yvalid,
                        augmentations=self.valid_aug,
                    )

                    self.test_dataset = ImageDataset(
                        image_paths=self.test_image_paths,
                        targets=self.test_targets,
                        augmentations=self.valid_aug,
                    )

            elif self.model_name in ["k1", "k2", "k3"]:
                # now implemented for keras
                # use keras flow_from_dataframe
                train_datagen = ImageDataGenerator(rescale=1.0 / 255)
                valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

                if self.use_cutmix != True:
                    self.train_dataset = train_datagen.flow_from_dataframe(
                        dataframe=self.xtrain,
                        directory=image_path,
                        target_size=(28, 28),  # images are resized to (28,28)
                        x_col=self.locker["id_name"],
                        y_col=self.locker["target_name"],
                        batch_size=32,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",  # "binary"
                    )
                elif self.use_cutmix == True:
                    train_datagen1 = train_datagen.flow_from_dataframe(
                        dataframe=self.xtrain,
                        directory=image_path,
                        target_size=(28, 28),  # images are resized to (28,28)
                        x_col=self.locker["id_name"],
                        y_col=self.locker["target_name"],
                        batch_size=32,
                        seed=42,
                        shuffle=True,  # Required for cutmix
                        class_mode="categorical",  # "binary"
                    )
                    train_datagen2 = train_datagen.flow_from_dataframe(
                        dataframe=self.xtrain,
                        directory=image_path,
                        target_size=(28, 28),  # images are resized to (28,28)
                        x_col=self.locker["id_name"],
                        y_col=self.locker["target_name"],
                        batch_size=32,
                        seed=42,
                        shuffle=True,  # Required for cutmix
                        class_mode="categorical",  # "binary"
                    )
                    self.train_dataset = CutMixImageDataGenerator(
                        generator1=train_generator1,
                        generator2=train_generator2,
                        img_size=(28, 28),
                        batch_size=32,
                    )
                self.valid_dataset = valid_datagen.flow_from_dataframe(
                    dataframe=self.xvalid,
                    directory=image_path,
                    target_size=(28, 28),  # images are resized to (28,28)
                    x_col=self.locker["id_name"],
                    y_col=self.locker["target_name"],
                    batch_size=32,
                    seed=42,
                    shuffle=True,
                    class_mode="categorical",  # "binary"
                )

        elif self.locker["data_type"] == "image_df":
            # # it is not good to use whole image everytime
            # # create a seperate test_df and sample_df aka test to store test set
            # self.test = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/" + "test_df.csv")
            t = []
            for n in self.useful_features:
                t += filter(lambda x: x.startswith(n), list(self.xtrain.columns))
                gc.collect()

            self.filtered_features = t

            if self._dataset in [
                "BengaliDataset",
            ]:
                # now implemented for pytorch
                # Can make our own custom dataset.. Note tez has dataloader inside the model so don't make
                self.train_dataset = BengaliDataset(  # train_dataset
                    csv=self.xtrain[
                        self.filtered_features + [self.locker["target_name"]]
                    ],
                    img_height=28,
                    img_width=28,
                    transform=self.train_aug,
                )

                self.valid_dataset = BengaliDataset(  # valid_dataset
                    csv=self.xvalid[
                        self.filtered_features + [self.locker["target_name"]]
                    ],
                    img_height=28,
                    img_width=28,
                    transform=self.valid_aug,
                )
                self.test_dataset = BengaliDataset(
                    df=self.test[self.filtered_features + [self.locker["target_name"]]],
                    img_height=28,
                    img_width=28,
                    augmentations=self.valid_aug,
                )

            elif self._dataset in [
                "DigitRecognizerDataset",
            ]:
                # DigitRecognizerDataset
                self.train_dataset = DigitRecognizerDataset(
                    df=self.xtrain[
                        self.filtered_features + [self.locker["target_name"]]
                    ],
                    augmentations=self.train_aug,
                    model_name=self.model_name,
                )
                self.valid_dataset = DigitRecognizerDataset(
                    df=self.xvalid[
                        self.filtered_features + [self.locker["target_name"]]
                    ],
                    augmentations=self.valid_aug,
                    model_name=self.model_name,
                )
                self.test_dataset = DigitRecognizerDataset(
                    df=self.test[self.filtered_features + [self.locker["target_name"]]],
                    augmentations=self.valid_aug,
                    model_name=self.model_name,
                )

                # nn.Crossentropy requires target to be single not from interval [0, #classes]
                # if self.model_name.startswith("p") and self.comp_type != "2class":
                #     self.ytrain = np_utils.to_categorical(self.ytrain)
                #     self.yvalid = np_utils.to_categorical(self.yvalid)

        elif self.locker["data_type"] == "image_folder":
            # folders of train test
            pass
            # use keras flow_from_directory don't use for now because it looks for subfolders with folder name as different targets like horses/humans

        elif self.locker["data_type"] == "tabular":
            # concept of useful feature don't make sense for image problem
            # self.xtrain = self.xtrain[self.useful_features]
            # self.xvalid = self.xvalid[self.useful_features]

            # self.xtest = self.test[self.useful_features]
            # del self.test
            # gc.collect()

            prep_dict = {
                "SiMe": SimpleImputer(strategy="mean"),
                "SiMd": SimpleImputer(strategy="median"),
                "SiMo": SimpleImputer(strategy="mode"),
                "Ro": RobustScaler(),
                "Sd": StandardScaler(),
                "Mi": MinMaxScaler(),
            }
            for f in self.prep_list:
                if f in list(prep_dict.keys()):
                    sc = prep_dict[f]
                    self.xtrain = sc.fit_transform(self.xtrain)
                    if self._state != "opt":
                        self.xtest = sc.transform(self.xtest)
                elif f == "Lg":
                    self.xtrain = pd.DataFrame(
                        self.xtrain, columns=self.useful_features
                    )
                    self.xvalid = pd.DataFrame(
                        self.xvalid, columns=self.useful_features
                    )
                    if self._state != "opt":
                        self.xtest = pd.DataFrame(self.xtest, columns=self.useful_features)
                    # xtest = pd.DataFrame(xtest, columns=useful_features)
                    for col in self.useful_features:
                        self.xtrain[col] = np.log1p(self.xtrain[col])
                        self.xvalid[col] = np.log1p(self.xvalid[col])
                        if self._state != "opt":
                            self.xtest[col] = np.log1p(self.xtest[col])
                        # xtest[col] = np.log1p(xtest[col])
                        gc.collect()
                else:

                    raise Exception(f"scaler {f} is invalid!")
                gc.collect()

            # create instances
            # enable below for multiclass problem
            # if self.model_name.startswith("k") and self.comp_type != "2class":
            #     ## to one hot
            #     self.ytrain = np_utils.to_categorical(self.ytrain)
            #     self.yvalid = np_utils.to_categorical(self.yvalid)

        gc.collect()

        if self._state == "opt":
            # optimization specific work do here 
            # only looking at feature importance while optimizing
            self.calculate_feature_importance = True
            self.calculate_permutation_feature_importance = False 
            global_variables.counter = 0 # since we can't get trial no from optuna we are maintaining our own
            # run() can be called for each new exp so we initialize here rather than at init
            self.feature_importance_table = None # store the feature importances each trial wise

            # create optuna study
            study = optuna.create_study(direction=self._aim, study_name=self.model_name)
            study.optimize(
                lambda trial: self.obj(trial),
                n_trials=self.n_trials,
            )  # it tries 50 different values to find optimal hyperparameter

            if self.save_models == True:
                # let's save logs
                c = self.current_dict[
                    "current_exp_no"
                ]  # optuna is called once in each exp so c will be correct
                save_pickle(
                    f"../configs/configs-{self.locker['comp_name']}/logs/log_exp_{c}.pkl",
                    self._log_table,
                )
                self._log_table = None

            # save feature importance
            if self.calculate_feature_importance or self.calculate_permutation_feature_importance:
                # need to calculate feature importance 
                save_pickle(f"../configs/configs-{self.locker['comp_name']}/feature_importance/feature_importance_e_{c}_opt.pkl",self.feature_importance_table)
                del self.feature_importance_table
                gc.collect()

            print("=" * 40)
            print("Best parameters found:")
            print(study.best_trial.value)
            self.params = study.best_trial.params  # crete params to be used in seed
            print(study.best_trial.params)
            print("=" * 40)
            # later put conditions on whether to put seed or not
            if self.model_name == "lgr":
                del self.params["c"]
            # seed_mean, seed_std = self._seed_it()  # generate seeds  seed is now generate seperately
            
            
            gc.collect()
            check_memory_usage("run", self, 0)
            return study, self._random_state  # , seed_mean, seed_std

        check_memory_usage("run", self, 0)


if __name__ == "__main__":
    import optuna

    a = OptunaOptimizer()

    del a

    """
    {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'dart', 'learning_rate': 0.009238354429187228, 'seed': 241, 'num_leaves': 105, 'feature_fraction': 0.16728169509013685, 'bagging_freq': 9, 'bagging_fraction': 0.500294582036411, 'n_jobs': -1, 'lambda_l2': 3, 'min_data_in_leaf': 45}
    
    """
