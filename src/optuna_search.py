from metrics import ClassificationMetrics
from metrics import RegressionMetrics
from collections import defaultdict
import pickle
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from scipy import stats
import gc
import psutil
import seaborn as sns

sns.set()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from optuna.integration import LightGBMPruningCallback

# get skewed features to impute median instead of mean
from scipy.stats import skew
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBRegressor, XGBRFRegressor
import itertools
import optuna
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
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
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings

# Filter up and down
np.random.seed(1337)  # for reproducibility
warnings.filterwarnings("ignore")

from metrics import ClassificationMetrics, RegressionMetrics


class OptunaOptimizer:
    def __init__(
        self,
        model_name="lgr",
        comp_type="2class",
        metrics_name="accuracy",
        aim="maximize",
        n_trials=50,
        optimize_on=0,
        prep_list=[],
    ):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
            a = pickle.load(f)
        self.locker = a

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
        ]
        self.model_list = ["lgr", "lir", "xgbc", "xgbr"]
        self._prep_list = ["SiMe", "SiMd", "SiMo", "Mi", "Ro", "Sd", "Lg"]
        self.prep_list = prep_list
        self.comp_type = comp_type
        self.metrics_name = metrics_name
        if self.metrics_name in [
            "accuracy",
            "f1",
            "recall",
            "precision",
            "auc",
            "auc_tf",
            "r2",
        ]:
            self._aim = "maximize"
        else:
            self._aim = "minimize"
        self.n_trials = 50
        self.best_params = None
        self.best_value = None
        self.model_name = "lgr"
        self.optimize_on = optimize_on
        self.sanity_check()

    def show_variables(self):
        print()
        for i, (k, v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>", v)
        print()

    def sanity_check(self):
        if self.comp_type not in self.comp_list:
            raise Exception(f"{self.comp_type} not in the list {self.comp_list}")
        if self.metrics_name not in self.metrics_list:
            raise Exception(f"{self.metrics_name} not in the list {self.metrics_name}")
        if self.model_name not in self.model_list:
            raise Exception(f"{self.model_name} not in the list {self.model_list}")
        if self.optimize_on >= self.locker["no_folds"]:
            raise Exception(
                f"{self.optimize_on} out of range {self.locker['no_folds']}"
            )
        for p in self.prep_list:
            if p not in list(self._prep_list):
                raise Exception(f"{p} is invalid preprocessing type!")

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
        comp_random_state = self.locker["random_state"]
        total_no_folds = self.locker["no_folds"]
        fold_on = self.optimize_on
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
        seed = comp_random_state + total_no_folds * 2 + fold_on * 3 + metric_no * 4
        seed += int(
            comp_type_no * 5 + model_no * 6 + prep_no * 7
        )  # + round_on * 4 + level_on * 5
        seed = int(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        return np.random.randint(3, 1000)  # it should return 5

    def get_params(self, trial, ytrain):
        model_name = self.model_name
        if model_name == "lgr":
            params = {
                "class_weight": trial.suggest_categorical(
                    "class_weight",
                    [
                        "balanced",
                        None,
                        {1: 1, 0: (sum(list(ytrain == 0)) / sum(list(ytrain == 1)))},
                    ],
                ),
                "penalty": trial.suggest_categorical(
                    "penalty", ["l2"]
                ),  # ['l1','l2']),
                "C": trial.suggest_float("c", 0.01, 1000),
            }
            return params
        if model_name == "lir":
            return {}
        if model_name == "xgbc":
            return {}
        if model_name == "xgbr":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
                "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10]),
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
                "tree_method": trial.suggest_categorical("tree_method", ["gpu_hist"]),
                "gpu_id": trial.suggest_categorical("gpu_id", [0]),
                "predictor": trial.suggest_categorical("predictor", ["gpu_predictor"]),
            }
            return params
        if model_name == "kearas":  # demo
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

    def update_table(self):
        self.Table.loc[Table.shape[0], :] = [
            0,
            10 ** (-1 * learning_rate),
            learning_rate,
            epochs,
            batch_size,
            no_hidden_layers,
            dropout_placeholder,
            units_placeholder,
            batch_norm_placeholder,
            activation_placeholder,
        ]

    def get_model(self, params):
        # ["lgr","lir","xgbc","xgbr"]
        model_name = self.model_name
        self._random_state = self.generate_random_no()
        if model_name == "lgr":
            return LogisticRegression(**params, random_state=self._random_state)
        if model_name == "lir":
            return LinearRegression(**params, random_state=self._random_state)
        if model_name == "xgbc":
            return XGBClassifier(**params, random_state=self._random_state)
        if model_name == "xgbr":
            return XGBRegressor(**params, random_state=self._random_state)
        else:
            raise Exception(f"{model_name} is invalid!")

    def obj(self, trial, xtrain, ytrain, xvalid, yvalid):

        params = self.get_params(trial, ytrain)
        model = self.get_model(params)

        model.fit(xtrain, ytrain)

        metrics_name = self.metrics_name
        # Classification
        cl = ClassificationMetrics()
        if metrics_name == "auc":
            valid_preds = model.predict_proba(xvalid)[:, 1]
            score = cl("auc", yvalid, valid_preds)
        if metrics_name == "accuracy":
            valid_preds = model.predict(xvalid)
            score = cl("accuracy", yvalid, valid_preds)
        if metrics_name == "f1":
            valid_preds = model.predict(xvalid)
            score = cl("f1", yvalid, valid_preds)
        if metrics_name == "recall":
            valid_preds = model.predict(xvalid)
            score = cl("recall", yvalid, valid_preds)
        if metrics_name == "precision":
            valid_preds = model.predict(xvalid)
            score = cl("precision", yvalid, valid_preds)
        if metrics_name == "logloss":
            valid_preds = model.predict_proba(xvalid)
            score = cl("logloss", yvalid, valid_preds)
        if metrics_name == "auc_tf":
            valid_preds = model.predict_proba(xvalid)[:, 1]
            score = cl("auc_tf", yvalid, valid_preds)

        # Regression
        rg = RegressionMetrics()
        if metrics_name == "mae":
            valid_preds = model.predict(xvalid)
            score = rg("mae", yvalid, valid_preds)
        if metrics_name == "mse":
            valid_preds = model.predict(xvalid)
            score = rg("mse", yvalid, valid_preds)
        if metrics_name == "rmse":
            valid_preds = model.predict(xvalid)
            score = rg("rmse", yvalid, valid_preds)
        if metrics_name == "msle":
            valid_preds = model.predict(xvalid)
            score = rg("msle", yvalid, valid_preds)
        if metrics_name == "rmsle":
            valid_preds = model.predict(xvalid)
            score = rg("rmsle", yvalid, valid_preds)
        if metrics_name == "r2":
            valid_preds = model.predict(xvalid)
            score = rg("r2", yvalid, valid_preds)
        return score

    def run(self, my_folds, useful_features, prep_list="--|--", optimize_on="--|--"):
        if optimize_on != "--|--":
            self.optimize_on = optimize_on
        if prep_list != "--|--":
            self.prep_list = prep_list
        my_folds1 = my_folds.copy()
        # test1  = test.copy()

        fold = self.optimize_on
        xtrain = my_folds1[my_folds1.fold != fold].reset_index(drop=True)
        xvalid = my_folds1[my_folds1.fold == fold].reset_index(drop=True)
        print(xtrain.shape, xvalid.shape)
        # xtest = test1.copy()
        # return
        target_name = self.locker["target_name"]
        ytrain = xtrain[target_name]
        yvalid = xvalid[target_name]

        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

        prep_dict = {
            "SiMe": SimpleImputer(strategy="mean"),
            "SiMd": SimpleImputer(strategy="median"),
            "SiMo": SimpleImputer(strategy="mode"),
            "Ro": RobustScaler(),
            "Sd": StandardScaler(),
        }
        for f in self.prep_list:
            if f in list(prep_dict.keys()):
                sc = prep_dict[f]
                xtrain = sc.fit_transform(xtrain)
                xvalid = sc.transform(xvalid)
            elif f == "Lg":
                xtrain = pd.DataFrame(xtrain, columns=useful_features)
                xvalid = pd.DataFrame(xvalid, columns=useful_features)
                # xtest = pd.DataFrame(xtest, columns=useful_features)
                for col in useful_features:
                    xtrain[col] = np.log1p(xtrain[col])
                    xvalid[col] = np.log1p(xvalid[col])
                    # xtest[col] = np.log1p(xtest[col])
            else:
                raise Exception(f"scaler {f} is invalid!")

        # create optuna study
        study = optuna.create_study(direction=self._aim, study_name=self.model_name)
        study.optimize(
            lambda trial: self.obj(trial, xtrain, ytrain, xvalid, yvalid),
            n_trials=self.n_trials,
        )  # it tries 50 different values to find optimal hyperparameter

        return study, self._random_state, []


if __name__ == "__main__":
    import optuna

    a = OptunaOptimizer()
