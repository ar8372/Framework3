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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
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
        with_gpu=False,
        save_models= True,
    ):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.locker = self.load_pickle(f"../models_{comp_name}/locker.pkl")
        self.current_dict = self.load_pickle(f"../models_{comp_name}/current_dict.pkl")
        self.save_models = save_models
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
        self.model_list = [
            "lgr",
            "lir",
            "xgbc",
            "xgbr",
            "cbc",
            "mlpc",
            "rg",
            "ls",
            "knnc",
            "dtc",
            "adbc",
            "gbmc",
            "hgbc",
            "lgbmc",
            "lgbmr",
            "rfc",
            "k1",
        ]
        self._prep_list = ["SiMe", "SiMd", "SiMo", "Mi", "Ro", "Sd", "Lg"]
        self.prep_list = prep_list
        self.comp_type = comp_type
        self.metrics_name = metrics_name
        self.with_gpu = with_gpu
        self._log_table = None # will track experiments
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
        self.n_trials = n_trials
        self.best_params = None
        self.best_value = None
        self.model_name = model_name
        self.optimize_on = optimize_on
        self.sanity_check()

    def save_pickle(self, path, to_dump):
        with open(path, "wb") as f:
            pickle.dump(to_dump, f)

    def load_pickle(self, path):
        with open(path, "rb") as f:
            o = pickle.load(f)
        return o

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
                "eta": trial.suggest_discrete_uniform("eta", 0.01, 0.1, 0.01),
                "reg_alpha": trial.suggest_int("reg_alpha", 1, 50),
                "reg_lambda": trial.suggest_int("reg_lambda", 5, 100),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            }
            return params
        if model_name == "xgbc":
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "subsample": trial.suggest_discrete_uniform(
                    "subsample", 0.6, 1.0, 0.05
                ),
                "n_estimators": trial.suggest_int("n_estimators", 1000, 10000, 100),
                "eta": trial.suggest_discrete_uniform("eta", 0.01, 0.1, 0.01),
                "reg_alpha": trial.suggest_int("reg_alpha", 1, 50),
                "reg_lambda": trial.suggest_int("reg_lambda", 5, 100),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 20),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
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
            params = {
                "iterations": trial.suggest_int("iterations", 300, 1200),
                "objective": trial.suggest_categorical(
                    "objective", ["Logloss", "CrossEntropy"]
                ),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                ),
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
            if with_gpu == True:
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
                "min_samples_split": trial.suggest_float("min_sample_split", 0.1, 0.5),
                "min_samples_leaf": trial.suggest_float("min_sample_split", 0.1, 0.5),
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
                "max_iter": trial.suggest_categorical("max_iter", [10000]),
                "max_depth": trial.suggest_int("max_depth", 2, 30),
                "max_bins": trial.suggest_int("max_bins", 100, 255),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100000),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 80),
            }
            return params

        if model_name == "lgbmc":
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
            }
            if with_gpu == True:
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
            if with_gpu == True:
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

        if model_name == "k1":
            params = {
                "epochs": trial.suggest_int("epochs", 5, 55, step=5, log=False),  # 5,55
                "batchsize": trial.suggest_int("batchsize", 8, 40, step=16, log=False),
                "learning_rate": trial.suggest_uniform("learning_rate", 0.002, 1),
                "o": trial.suggest_categorical("o", [True, False]),
            }
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

    # def update_table(self):
    #     self.Table.loc[Table.shape[0], :] = [
    #         0,
    #         10 ** (-1 * learning_rate),
    #         learning_rate,
    #         epochs,
    #         batch_size,
    #         no_hidden_layers,
    #         dropout_placeholder,
    #         units_placeholder,
    #         batch_norm_placeholder,
    #         activation_placeholder,
    #     ]

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
        if model_name == "rg":
            return Ridge(**params, random_state=self._random_state)
        if model_name == "ls":
            return Lasso(**params, random_state=self._random_state)
        if model_name == "knnc":
            return KNeighborsClassifier(**params)
        if model_name == "dtc":
            return DecisionTreeClassifier(**params, random_state=self._random_state)
        if model_name == "adbc":
            return AdaBoostClassifier(**params, random_state=self._random_state)
        if model_name == "gbmc":
            return GradientBoostingClassifier(**params, random_state=self._random_state)
        if model_name == "hgbc":
            return HistGradientBoostingClassifier(
                **params, random_state=self._random_state
            )
        if model_name == "lgbmc":
            return LGBMClassifier(**params, random_state=self._random_state)
        if model_name == "lgbmr":
            return LGBMRegressor(**params, random_state=self._random_state)
        if model_name == "rfc":
            return RandomForestClassifier(**params, random_state=self._random_state)
        if model_name == "k1":
            return self._k1(params)
        else:
            raise Exception(f"{model_name} is invalid!")

    def _k1(self, params):
        # simple model
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(16, activation="relu"))
        model.add(BatchNormalization())
        adam = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
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
                model.compile(
                    loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"]
                )
            else:
                # binary with one hot y no more binary problem so it is like multi class := don't use this case use above one instead
                model.add(Dense(self.ytrain.shape[1], activation="softmax"))
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=adam,
                    metrics=["accuracy"],
                )
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
                model.compile(loss=loss, optimizer=adam, metrics=metrics)
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
                model.compile(loss=loss, optimizer=adam, metrics=metrics)
        elif self.comp_type == "multi_label":
            model.add(Dense(self.ytrain.shape[1], activation="sigmoid"))
            model.compile(
                loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
            )

        return model

    def save_logs(self, params):
        if self._log_table is None:
            # not initialized
            self._log_table = pd.DataFrame(columns=list(params.keys()))
        self._log_table.loc[self._log_table.shape[0],:] = list(params.values())


    def obj(self, trial):

        params = self.get_params(trial)
        # Let's save these values 
        self.save_logs(params)
        model = self.get_model(params)

        if self.model_name == "lgbmr":
            model.fit(
                self.xtrain,
                self.ytrain,
                eval_set=[(self.xvalid, self.yvalid)],
                eval_metric="auc",
                early_stopping_rounds=1000,
                callbacks=[LightGBMPruningCallback(trial, "auc")],
                verbose=0,
            )
        if self.model_name == "k1":
            stop = EarlyStopping(monitor="auc", mode="max", patience=50, verbose=1)
            checkpoint = ModelCheckpoint(
                filepath="./",
                save_weights_only=True,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_auc", factor=0.5, patience=5, min_lr=0.00001, verbose=1
            )
            history = model.fit(
                x=self.xtrain,
                y=self.ytrain,
                batch_size=params["batchsize"],
                epochs=params["epochs"],
                shuffle=True,
                validation_split=0.15,
                callbacks=[stop, checkpoint, reduce_lr],
            )
        else:
            model.fit(self.xtrain, self.ytrain)

        metrics_name = self.metrics_name
        # Classification
        cl = ClassificationMetrics()
        if metrics_name == "auc":
            if self.comp_type == "2class":
                valid_preds = model.predict_proba(self.xvalid)[:, 1]
            else:
                valid_preds = model.predict_proba(self.xvalid)
            score = cl("auc", self.yvalid, valid_preds)
        if metrics_name == "accuracy":
            valid_preds = model.predict(self.xvalid)
            score = cl("accuracy", self.yvalid, valid_preds)
        if metrics_name == "f1":
            valid_preds = model.predict(self.xvalid)
            score = cl("f1", self.yvalid, valid_preds)
        if metrics_name == "recall":
            valid_preds = model.predict(self.xvalid)
            score = cl("recall", self.yvalid, valid_preds)
        if metrics_name == "precision":
            valid_preds = model.predict(self.xvalid)
            score = cl("precision", self.yvalid, valid_preds)
        if metrics_name == "logloss":
            if self.comp_type == "2class":
                valid_preds = model.predict_proba(self.xvalid)[:, 1]
            else:
                valid_preds = model.predict_proba(self.xvalid)
            score = cl("logloss", self.yvalid, valid_preds)
        if metrics_name == "auc_tf":
            if self.comp_type == "2class":
                valid_preds = model.predict_proba(self.xvalid)[:, 1]
            else:
                valid_preds = model.predict_proba(self.xvalid)
            score = cl("auc_tf", self.yvalid, valid_preds)

        # Regression
        rg = RegressionMetrics()
        if metrics_name == "mae":
            valid_preds = model.predict(self.xvalid)
            score = rg("mae", self.yvalid, valid_preds)
        if metrics_name == "mse":
            valid_preds = model.predict(self.xvalid)
            score = rg("mse", self.yvalid, valid_preds)
        if metrics_name == "rmse":
            valid_preds = model.predict(self.xvalid)
            score = rg("rmse", self.yvalid, valid_preds)
        if metrics_name == "msle":
            valid_preds = model.predict(self.xvalid)
            score = rg("msle", self.yvalid, valid_preds)
        if metrics_name == "rmsle":
            valid_preds = model.predict(self.xvalid)
            score = rg("rmsle", self.yvalid, valid_preds)
        if metrics_name == "r2":
            valid_preds = model.predict(self.xvalid)
            score = rg("r2", self.yvalid, valid_preds)
        return score

    def run(
        self,
        my_folds,
        useful_features,
        with_gpu="--|--",
        prep_list="--|--",
        optimize_on="--|--",
    ):
        if with_gpu != "--|--":
            self.with_gpu = with_gpu
        if optimize_on != "--|--":
            self.optimize_on = optimize_on
        if prep_list != "--|--":
            self.prep_list = prep_list
        my_folds1 = my_folds.copy()
        # test1  = test.copy()

        fold = self.optimize_on
        xtrain = my_folds1[my_folds1.fold != fold].reset_index(drop=True)
        xvalid = my_folds1[my_folds1.fold == fold].reset_index(drop=True)
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

        # create instances
        if self.model_name.startswith("k") and self.comp_type != "2class":
            ## to one hot
            ytrain = np_utils.to_categorical(ytrain)
            yvalid = np_utils.to_categorical(yvalid)
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xvalid = xvalid
        self.yvalid = yvalid
        # create optuna study
        study = optuna.create_study(direction=self._aim, study_name=self.model_name)
        study.optimize(
            lambda trial: self.obj(trial),
            n_trials=self.n_trials,
        )  # it tries 50 different values to find optimal hyperparameter

        if self.save_models == True:
            # let's save logs
            c =  self.current_dict['current_exp_no'] # optuna is called once in each exp so c+1 will be correct
            self.save_pickle(f"../models_{self.locker['comp_name']}/log_exp_{c+1}.pkl", self._log_table)
        return study, self._random_state


if __name__ == "__main__":
    import optuna

    a = OptunaOptimizer()
