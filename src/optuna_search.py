from metrics import ClassificationMetrics
from metrics import RegressionMetrics
from collections import defaultdict
import pickle
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score,f1_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from scipy import stats
import gc
import psutil
import seaborn as sns
sns.set()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer,MinMaxScaler
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
from sklearn.linear_model import Ridge,Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBRegressor, XGBRFRegressor
import itertools
import optuna
from lightgbm import LGBMClassifier,LGBMRegressor
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
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate,Bidirectional
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization,LSTM
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import warnings
#Filter up and down
np.random.seed(1337) # for reproducibility
warnings.filterwarnings("ignore")

            # "accuracy": self._accuracy,
            # "f1": self._f1,
            # "recall": self._recall,
            # "precision": self._precision,
            # "auc": self._auc,
            # "logloss": self._logloss,
            # "auc_tf": self._auc_tf,

            # "mae": self._mae,
            # "mse": self._mse,
            # "rmse": self._rmse,
            # "msle": self._msle,
            # "rmsle": self._rmsle,
            # "r2": self._r2,

class OptunaOptimizer:
    def __init__(self, model_name= "lgr",comp_type="2class",metrics_name="accuracy", aim='maximize',n_trials=50,optimize_on=0):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
            a = pickle.load(f)
        self.locker = a 

        self.comp_list = ["regression", "2class","multi_class", "multi_label"]
        self.metrics_list = ["accuracy","f1","recall","precision", "auc", "logloss","auc_tf","mae","mse","rmse","msle","rmsle","r2"]
        self.model_list = ["lgr","lir","xgbc","xgbr"]

        self.comp_type = comp_type
        self.metrics_name = metrics_name
        self.aim = aim
        self.n_trials = 50 
        self.best_params = None
        self.best_value = None
        self.model_name =  "lgr" 
        self.optimize_on = optimize_on
        self.sanity_check()

    def show(self):
        print(f"comp_type: {self.comp_type}")
        print(f"metrics_name: {self.metrics_name}")
        print(f"aim: {self.aim}")
        print(f"n_trials: {self.n_trials}")
        print(f"best_params: {self.best_params}")
        print(f"best_params: {self.best_params}")
        print(f"model_name: {self.model_name}")
        print(f"optimize_on: {self.optimize_on}")

    def sanity_check(self):
        if self.comp_type not in self.comp_list:
            raise Exception(f"{self.comp_type} not in the list {self.comp_list}")
        if self.metrics_name not in self.metrics_list:
            raise Exception( f"{self.metrics_name} not in the list {self.metrics_name}")
        if self.model_name not in self.model_list:
            raise Exception( f"{self.model_name} not in the list {self.model_list}")
        if self.optimize_on >= self.locker['no_folds']:
            raise Exception( f"{self.optimize_on} out of range {self.locker['no_folds']}")

    def help(self):
        print("comp_type:=> ",[comp for i,comp in enumerate(self.comp_list)])
        print("metrics_name:=>",[mt for i,mt in enumerate(self.metrics_list)])
        print("model_name:=>",[mt for i,mt in enumerate(self.model_list)])

    def generate_random_no(self):
        comp_random_state = self.locker["random_state"]
        total_no_folds = self.locker["no_folds"]
        fold_on = self.optimize_on
        metric_no = self.metrics_list.index(self.metrics_name)
        comp_type_no = self.comp_list.index(self.comp_type)
        model_no = self.model_list.index(self.model_name)
        # round_on 
        # level_on 
        # 
        seed = comp_random_state + total_no_folds * 2 + fold_on* 3 + metric_no*4 
        seed += comp_type_no * 5 + moel_no * 6 # + round_on * 4 + level_on * 5

        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        return np.random.randint(3,1000) # it should return 5

    def get_params(self,trial):
        model_name = self.model_name 
        if model_name == "lgr":
                params = {
                            "class_weight": trial.suggest_categorical("class_weight", ['balanced',None,{1:1,0:(sum(list(ytrain==0))/sum(list(ytrain==1)))}]),
                            "penalty": trial.suggest_categorical("penalty", ['l1','l2']),
                            'C': trial.suggest_float('c',.01,1000)
                        } 
            return LogisticRegression(**params, random_state)
        if model_name == "lir":
            return LinearRegression(**params, random_state)
        if model_name == "xgbc":
            return XGBClassifier(**params, random_state)
        if model_name == "xgbr":
            return XGBRegressor(**params, random_state)        
    
    def get_model(self,params):
        #["lgr","lir","xgbc","xgbr"]
        model_name = self.model_name
        random_state = generate_random_no(self)
        if model_name == "lgr":
            return LogisticRegression(**params, random_state)
        if model_name == "lir":
            return LinearRegression(**params, random_state)
        if model_name == "xgbc":
            return XGBClassifier(**params, random_state)
        if model_name == "xgbr":
            return XGBRegressor(**params, random_state)

        
            
        list_models = []

    def obj(self,trial,xtrain,ytrain,xvalid,yvalid):
        
        params = get_params(self)  
            
        model = CatBoostClassifier(**params, random_state=141)

        model.fit(xtrain, ytrain)

        valid_preds = model.predict_proba(xvalid)[:,1]
        valid_preds = (valid_preds > 0.5).astype(int)
        #test_preds = model.predict(xtest)
        score = accuracy_score(yvalid, valid_preds)  # since it is a classification problem so we will use roc auc score


        return score

    def run(self, my_folds, useful_features, optimize_on="--|--"):
        if optimize_on != "--|--":
            self.optimize_on = optimize_on 

        my_folds1 = my_folds.copy()
        #test1  = test.copy()

        fold= self.optimize_on
        xtrain = my_folds[my_folds1.fold != fold].reset_index(drop=True)
        xvalid = my_folds[my_folds1.fold == fold].reset_index(drop=True)
        print(xtrain.shape, xvalid.shape)
        #xtest = test1.copy()
        #return 
        target_name = self.locker["target_name"]
        ytrain = xtrain[target_name]
        yvalid = xvalid[target_name]

        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

        ## preprocess
        si = SimpleImputer(strategy='median')
        xtrain = si.fit_transform(xtrain)
        xvalid = si.transform(xvalid)
        #xtest = si.transform(xtest)

        # scale
        ss = MinMaxScaler()
        xtrain = ss.fit_transform(xtrain)
        xvalid = ss.transform(xvalid)
        #xtest = ss.transform(xtest)

        xtrain = pd.DataFrame(xtrain, columns=useful_features)
        xvalid = pd.DataFrame(xvalid, columns=useful_features)
        #xtest = pd.DataFrame(xtest, columns=useful_features)
        
    #     for col in useful_features:
    #         xtrain[col] = np.log1p(xtrain[col])
    #         xvalid[col] = np.log1p(xvalid[col])
    #         #xtest[col] = np.log1p(xtest[col])
            
        #create optuna study
        study = optuna.create_study(
            direction=self.aim,
            study_name= self.model_name
        )

        study.optimize( lambda trial: obj(trial,xtrain,ytrain,xvalid,yvalid),n_trials= self.n_trials ) # it tries 50 different values to find optimal hyperparameter
        
        return study

if __name__ == "__main__":
    import optuna
    a = OptunaOptimizer()