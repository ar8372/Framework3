import pickle
import json
import os
import numpy as np
import pandas as pd 
import math
import random
import tensorflow as tf
import sys
import gc
import tracemalloc
import global_variables
import gzip
from datetime import datetime

# for animation
import itertools
import threading
import time
import sys
import global_variables 

# https://stackoverflow.com/questions/47152610/what-is-the-difference-between-xgb-train-and-xgb-xgbregressor-or-xgb-xgbclassif
import xgboost as xgb # when calling the low level api

# dart callback 
import joblib 

def save_pickle(path, to_dump):
    with open(path, "wb") as f:
        pickle.dump(to_dump, f)


def load_pickle(path):
    with open(path, "rb") as f:
        o = pickle.load(f)
    return o

def save_gzip(path, to_dump):
    with gzip.open(path, "wb") as f:
        f.write(to_dump)


def load_gzip(path):
    with gzip.open(path, "rb") as f:
        o = f.read(f)
    return o

def save_json(path, to_dump):
    json.dump( to_dump, open( path, 'w' ) )

def load_json(path):
    return json.load( open( path) )

def coln_3_1(arr):
    # array with three columns
    return np.array(arr).reshape(-1)




def fix_random(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(
        seed
    )  # f"The truth value of a {type(self).__name__} is ambiguous. "
    return seed  # np.random.randint(3, 1000) # it should return 5

def true_random(size):
    val = os.urandom(100)
    val = str(val)
    total = 0
    for i,v in enumerate(val):
        total += (i+1)*ord(v)
    return int(total) # sanity check

# https://www.kaggle.com/code/sietseschrder/xgboost-starter-0-793/notebook
# NEEDED WITH DeviceQuantileDMatrix BELOW
class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, batch_size=256*1024): #self, df=None, features=None, target=None, batch_size=256*1024):
        # self.features = features
        # self.target = target
        self.df = df # is a numpy 2D array (no_of_data_pts, [features,target])
        self.it = 0 # set iterator to 0
        self.batch_size = batch_size
        self.batches = int( np.ceil( df.shape[0] / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, self.df.shape[0] )
        dt = cudf.DataFrame(self.df[a:b]) # can use it without cudf
        input_data(data=dt[:,:-1], label=dt[:, -1]) #, weight=dt['weight']) # may need to reset few
        self.it += 1
        return 1


# generte random no time based but not good, repeats in fractions of seconds
def generate_random_no(adder="--|--"):
    # makes sure each time we are at differnet random state
    # random_state should only be used for reproducibility and should not give a better model
    seed = int(datetime.now().strftime("%H%M%S"))  # seed selected based on current time
    if adder != "--|--":
        seed += adder  # datetime can't give new no in vicinity of fraction of seconds so introducing this adder
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(
        seed
    )  # f"The truth value of a {type(self).__name__} is ambiguous. "
    return seed  # np.random.randint(3, 1000) # it should return 5

# https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
# https://stackoverflow.com/questions/633127/viewing-all-defined-variables
def sizeof_fmt(num, suffix="B"):
    """by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)

# https://stackoverflow.com/questions/22029562/python-how-to-make-simple-animated-loading-while-process-is-running
#here is the animation
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if global_variables.done:
            break
        sys.stdout.write('\rRefreshing ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!       ')
    sys.stdout.write('\n')


def check_memory_usage(title, object_name, verbose=1):
    # https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
    # https://stackoverflow.com/questions/633127/viewing-all-defined-variables
    if verbose > 0:
        print()
        print("*" * 40)
        print("{:>30}".format(title))
        print("*" * 40)
        mem = tracemalloc.get_traced_memory()
        print(
            "current, peak: {:>8} : {:>8}".format(
                sizeof_fmt(mem[0]), sizeof_fmt(mem[1])
            )
        )
    else:
        mem = tracemalloc.get_traced_memory()
        print(
            "{}=> current usage, peak usage: {:>8} : {:>8}".format(
                title, sizeof_fmt(mem[0]), sizeof_fmt(mem[1])
            )
        )
    if verbose > 0:
        print()
        # # Locals
        # print("Local Variables")
        # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
        #                         key= lambda x: -x[1])[:10]:
        #     print("{:>20}: {:>8}".format(name, sizeof_fmt(size)))
        # # globals
        # print("Global Variables")
        # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()),
        #                         key= lambda x: -x[1])[:10]:
        #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        if object_name != "--|--":
            # object
            print("object variables")
            for name, size in sorted(
                (
                    (name, sys.getsizeof(value))
                    for name, value in object_name.__dict__.items()
                ),
                key=lambda x: -x[1],
            )[:10]:
                print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        # #==>
        # print("List of object")
        # [o for o in gc.get_objects() if isinstance(o, Foo)]
        # for all_objects in gc.get_objects():
        #     for o in all_object
        #     name = o
        #     size = sys.getsizeof(o)
        #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        print("-" * 40)
        print()


def cosine_decay(epoch):
    if global_variables.epochs > 1:
        w = (1 + math.cos(epoch / (global_variables.epochs - 1) * math.pi)) / 2
    else:
        w = 1
    return w * global_variables.lr_start + (1 - w) * global_variables.lr_end


def exponential_decay(epoch):
    # v decays from e^a to 1 in every cycle
    # w decays from 1 to 0 in every cycle
    # epoch == 0                  -> w = 1 (first epoch of cycle)
    # epoch == epochs_per_cycle-1 -> w = 0 (last epoch of cycle)
    # higher a -> decay starts with a steeper decline
    a = 3
    CYCLES = global_variables.epochs //5 # 5 epochs in one cycle
    assert CYCLES < global_variables.epochs

    epochs_per_cycle = global_variables.epochs // CYCLES
    epoch_in_cycle = epoch % epochs_per_cycle
    if epochs_per_cycle > 1:
        v = math.exp(a * (1 - epoch_in_cycle / (epochs_per_cycle - 1)))
        w = (v - 1) / (math.exp(a) - 1)
    else:
        w = 1
    n = w * global_variables.lr_start + (1 - w) * global_variables.lr_end
    print("lr-->", n)
    return n

def get_test(input_dict,comp_name, useful_features, no_rows):
    # optimize_on : [3,1]
    features_in = []
    new_array = np.array([], dtype=np.int8).reshape(no_rows,0)
    for key,feat_list in input_dict.items():
        if key == "id_folds_target":
            continue 
        if all(x in useful_features for x in feat_list):
            # pull whole set
            features_in += feat_list 
            d1 = pd.read_parquet(f"../input/input-{comp_name}/test_{key}.parquet")
            new_array=np.concatenate((new_array, np.array(d1.values)), axis=1)
            # xtrain = np.array(my_folds[values].values)
            # xtrain= np.concatenate((xtrain, np.array(rest_train)), axis=1)
        elif any(x in useful_features for x in feat_list):
            # partial set 
            present = list(set(feat_list) & set(useful_features))
            features_in += present
            d1 = pd.read_parquet(f"../input/input-{comp_name}/test_{key}.parquet")[present]
            new_array=np.concatenate((new_array, np.array(d1.values)), axis=1)
        else:
            # none of values present 
            pass 
    # it is asking for full dataset 
    return new_array, features_in


def bottleneck_test(comp_name, useful_features,  return_type, verbose=0):
    ordered_list = []
    # There is a risk that we may not have same order of items in test vs train/valid
    # RETURNS TEST
    # base is a list 
    base = load_pickle(f"../configs/configs-{comp_name}/useful_features_l_1.pkl")
    input_dict = load_pickle(f"../input/input-{comp_name}/input_dict.pkl")
    # just picking a random set 
    no_rows = pd.read_parquet(f"../input/input-{comp_name}/test_{list(input_dict.keys())[1]}.parquet").shape[0]
    if all(x in base for x in useful_features):
        # all the features are in my_folds 
        # we need xtest preprocessed 
        xtest = None
        #xtest = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
        xtest, ordered_list  = get_test(input_dict,comp_name, useful_features, no_rows)
        xtest= pd.DataFrame(xtest, columns= ordered_list)[useful_features]
        #print(xtest.iloc[:5,:7])
        xtest = np.array(xtest.values)
        return xtest , useful_features      
    elif any(x in base for x in useful_features):
        # part in my_folds part outside 

        # we need xtest preprocessed
        xtest = None 
        #xtest = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
        #features_in = list(set(input_dict['base']) & set(useful_features))
        features_in = list(set(base) & set(useful_features))
        #features_in = list(set(xtest.columns) & set(useful_features))
        features_out = list(set(useful_features) - set(base)) 
        xtest, features_in  = get_test(input_dict,comp_name, features_in, no_rows)
        #xtest = np.array(xtest[features_in].values)
        rest_test = []
        ######################################################################################################
        """
         is it in oof or in features
         useful_features: 
            pred_e_121_fold5.pkl # experiment no is the unique identifier # one experiment can have multiple preds
            feat_l_1_f_12_std                 
            # feature no is the unique identifier # level no is good since we can have std feature
            # at two levels and it is good to seperate our features based on at which leve we are working 
            # for predictions the TABLE has a level column for identifying level
            feat_l_1_f_12_mean
        """
        # first filter features based on where they come from.
        useful_features_pred = [item for item in features_out if item.split("_")[0]=="pred"]
        useful_features_feat = [item for item in features_out if item.split("_")[0]=="feat"]
        # sanity check 
        assert len(useful_features_pred) + len(useful_features_feat) == len(features_out)



        # collecting preds
        for f in useful_features_pred:
            try: 
                rest_test.append(load_pickle(f"../configs/configs-{comp_name}/test_preds/test_{f}.pkl"))
                ordered_list.append(f)
            except:
                raise Exception(f"Feature: {f} not found")
        # collecting feats 
        for f in useful_features_feat:
            try:
                rest_test.append(load_pickle(f"../configs/configs-{comp_name}/test_feats/test_{f}.pkl"))
                ordered_list.append(f)
            except:
                raise Exception(f"Feature: {f} not found")

        if rest_test == []:
            raise Exception("Error in bottleneck")
        
        rest_test = np.array(rest_test).T
        #rest_test = np.array(rest_test).reshape(-1,len(features_out))

        del useful_features_feat, useful_features_pred 
        gc.collect()
        ###################################################################################################
        xtest = np.concatenate((xtest, rest_test), axis=1)
        ordered_list = features_in + ordered_list 
        xtest= pd.DataFrame(xtest, columns= ordered_list)[useful_features]
        xtest = np.array(xtest.values)
        return xtest, useful_features
    else:
        # all the features are present outside
        xtest = None
        # we need xtest preprocessed 
        rest_test = []
        ######################################################################################################
        """
         is it in oof or in features
         useful_features: 
            pred_e_121_fold5.pkl # experiment no is the unique identifier # one experiment can have multiple preds
            feat_l_1_f_12_std                 
            # feature no is the unique identifier # level no is good since we can have std feature
            # at two levels and it is good to seperate our features based on at which leve we are working 
            # for predictions the TABLE has a level column for identifying level
            feat_l_1_f_12_mean
        """
        # first filter features based on where they come from.
        useful_features_pred = [item for item in useful_features if item.split("_")[0]=="pred"]
        useful_features_feat = [item for item in useful_features if item.split("_")[0]=="feat"]
        # sanity check 
        assert len(useful_features_pred) + len(useful_features_feat) == len(useful_features)



        # collecting preds
        for f in useful_features_pred:
            try: 
                rest_test.append(load_pickle(f"../configs/configs-{comp_name}/test_preds/test_{f}.pkl"))
                ordered_list.append(f)
            except:
                raise Exception(f"Feature: {f} not found")
        # collecting feats 
        for f in useful_features_feat:
            try:
                rest_test.append(load_pickle(f"../configs/configs-{comp_name}/test_feats/test_{f}.pkl"))
                ordered_list.append(f)
            except:
                raise Exception(f"Feature: {f} not found")

        if rest_test == []:
            raise Exception("Error in bottleneck")
        
        xtest = np.array(rest_test).T
        #xtest = np.array(rest_test).reshape(-1,len(useful_features))

        del useful_features_feat, useful_features_pred 
        gc.collect()
        ###################################################################################################
        xtest= pd.DataFrame(xtest, columns= ordered_list)[useful_features]
        xtest = np.array(xtest.values)
        return xtest, useful_features

def get_train(input_dict,id_folds_target, comp_name, useful_features, fold_name, istrain,optimize_on=[]):
    # optimize_on : [3,1]
    no_cols = id_folds_target.shape[1]
    
    features_base = list(id_folds_target.columns)
    features_in = []
    for key,feat_list in input_dict.items():
        if key == "id_folds_target":
            continue 
        #
        if all(x in useful_features for x in feat_list):
            # pull whole set
            features_in += feat_list 
            d1 = pd.read_parquet(f"../input/input-{comp_name}/train_{key}.parquet")
            id_folds_target=np.concatenate((id_folds_target, np.array(d1.values)), axis=1)
            # xtrain = np.array(my_folds[values].values)
            # xtrain= np.concatenate((xtrain, np.array(rest_train)), axis=1)

        elif any(x in useful_features for x in feat_list):
            # partial set 
            present = list(set(feat_list) & set(useful_features))
            features_in += present
            d1 = pd.read_parquet(f"../input/input-{comp_name}/train_{key}.parquet")[present]
            id_folds_target=np.concatenate((id_folds_target, np.array(d1.values)), axis=1)
        else:
            # none of values present 
            pass 
        d1 = None
        
    if fold_name is not None:
        # we are taking partial fold 
        id_folds_target = pd.DataFrame(id_folds_target, columns=features_base+features_in)
        if istrain:
            # so train set everythign except 
            id_folds_target = np.array(id_folds_target[~id_folds_target[fold_name].isin(optimize_on) ].values)
        else:
            # valid set
            id_folds_target = np.array(id_folds_target[id_folds_target[fold_name].isin(optimize_on) ].values)
        return id_folds_target[:,no_cols:],id_folds_target[:,no_cols-1], features_in
    else:
        # it is asking for full dataset 
        return id_folds_target[:,no_cols:], id_folds_target[:,no_cols-1],features_in
        #      xtrain, ytrain, features_in




def bottleneck(comp_name,useful_features, fold_name, optimize_on, _state, return_type, verbose=0):
    ordered_list=[]
    # RETURNS XTRAIN, XVALID, YTRAIN, YVLID, XVALID_IDX
    locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
    #my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
    #test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
    # base is a list 
    # base is the combined input_dict is splitted
    base = load_pickle(f"../configs/configs-{comp_name}/useful_features_l_1.pkl")
    id_folds_target = pd.read_parquet(f"../input/input-{comp_name}/id_folds_target.parquet")
    input_dict = load_pickle(f"../input/input-{comp_name}/input_dict.pkl")
    """
    # NO NEED OF THIS CAME
    # it is a dictonary for each set of features created 
    # each set has a unique name as key 
    "l_1_f_0": ['column names generate', 'columns used to generate', 'title give to the process']
    "e_30" : [column_names_generated, columns_name_used_to_generate]
    "nan_count": [['nan_f34','nan_32'], ['f_1', 'f_2']]
    "base" : [['f_1', 'f_4'], 0]
    : column names generate pred_l_1_e_0, mean_f_24 etc 
    : columns used to generate
    : title given  nan_count, exp_2 etc
    """
    # First aim is to find where we have to search for the features is it in 
    # my_folds or it is not 
    # base is the list of all the column of my_folds
    a = load_pickle(f"../configs/configs-{comp_name}/features_dict.pkl")
    if all(x in base for x in useful_features):
        # all the features are in my_folds 
        if verbose != 0:
            print("all the features are in my_folds")
        #my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
        input_dict = load_pickle(f"../input/input-{comp_name}/input_dict.pkl")
        val_idx = None 
        if _state in ["fold", "opt"]:
            #  for seed we don't need to find val_idx

            val_idx = id_folds_target[id_folds_target[fold_name].isin(optimize_on)][locker["id_name"]].values.tolist()
            #val_idx = my_folds[my_folds[fold_name] == optimize_on][locker["id_name"]].values.tolist()
        if _state == "seed":
            xtrain, ytrain, temp_features = get_train(input_dict,id_folds_target.copy(), comp_name, useful_features, None, True,optimize_on)
            xtrain = pd.DataFrame(xtrain, columns=temp_features)[useful_features]
            xtrain = np.array(xtrain.values)            
            #xtrain = np.array(my_folds[useful_features].values)
            #ytrain = np.array(my_folds[locker["target_name"]].values)
            xvalid = None 
            yvalid = None 
            val_idx = None 
            return val_idx, xtrain, xvalid, ytrain, yvalid, useful_features  
        # xvalid [2] xtrain [0,1,2,3,4] #fold5 
        # #
        # fold_nos = [i for i in range(int(fold_name.split("d")[1]))]
        # train_fold_nos = list(set(fold_nos) - set(optimize_on))
        # print("This is train fold nos")
        # print(train_fold_nos)
        # print(optimize_on)
        # print("done")
        xtrain, ytrain, temp_features= get_train(input_dict,id_folds_target.copy(), comp_name, useful_features, fold_name, True,optimize_on)
        xtrain = pd.DataFrame(xtrain, columns=temp_features)[useful_features]
        xtrain = np.array(xtrain.values)
        #xtrain = np.array(my_folds[my_folds[fold_name] != optimize_on][useful_features].values)
        #ytrain = np.array(my_folds[my_folds[fold_name] != optimize_on][locker["target_name"]].values)
        xvalid, yvalid, temp_features= get_train(input_dict,id_folds_target.copy(), comp_name, temp_features, fold_name, False,optimize_on)
        xvalid = pd.DataFrame(xvalid, columns=temp_features)[useful_features]
        xvalid = np.array(xvalid.values)        
        #xvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][useful_features].values)
        #yvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][locker["target_name"]].values)

        
        return val_idx, xtrain, xvalid, ytrain, yvalid, useful_features
    elif any(x in base for x in useful_features):
        # part in my_folds part outside  
        if verbose != 0:
            print("part in my_folds part outside")
        #my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
        features_in = list(set(base) & set(useful_features))
        features_out = list(set(useful_features) - set(base))

        rest_train = []
        ######################################################################################################
        """
         is it in oof or in features
         useful_features: 
            pred_e_121_fold5.pkl # experiment no is the unique identifier # one experiment can have multiple preds
            feat_l_1_f_12_std                 
            # feature no is the unique identifier # level no is good since we can have std feature
            # at two levels and it is good to seperate our features based on at which leve we are working 
            # for predictions the TABLE has a level column for identifying level
            feat_l_1_f_12_mean
        """
        # first filter features based on where they come from.
        useful_features_pred = [item for item in features_out if item.split("_")[0]=="pred"]
        useful_features_feat = [item for item in features_out if item.split("_")[0]=="feat"]
        # sanity check 
        assert len(useful_features_pred) + len(useful_features_feat) == len(features_out)

        # This is the order they are accessed 
        #ordered_list = features_in + useful_features_pred + useful_features_feat


        # collecting preds
        for f in useful_features_pred:
            try: 
                rest_train.append(load_pickle(f"../configs/configs-{comp_name}/oof_preds/oof_{f}.pkl"))
            except:
                raise Exception(f"Feature: {f} not found")
        ordered_list += useful_features_pred 
        # collecting feats 
        for f in useful_features_feat:
            try:
                rest_train.append(load_pickle(f"../configs/configs-{comp_name}/train_feats/train_{f}.pkl"))
            except:
                raise Exception(f"Feature: {f} not found")
        ordered_list += useful_features_feat

        if rest_train == []:
            raise Exception("Error in bottleneck")
        
        rest_train = np.array(rest_train).T
        #rest_train = np.array(rest_train).reshape(-1,len(features_out))

        
        #del useful_features_feat, useful_features_pred 
        gc.collect()
        ###################################################################################################


        if _state == "seed":
            xtrain, ytrain, features_in= get_train(input_dict,id_folds_target.copy(), comp_name, features_in, None, True,optimize_on)
            #xtrain = np.array(my_folds[features_in].values)
            xtrain= np.concatenate((xtrain, np.array(rest_train)), axis=1)
            #ytrain = np.array(my_folds[locker["target_name"]].values)
            xvalid = None 
            yvalid = None 
            val_idx = None 
            ordered_list += features_in
            xtrain= pd.DataFrame(xtrain, columns= ordered_list)[useful_features].values
            return val_idx, xtrain, xvalid, ytrain, yvalid, useful_features

        #xtrain, ytrain, features_in= get_train(input_dict,id_folds_target.copy(), comp_name, features_in, fold_name, True,optimize_on)
        #xtrain = np.array(my_folds[my_folds[fold_name] != optimize_on][features_in].values)
        #ytrain = np.array(my_folds[my_folds[fold_name] != optimize_on][locker["target_name"]].values)

        #xvalid, yvalid, features_in= get_train(input_dict,id_folds_target.copy(), comp_name, features_in, fold_name, False,optimize_on)
        #xvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][features_in].values)
        #yvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][locker["target_name"]].values)




        #mask  = np.array((my_folds[fold_name] != optimize_on).values, dtype=bool)
        mask = np.array(~id_folds_target[fold_name].isin(optimize_on).values, dtype=bool)
        xtrain, ytrain, features_in= get_train(input_dict,id_folds_target.copy(), comp_name, features_in, fold_name, True,optimize_on)
        #xtrain= np.concatenate((xtrain, rest_train[mask].values), axis=1)
        xtrain= np.concatenate((xtrain, rest_train[mask]), axis=1)
        ordered_list = features_in + ordered_list 
        #xtrain= np.concatenate((xtrain, np.array(rest_train[(my_folds[fold_name] != optimize_on).values])), axis=1)
        #ytrain = np.array(my_folds[my_folds[fold_name] != optimize_on][locker["target_name"]].values)

        mask = np.array(id_folds_target[fold_name].isin(optimize_on).values, dtype=bool)
        xvalid, yvalid, features_in= get_train(input_dict,id_folds_target.copy(), comp_name, features_in, fold_name, False,optimize_on)
        xvalid = np.concatenate((xvalid, rest_train[mask]), axis=1)
        #xvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][features_in].values)
        #xvalid = np.concatenate((xvalid, np.array(rest_train[(my_folds[fold_name] == optimize_on).values])), axis=1)
        #yvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][locker["target_name"]].values)

        
        val_idx = None 
        if _state in ["fold", "opt"]:
            #  for seed we don't need to find val_idx
            val_idx= id_folds_target[id_folds_target[fold_name].isin(optimize_on)][locker["id_name"]].values.tolist()
            #val_idx = my_folds[my_folds[fold_name] == optimize_on][locker["id_name"]].values.tolist()
        

        xtrain= pd.DataFrame(xtrain, columns= ordered_list)[useful_features].values
        xvalid= pd.DataFrame(xvalid, columns= ordered_list)[useful_features].values
        return val_idx, xtrain, xvalid, ytrain, yvalid, useful_features

    else:
        # all the features are present outside 
        #my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
        if verbose != 0:
            print("all the features are present outside")
        rest_train = [] # append 1D arrays
        ######################################################################################################
        """
         is it in oof or in features
         useful_features: 
            pred_e_121_fold5.pkl # experiment no is the unique identifier # one experiment can have multiple preds
            feat_l_1_f_12_std                 
            # feature no is the unique identifier # level no is good since we can have std feature
            # at two levels and it is good to seperate our features based on at which leve we are working 
            # for predictions the TABLE has a level column for identifying level
            feat_l_1_f_12_mean
        """
        # first filter features based on where they come from.
        useful_features_pred = [item for item in useful_features if item.split("_")[0]=="pred"]
        useful_features_feat = [item for item in useful_features if item.split("_")[0]=="feat"]
        # sanity check 
        assert len(useful_features_pred) + len(useful_features_feat) == len(useful_features)

        #ordered_list = useful_features_pred + useful_features_feat 
        # collecting preds
        for f in useful_features_pred:
            try: 
                rest_train.append(load_pickle(f"../configs/configs-{comp_name}/oof_preds/oof_{f}.pkl"))
            except:
                raise Exception(f"Feature: {f} not found")
        ordered_list += useful_features_pred 
        # collecting feats 
        for f in useful_features_feat:
            try:
                rest_train.append(load_pickle(f"../configs/configs-{comp_name}/train_feats/train_{f}.pkl"))
            except:
                raise Exception(f"Feature: {f} not found")
        ordered_list += useful_features_feat 

        if rest_train == []:
            raise Exception("Error in bottleneck")
        # print(np.array(rest_train)[:3,:])
        # print("check")
        # print(np.array(rest_train).T[:3,:])
        # print("check2")
        # rest_train = np.array(rest_train).reshape(-1,len(useful_features))
        # print(rest_train[:3,:])
        rest_train = np.array(rest_train).T 

        #del useful_features_feat, useful_features_pred 
        #del useful_features 
        gc.collect()
        ###################################################################################################
        if _state == "seed":
            xtrain= rest_train
            xtrain= pd.DataFrame(xtrain, columns= ordered_list)[useful_features].values
            ytrain = np.array(my_folds[locker["target_name"]].values)
            xvalid = None 
            yvalid = None 
            val_idx = None 

            return val_idx, xtrain, xvalid, ytrain, yvalid , useful_features 

 
        mask = np.array(~id_folds_target[fold_name].isin(optimize_on).values, dtype=bool)
        #mask  = np.array((my_folds[fold_name] != optimize_on).values, dtype=bool)
        xtrain= rest_train[mask]
        ytrain = np.array(id_folds_target[mask][locker["target_name"]].values)

        mask = np.array(id_folds_target[fold_name].isin(optimize_on).values, dtype=bool)
        xvalid = rest_train[mask]
        yvalid = np.array(id_folds_target[mask][locker["target_name"]].values)

  
        val_idx = None 

        if _state in ["fold", "opt"]:
            #  for seed we don't need to find val_idx
            val_idx= id_folds_target[id_folds_target[fold_name].isin(optimize_on)][locker["id_name"]].values.tolist()
            #val_idx = my_folds[my_folds[fold_name] == optimize_on][locker["id_name"]].values.tolist()


        xtrain= pd.DataFrame(xtrain, columns= ordered_list)[useful_features].values
        xvalid= pd.DataFrame(xvalid, columns= ordered_list)[useful_features].values
        return val_idx, xtrain, xvalid, ytrain, yvalid , useful_features    



# def bottleneck_test(comp_name, useful_features,  return_type, verbose=0):
#     ordered_list = []
#     # There is a risk that we may not have same order of items in test vs train/valid
#     # RETURNS TEST
#     # base is a list 
#     base = load_pickle(f"../configs/configs-{comp_name}/useful_features_l_1.pkl")
#     if all(x in base for x in useful_features):
#         # all the features are in my_folds 
#         # we need xtest preprocessed 
#         xtest = None
#         xtest = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
#         xtest = np.array(xtest[useful_features].values)
#         return xtest , useful_features      
#     elif any(x in base for x in useful_features):
#         # part in my_folds part outside 

#         # we need xtest preprocessed
#         xtest = None 
#         xtest = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
#         features_in = list(set(xtest.columns) & set(useful_features))
#         features_out = list(set(useful_features) - set(xtest.columns)) 
#         xtest = np.array(xtest[features_in].values)
#         rest_test = []
#         ######################################################################################################
#         """
#          is it in oof or in features
#          useful_features: 
#             pred_e_121_fold5.pkl # experiment no is the unique identifier # one experiment can have multiple preds
#             feat_l_1_f_12_std                 
#             # feature no is the unique identifier # level no is good since we can have std feature
#             # at two levels and it is good to seperate our features based on at which leve we are working 
#             # for predictions the TABLE has a level column for identifying level
#             feat_l_1_f_12_mean
#         """
#         # first filter features based on where they come from.
#         useful_features_pred = [item for item in features_out if item.split("_")[0]=="pred"]
#         useful_features_feat = [item for item in features_out if item.split("_")[0]=="feat"]
#         # sanity check 
#         assert len(useful_features_pred) + len(useful_features_feat) == len(features_out)



#         # collecting preds
#         for f in useful_features_pred:
#             try: 
#                 rest_test.append(load_pickle(f"../configs/configs-{comp_name}/test_preds/test_{f}.pkl"))
#                 ordered_list.append(f)
#             except:
#                 raise Exception(f"Feature: {f} not found")
#         # collecting feats 
#         for f in useful_features_feat:
#             try:
#                 rest_test.append(load_pickle(f"../configs/configs-{comp_name}/test_feats/test_{f}.pkl"))
#                 ordered_list.append(f)
#             except:
#                 raise Exception(f"Feature: {f} not found")

#         if rest_test == []:
#             raise Exception("Error in bottleneck")
#         rest_test = np.array(rest_test).reshape(-1,len(features_out))

#         del useful_features_feat, useful_features_pred 
#         gc.collect()
#         ###################################################################################################
#         xtest = np.concatenate((xtest, rest_test), axis=1)
#         ordered_list = features_in + ordered_list 
#         xtest= pd.DataFrame(xtest, columns= ordered_list)[useful_features].values
#         return xtest, useful_features
#     else:
#         # all the features are present outside
#         xtest = None
#         # we need xtest preprocessed 
#         rest_test = []
#         ######################################################################################################
#         """
#          is it in oof or in features
#          useful_features: 
#             pred_e_121_fold5.pkl # experiment no is the unique identifier # one experiment can have multiple preds
#             feat_l_1_f_12_std                 
#             # feature no is the unique identifier # level no is good since we can have std feature
#             # at two levels and it is good to seperate our features based on at which leve we are working 
#             # for predictions the TABLE has a level column for identifying level
#             feat_l_1_f_12_mean
#         """
#         # first filter features based on where they come from.
#         useful_features_pred = [item for item in useful_features if item.split("_")[0]=="pred"]
#         useful_features_feat = [item for item in useful_features if item.split("_")[0]=="feat"]
#         # sanity check 
#         assert len(useful_features_pred) + len(useful_features_feat) == len(useful_features)



#         # collecting preds
#         for f in useful_features_pred:
#             print(f,"got it")
#             try: 
#                 rest_test.append(load_pickle(f"../configs/configs-{comp_name}/test_preds/test_{f}.pkl"))
#                 ordered_list.append(f)
#             except:
#                 raise Exception(f"Feature: {f} not found")
#         # collecting feats 
#         for f in useful_features_feat:
#             try:
#                 rest_test.append(load_pickle(f"../configs/configs-{comp_name}/test_feats/test_{f}.pkl"))
#                 ordered_list.append(f)
#             except:
#                 raise Exception(f"Feature: {f} not found")

#         if rest_test == []:
#             raise Exception("Error in bottleneck")
#         xtest = np.array(rest_test).reshape(-1,len(useful_features))

#         del useful_features_feat, useful_features_pred 
#         gc.collect()
#         ###################################################################################################
#         xtest= pd.DataFrame(xtest, columns= ordered_list)[useful_features].values
#         return xtest, useful_features


# def bottleneck(comp_name,useful_features, fold_name, optimize_on, _state, return_type, verbose=0):
#     ordered_list=[]
#     # RETURNS XTRAIN, XVALID, YTRAIN, YVLID, XVALID_IDX
#     locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
#     #my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
#     #test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
#     # base is a list 
#     base = load_pickle(f"../configs/configs-{comp_name}/useful_features_l_1.pkl")
#     """
#     # NO NEED OF THIS CAME
#     # it is a dictonary for each set of features created 
#     # each set has a unique name as key 
#     "l_1_f_0": ['column names generate', 'columns used to generate', 'title give to the process']
#     "e_30" : [column_names_generated, columns_name_used_to_generate]
#     "nan_count": [['nan_f34','nan_32'], ['f_1', 'f_2']]
#     "base" : [['f_1', 'f_4'], 0]
#     : column names generate pred_l_1_e_0, mean_f_24 etc 
#     : columns used to generate
#     : title given  nan_count, exp_2 etc
#     """
#     # First aim is to find where we have to search for the features is it in 
#     # my_folds or it is not 
#     # base is the list of all the column of my_folds
#     a = load_pickle(f"../configs/configs-{comp_name}/features_dict.pkl")
#     if all(x in base for x in useful_features):
#         # all the features are in my_folds 
#         if verbose != 0:
#             print("all the features are in my_folds")
#         my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")

#         val_idx = None 
#         if _state in ["fold", "opt"]:
#             #  for seed we don't need to find val_idx
#             val_idx = my_folds[my_folds[fold_name] == optimize_on][locker["id_name"]].values.tolist()
#         if _state == "seed":
#             xtrain = np.array(my_folds[useful_features].values)
#             ytrain = np.array(my_folds[locker["target_name"]].values)
#             xvalid = None 
#             yvalid = None 
#             val_idx = None 
#             return val_idx, xtrain, xvalid, ytrain, yvalid, useful_features
#         xtrain = np.array(my_folds[my_folds[fold_name] != optimize_on][useful_features].values)
#         ytrain = np.array(my_folds[my_folds[fold_name] != optimize_on][locker["target_name"]].values)

#         xvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][useful_features].values)
#         yvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][locker["target_name"]].values)


        
#         return val_idx, xtrain, xvalid, ytrain, yvalid, useful_features
#     elif any(x in base for x in useful_features):
#         # part in my_folds part outside  
#         if verbose != 0:
#             print("part in my_folds part outside")
#         my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
#         features_in = list(set(my_folds.columns) & set(useful_features))
#         features_out = list(set(useful_features) - set(my_folds.columns))

#         rest_train = []
#         ######################################################################################################
#         """
#          is it in oof or in features
#          useful_features: 
#             pred_e_121_fold5.pkl # experiment no is the unique identifier # one experiment can have multiple preds
#             feat_l_1_f_12_std                 
#             # feature no is the unique identifier # level no is good since we can have std feature
#             # at two levels and it is good to seperate our features based on at which leve we are working 
#             # for predictions the TABLE has a level column for identifying level
#             feat_l_1_f_12_mean
#         """
#         # first filter features based on where they come from.
#         useful_features_pred = [item for item in features_out if item.split("_")[0]=="pred"]
#         useful_features_feat = [item for item in features_out if item.split("_")[0]=="feat"]
#         # sanity check 
#         assert len(useful_features_pred) + len(useful_features_feat) == len(features_out)

#         # This is the order they are accessed 
#         ordered_list = features_in + useful_features_pred + useful_features_feat


#         # collecting preds
#         for f in useful_features_pred:
#             try: 
#                 rest_train.append(load_pickle(f"../configs/configs-{comp_name}/oof_preds/oof_{f}.pkl"))
#             except:
#                 raise Exception(f"Feature: {f} not found")
#         # collecting feats 
#         for f in useful_features_feat:
#             try:
#                 rest_train.append(load_pickle(f"../configs/configs-{comp_name}/train_feats/train_{f}.pkl"))
#             except:
#                 raise Exception(f"Feature: {f} not found")

#         if rest_train == []:
#             raise Exception("Error in bottleneck")
#         rest_train = np.array(rest_train).reshape(-1,len(features_out))

        
#         del useful_features_feat, useful_features_pred 
#         gc.collect()
#         ###################################################################################################

#         if rest_train == []:
#             raise Exception("Error in bottleneck")
#         rest_train = np.array(rest_train).reshape(-1,len(features_out))

#         if _state == "seed":
#             xtrain = np.array(my_folds[features_in].values)
#             xtrain= np.concatenate((xtrain, np.array(rest_train)), axis=1)
#             ytrain = np.array(my_folds[locker["target_name"]].values)
#             xvalid = None 
#             yvalid = None 
#             val_idx = None 
#             xtrain= pd.DataFrame(xtrain, columns= ordered_list)[useful_features].values
#             return val_idx, xtrain, xvalid, ytrain, yvalid, useful_features

#         xtrain = np.array(my_folds[my_folds[fold_name] != optimize_on][features_in].values)
#         ytrain = np.array(my_folds[my_folds[fold_name] != optimize_on][locker["target_name"]].values)

#         xvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][features_in].values)
#         yvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][locker["target_name"]].values)




#         mask  = np.array((my_folds[fold_name] != optimize_on).values, dtype=bool)
#         xtrain= np.concatenate((xtrain, np.array(rest_train[(my_folds[fold_name] != optimize_on).values])), axis=1)
#         ytrain = np.array(my_folds[my_folds[fold_name] != optimize_on][locker["target_name"]].values)

#         xvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][features_in].values)
#         xvalid = np.concatenate((xvalid, np.array(rest_train[(my_folds[fold_name] == optimize_on).values])), axis=1)
#         yvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][locker["target_name"]].values)

        
#         val_idx = None 
#         if _state in ["fold", "opt"]:
#             #  for seed we don't need to find val_idx
#             val_idx = my_folds[my_folds[fold_name] == optimize_on][locker["id_name"]].values.tolist()
        

#         xtrain= pd.DataFrame(xtrain, columns= ordered_list)[useful_features].values
#         xvalid= pd.DataFrame(xvalid, columns= ordered_list)[useful_features].values
#         return val_idx, xtrain, xvalid, ytrain, yvalid, useful_features

#     else:
#         # all the features are present outside 
#         my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
#         if verbose != 0:
#             print("all the features are present outside")
#         rest_train = [] # append 1D arrays
#         ######################################################################################################
#         """
#          is it in oof or in features
#          useful_features: 
#             pred_e_121_fold5.pkl # experiment no is the unique identifier # one experiment can have multiple preds
#             feat_l_1_f_12_std                 
#             # feature no is the unique identifier # level no is good since we can have std feature
#             # at two levels and it is good to seperate our features based on at which leve we are working 
#             # for predictions the TABLE has a level column for identifying level
#             feat_l_1_f_12_mean
#         """
#         # first filter features based on where they come from.
#         useful_features_pred = [item for item in useful_features if item.split("_")[0]=="pred"]
#         useful_features_feat = [item for item in useful_features if item.split("_")[0]=="feat"]
#         # sanity check 
#         assert len(useful_features_pred) + len(useful_features_feat) == len(useful_features)

#         ordered_list = useful_features_pred + useful_features_feat 

#         # collecting preds
#         for f in useful_features_pred:
#             try: 
#                 rest_train.append(load_pickle(f"../configs/configs-{comp_name}/oof_preds/oof_{f}.pkl"))
#             except:
#                 raise Exception(f"Feature: {f} not found")
#         # collecting feats 
#         for f in useful_features_feat:
#             try:
#                 rest_train.append(load_pickle(f"../configs/configs-{comp_name}/train_feats/train_{f}.pkl"))
#             except:
#                 raise Exception(f"Feature: {f} not found")

#         if rest_train == []:
#             raise Exception("Error in bottleneck")
#         rest_train = np.array(rest_train).reshape(-1,len(useful_features))

#         del useful_features_feat, useful_features_pred 
#         del useful_features 
#         gc.collect()
#         ###################################################################################################
#         if _state == "seed":
#             xtrain= rest_train
#             xtrain= pd.DataFrame(xtrain, columns= ordered_list)[useful_features].values
#             ytrain = np.array(my_folds[locker["target_name"]].values)
#             xvalid = None 
#             yvalid = None 
#             val_idx = None 

#             return val_idx, xtrain, xvalid, ytrain, yvalid , useful_features 

#         mask  = np.array((my_folds[fold_name] != optimize_on).values, dtype=bool)
#         xtrain= rest_train[(my_folds[fold_name] != optimize_on).values]
#         ytrain = np.array(my_folds[my_folds[fold_name] != optimize_on][locker["target_name"]].values)

#         xvalid = rest_train[(my_folds[fold_name] == optimize_on).values]
#         yvalid = np.array(my_folds[my_folds[fold_name] == optimize_on][locker["target_name"]].values)

  
#         val_idx = None 

#         if _state in ["fold", "opt"]:
#             #  for seed we don't need to find val_idx
#             val_idx = my_folds[my_folds[fold_name] == optimize_on][locker["id_name"]].values.tolist()
        
#         xtrain= pd.DataFrame(xtrain, columns= ordered_list)[useful_features].values
#         xvalid= pd.DataFrame(xvalid, columns= ordered_list)[useful_features].values
#         return val_idx, xtrain, xvalid, ytrain, yvalid , useful_features    

# https://www.kaggle.com/competitions/amex-default-prediction/discussion/332575#1829172
import pathlib
class SaveModelCallback:
    def __init__(self,
                 models_folder: pathlib.Path,
                 fold_id: int,
                 min_score_to_save: float,
                 every_k: int,
                 order: int = 0):
        self.min_score_to_save: float = min_score_to_save
        self.every_k: int = every_k
        self.current_score = min_score_to_save
        self.order: int = order
        self.models_folder: pathlib.Path = models_folder
        self.fold_id: int = fold_id

    def __call__(self, env):
        iteration = env.iteration
        score = env.evaluation_result_list[3][2]
        if iteration % self.every_k == 0:
            print(f'iteration {iteration}, score={score:.05f}')
            if score > self.current_score:
                self.current_score = score
                for fname in self.models_folder.glob(f'fold_id_{self.fold_id}*'):
                    fname.unlink()
                print(f'High Score: iteration {iteration}, score={score:.05f}')
                joblib.dump(env.model, self.models_folder / f'fold_id_{self.fold_id}_{score:.05f}.pkl')


def save_model2(models_folder: pathlib.Path, fold_id: int, min_score_to_save: float = 0.78, every_k: int = 50):
    return SaveModelCallback(models_folder=models_folder, fold_id=fold_id, min_score_to_save=min_score_to_save, every_k=every_k)

def save_model1():
   def callback(env):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()      
        max_score = global_variables.max_score
        iteration = env.iteration
        score = env.evaluation_result_list[0][2]
        if iteration % 100 == 0:
            print('iteration {}, score= {:.05f}'.format(iteration,score))
        if score > max_score:
            max_score = score
            path = f"../models/models-{comp_name}/callback_logs/lgb_models_e_{global_variables.exp_no}_f_{global_variables.counter}_{global_variables._state}/"  #'Models/'
            # if path don't exists create one 
            # if exists throw error or delete it first 
            for fname in os.listdir(path):
                    if fname.startswith("fold_{}".format(global_variables.fold)):
                        os.remove(os.path.join(path, fname))
            print('High Score: iteration {}, score={:.05f}'.format(iteration, score))
            joblib.dump(env.model, f"../models/models-{comp_name}/callback_logs/lgb_models_e_{global_variables.exp_no}_f_{global_variables.counter}_{global_variables._state}/"+'{:.05f}.pkl'.format(score))

            global_variables.max_score = max_score
   callback.order = 0
   return callback

def mkdir_from_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
      print(f"Folder: {path} already exists, do you want to overwrite?")
      print("Enter Y/y for overwrite or any other button to terminate")
      print(": ",end="")
      a = input()
      if a.upper() == "Y":
        # overrite 
        print("Overwritting...")
      else:
        raise Exception("Folder already exists so process Terminated!!!")
    return path

if __name__ == "__main__":
    comp_name= 'amex3'
    useful_features=['B_1_last', 'B_11_last', 'B_12_last','pred_e_5_fold5']
    fold_name = "fold3"
    optimize_on= [2]
    _state = "seed"
    # opt: idx, train, valid, [test]
    # fold: idx, train, valid, test 
    # seed:   train, test 
    return_type = "numpy_array"
    # self.val_idx, self.xtrain, self.xvalid, self.ytrain, self.yvalid, self.ordered_list_train
    val_idx, xtrain, xvalid, ytrain, yvalid, my_list = bottleneck(comp_name,useful_features, fold_name,  optimize_on, _state, return_type,0)
    print(val_idx)
    print()
    print(xtrain)
    print()
    print(xvalid)
    print()
    print(ytrain)
    print()
    print(yvalid)
    print()
    print(my_list)