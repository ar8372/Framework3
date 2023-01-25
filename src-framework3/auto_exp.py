import random
import numpy as np
import pandas as pd
import time
import os
import gc

# import quantumrandom
import ast  # for literal
import tensorflow as tf
import pickle
from datetime import datetime
from utils import *
from feature_picker import *
from experiment import *
from settings import *


def make_selection_prep(my_list, shuffle=True, random_state=24):
    # The first thing we do is set random state to make it reproducible:
    fix_random(random_state)

    # finds all possible SELECTION possible
    # col_list = ["f_00","f_01","f_02","f_05"]
    new_list = []
    for i in range(2 ** len(my_list)):
        val = int(bin(i)[2:])
        temp_list = []
        counter = 1
        while val != 0:
            if val % 10 == 1:
                # take it
                temp_list.append(my_list[-counter])
            val = val // 10
            counter += 1
        new_list.append(temp_list)
    ch = random.choice(new_list)
    if shuffle == True:
        random.shuffle(ch)
    return ch  # for list of list we must append each other but that is a manual task

def make_selection(my_list, shuffle=True, random_state=24):
    print("start-->")
    # The first thing we do is set random state to make it reproducible:
    fix_random(random_state)

    # finds all possible SELECTION possible
    # col_list = ["f_00","f_01","f_02","f_05"]
    i = random.randint(0,2 ** len(my_list))
    val = int(bin(i)[2:])
    temp_list = []
    counter = 1
    while val != 0:
        if val % 10 == 1:
            # take it
            temp_list.append(my_list[-counter])
        val = val // 10
        counter += 1
    ch = temp_list
    if shuffle == True:
        random.shuffle(ch)
    print("done-->")
    return ch  # for list of list we must append each other but that is a manual task


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


def sanity_check(table, *rest):
    global counter 
    counter += 1
    for index, row in table.iterrows():
        row = row.values
        # row
        assert len(rest) == len(row[1:])
        same = True
        for i,(val1, val2) in enumerate(zip(rest, row[1:])):
            # i=5 optimize_on [0,2] special case
            #print(i, val1, val2) #best way to check if allow duplicates
            if not isinstance(val1, list):
                if val1 != val2:
                    # "fold5" "fold10"
                    same = False
                    break
            if isinstance(val1, list) and i != 5:
                # list of lenght 2
                # val2 is a list inform of string '[10,0]' but not '[10]'
                val2 = ast.literal_eval(val2)
                # val2 = val2[1:-1].split(',')
                # it is a list
                if val1[1] == 1:
                    # order matters
                    if val1[0] != val2:
                        same = False
                        break
                else:
                    # order don't matter
                    set1 = set(val1[0])
                    set2 = set(val2)
                    if set1 != set2:
                        same = False
                        break
            # elif i == 5:
            #     # if this elif is not written then it will consider all different optimize on duplicate
            #     # it is optimize on
            #     val2 = ast.literal_eval(val2) 
            #     if val1 != val2:
            #         same = False 
            #         break 


        if same == True:
            # duplicate
            print(row, "found in table")
            return True
    return False




def auto_select():
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    if comp_name == "amex":
        amex = amex_settings()
    elif comp_name == "amex2":
        amex = amex2_settings() 
    elif comp_name == "amex3":
        amex = amex3_settings()
    elif comp_name == "amex4":
        amex = amex4_settings()
    elif comp_name == "getaroom":
        amex = getaroom_settings()
    else:
        raise Exception(f"comp name {comp_name} not valid")
    # =============================================================
    #                    Features                                #
    # ==============================================================
    # seed it Fix random no for this auto_exp
    random_no = (
        true_random(100)
    )  # never gives same random value quite random done to bring variabiligy
    # random_no = 111
    # ============================================================
    # each set must have a name in features for better visualisation
    # auto_features = ['ragnar']

    # clean_features = amex.feature_dict2['ragnar']

    """
    auto_features = []
    clean_features = []
    v = list(amex.feature_dict2.keys())
    auto_features = make_selection(v, shuffle=False, random_state=random_no) #amex.feature_keys
    #auto_features2 = random.choice([[random.choice(v)], []])
    # f_base should always be there
    # # # deal with empty list
    while auto_features == []:
        print("features empty!!")
        #============================================================
        # seed it
        random_no = (true_random(100))
        #random_no = 111
        #============================================================
        auto_features = make_selection(v, shuffle=False, random_state = random_no)
    print()
    print(auto_features)
    print()
    # #------------------------------------------------

    # for list of list we must append each other but that is a manual task
    #auto_features = ['ragnar', 'date_filter', 'cat_last_interact']
    auto_features = ['ragnar' ]#, 'cat_last_interact']
    """
    #auto_features.append(random.choice( list(amex.feature_dict2.keys()) ))

    auto_features = random.choices( list(amex.feature_dict.keys()))

    clean_features = []
    for f in auto_features:
        clean_features += amex.feature_dict[f]
    #clean_features += amex.filter_feature[auto_features[0]]
    #clean_features += amex.feature_dict2[auto_features[1]]

    # auto_features2 = ['filter4', 'filter5', 'filter6']
    # for f in auto_features2:
    #     clean_features += amex.filter_feature[f]
    #     #clean_features += amex.feature_dict2[f]
    # auto_features += auto_features2 

    

    auto_features = list(set(auto_features))
    
    clean_features = list(set(clean_features))

#-------------------------------------------------
    auto_features2 = random.choices( list(amex.filtered_features.keys()))

    clean_features2 = []
    for f in auto_features2:
        clean_features2 += amex.filtered_features[f]
    #clean_features += amex.filter_feature[auto_features[0]]
    #clean_features += amex.feature_dict2[auto_features[1]]

    # auto_features2 = ['filter4', 'filter5', 'filter6']
    # for f in auto_features2:
    #     clean_features += amex.filter_feature[f]
    #     #clean_features += amex.feature_dict2[f]
    # auto_features += auto_features2 

    

    auto_features2 = list(set(auto_features2))
    
    clean_features2 = list(set(clean_features2))


    auto_features = list(set(auto_features + auto_features2))
    
    clean_features = list(set(clean_features + clean_features2))

    # if len(clean_features) > 2000:
    #     # cut it 
    #     fix_random(random_no)
    #     clean_features = random.choices(clean_features, k=2000)
    #     auto_features.append('trim')
    # print(len(clean_features),"no of features")

    # ------------------------------------------------------
    print("seed:", random_no)
    print()
    # =============================================================
    #                    prep_list                               #
    # ===================================-===========================
    
    #auto_prep = make_selection_prep(amex.prep_list, shuffle=True, random_state=random_no)
    auto_prep = ["Mi", "Ro", "Sd", "Lg"]
    fix_random(random_no)
    auto_prep = random.choice(auto_prep)
    auto_prep= random.choice([[auto_prep], [], []]) # move prep towards empty since gives better performance
    #auto_prep = []
    
    print("Preprocessing:")
    print(auto_prep)
    print()
    # =============================================================
    #                    optimize_on                             #
    # ==============================================================
    # The first thing we do is set random state to make it reproducible:
    fix_random(random_no)

    fold_name = random.choice([i for i in ['fold3', 'fold5']]) #amex.fold_list]) #  fold20 too much folds
    # fold5: then train on 80% data, fold3: then train on 66% data
    fold_name = "fold5"
    #fold_name = random.choice(["fold10",  fold_name]) # move towards "fold10"
    #fold_name = random.choice(["fold3", "fold5"]) #random.choice([i for i in amex.fold_list])
    #fold_name = "fold10"
    # fold_name : "fold3"

    fold_nos = [i for i in range(int(fold_name.split("d")[1]))]
    # fold_nos = [0,1,2]
    # for now let's pick only one [1]
    fix_random(random_no)
    optimize_on = [random.choice(fold_nos)]

    # [0,2]
    # optimize_on = make_selection(fold_nos, shuffle=False, random_state=random_no)
    # while optimize_on == [] or len(optimize_on) == len(fold_nos): # don't optimize on all folds or empty
    #     optimize_on = make_selection(fold_nos, shuffle=False, random_state=random_no)

    print("optimizing on fold name:")
    print(fold_name)
    print()
    print("optimizing on fold:")
    print(optimize_on)
    print()
    # # The first thing we do is set random state to make it reproducible:
    # fix_random(random_no)
    # optimize_on = random.choice([i for i in range(locker["no_folds"])])
    # print("optimizing on fold:")
    # print(optimize_on)
    # print()
    # ==============================================================
    print(auto_features, auto_prep, fold_name, optimize_on,"Is it duplicate")
    return clean_features,auto_features, auto_prep, fold_name, optimize_on
    # ================================================

def custom_select(exp_no, model_name):
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    # ================================================
    # Selecting Features
    # ================================================

    #amex = amex_settings()
    if comp_name == "amex":
        amex = amex_settings()
    elif comp_name == "amex2":
        amex = amex2_settings() 
    elif comp_name == "amex3":
        amex = amex3_settings()
    else:
        raise Exception(f"comp name {comp_name}not valid")
    # =============================================================
    #                    Features                                #
    # ==============================================================
    # no need to seed 
    try:
        #input("Work on this part model name not defined")
        #model_name = input("Model Name: ")
        #print(f"../configs/configs-{comp_name}/auto_exp_tables/auto_exp_table_{model_name}.csv")
        auto_exp_table = pd.read_csv(f"../configs/configs-{comp_name}/auto_exp_tables/auto_exp_table_{model_name}.csv")  
        assert auto_exp_table.model_name.values[0] == model_name
    except:
        raise Exception("There must be an auto_exp_table else we can't custom train!")
    
    # model_name	feature_names	prep_list	fold_name	optimize_on	with_gpu
    #.feature_names[exp_no] # works when index and exp no same so will work only for Table since it captures all exp
    feature_names = ast.literal_eval(auto_exp_table[auto_exp_table.exp_no == exp_no].feature_names.values[0])
    prep_list = ast.literal_eval(auto_exp_table[auto_exp_table.exp_no == exp_no].prep_list.values[0])
    fold_name = auto_exp_table[auto_exp_table.exp_no == exp_no].fold_name.values[0]
    optimize_on = ast.literal_eval(auto_exp_table[auto_exp_table.exp_no == exp_no].optimize_on.values[0])
    print("Retrieved:=>")
    print(feature_names, prep_list, fold_name, optimize_on)
    #======================================================================
    #amex = amex_settings()
    clean_features = []
    for f in feature_names:
        clean_features += amex.feature_dict[f]
    auto_prep = prep_list 
    return  clean_features, feature_names, auto_prep, fold_name, optimize_on

def RUN_EXP(exp_no="--|--"):
    global no_exp, repeat, counter
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    # ================================================
    # Selecting Features
    # ================================================
    locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
    current_dict = load_pickle(
        f"../configs/configs-{locker['comp_name']}/current_dict.pkl"
    )
    e_no = current_dict["current_exp_no"]
    print("Current Exp No :", e_no)
    print()

    # ================================================
    # Running
    # ================================================
    #model_name = "lgb" #"xgb"
    model_name = random.choice(["xgbr", "cbr", "gbmr", "rfr"])
    model_name = "k1"
    comp_type = "regression"
    metrics_name = "getaroom_metrics"
    n_trials = 5 # we change this when there is great change in parameter set like , optimized on different parameter range
    with_gpu = True
    aug_type = "aug2"
    _dataset = "DigitRecognizerDataset"
    use_cutmix = False
    if exp_no != "--|--":
        note = f"repeat_exp_{exp_no}_with_{n_trials}_trials"
    else:
        note = f"Leap_"
    # ================================================
    # ================================================

    # ================================================
    # Get Settings
    # ================================================
    if exp_no != "--|--":
        # it's custom
        clean_features, auto_features, auto_prep, fold_name, optimize_on = custom_select(exp_no, model_name)
        print("Old Exp No :", exp_no)
    else:
        clean_features, auto_features, auto_prep, fold_name, optimize_on = auto_select()


    # e = Agent(
    #     useful_features=clean_features,
    #     model_name=model_name,
    #     comp_type=comp_type,
    #     metrics_name=metrics_name,
    #     n_trials=n_trials,
    #     prep_list=auto_prep,
    #     fold_name = fold_name,
    #     optimize_on=optimize_on,
    #     with_gpu=with_gpu,
    #     aug_type=aug_type,
    #     _dataset=_dataset,
    #     use_cutmix=use_cutmix,
    #     note=note,
    # )

    # Sanity check have we done the same experiment before:
    # What things to check:
    # 1> clean_features [order don't matter]
    # 2> model_name
    # 3> prep_list [order matters]
    # 4> optimize_on
    # each set must have a name in features for better visualisation
    # Table already created: Table = pd.DataFrame(columns=['exp_no','model_name','feature_names','prep_list',"optimize_on"])
    try:
        auto_exp_table = pd.read_csv(
            f"../configs/configs-{comp_name}/auto_exp_tables/auto_exp_table_{model_name}.csv"
        )
    except:
        print("Creating Table:")
        auto_exp_table = pd.DataFrame(
            columns=[
                "exp_no",
                "n_trials",
                "model_name",
                "feature_names",
                "prep_list",
                "fold_name",
                "optimize_on",
                "with_gpu",
            ]
        )
        auto_exp_table["exp_no"] = auto_exp_table["exp_no"].astype(int)
        #auto_exp_table["optimize_on"] = auto_exp_table["optimize_on"].astype(int)

    if not sanity_check(
        auto_exp_table,
        n_trials,
        model_name,
        [auto_features, 0],
        [auto_prep, 1],
        fold_name,
        optimize_on,
        with_gpu,
    ):
        # true means
        auto_exp_table.loc[auto_exp_table.shape[0], :] = [
            int(e_no),
            n_trials,
            model_name,
            auto_features,
            auto_prep,
            fold_name,
            optimize_on,
            with_gpu,
        ]
        auto_exp_table["exp_no"] = auto_exp_table["exp_no"].astype(int)
        auto_exp_table["n_trials"] = auto_exp_table["n_trials"].astype(int)
        #auto_exp_table["optimize_on"] = auto_exp_table["optimize_on"].astype(int)
        print()
        print(auto_exp_table)

        e = Agent(
            useful_features=clean_features,
            model_name=model_name,
            comp_type=comp_type,
            metrics_name=metrics_name,
            n_trials=n_trials,
            prep_list=auto_prep,
            fold_name = fold_name,
            optimize_on=optimize_on,
            with_gpu=with_gpu,
            aug_type=aug_type,
            _dataset=_dataset,
            use_cutmix=use_cutmix,
            note=note,
        )

        print("=" * 40)
        print("Useful_features:", clean_features)
        # critical part
        e.run()
        counter = 0 # resetting
        auto_exp_table.to_csv(
            f"../configs/configs-{comp_name}/auto_exp_tables/auto_exp_table_{model_name}.csv", index=False
        )
        del e
        gc.collect()

        # Make prediction also 
        os.system(f"python predict.py")

        return True
    else:
        print("Duplicate SET found!!!")
        if counter > 100:
            raise Exception("Too many times duplicte found ")
    no_exp += 1
    #del e # already delted
    gc.collect()
    return False

def Timer(rem_time, expected_time, exp_type, repeat):
    """
    rem_time = 12 * 60 * 60  # Run it for 3 hours
    expected_time = 1 * 60 * 60

    exp_type = "custom" # "auto"
    repeat = [261, 215]
    """
    current = 0
    no_exp = len(repeat)
    time_list = []
    while (exp_type == "auto" and rem_time >= expected_time) or (exp_type == "custom" and current < no_exp):
            print("%" * 40)
            start_time = time.time()
            # ============================================
            if exp_type == "custom":
                val = repeat[current]
            else:
                val = "--|--"
            #=============================================
            if RUN_EXP(val):
                # ============================================
                end_time = time.time()
                rem_time -= end_time - start_time
                time_list.append(end_time - start_time)
                expected_time = np.max(time_list)  # taking upper bound
                print(f"Experiment Done in {end_time- start_time} seconds")
                print("Rem Time:", rem_time)
                print("Expected Time:", expected_time)
                for i in range(5):
                    print()
            else:
                print("=" * 40)
                print("Sanity Check Failed!, Trying again")
                print("=" * 40)
            current += 1
            gc.collect()
            # break

        # #================================================
        # # Saving
        # #================================================
        # # # version data
        # from datetime import datetime
        # version_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        # version_name += "_fresh"
        # print(f"Versioning at {version_name}")

        # # # configs
        # os.system(f"kaggle datasets version -m {version_name} -p /kaggle/configs/configs-{comp_name}/")

        # # # models
        # # #!kaggle datasets version -m {version_name} -p /kaggle/models/models-{comp_name}/ -r zip -q

        # # # src
        # # #!kaggle datasets version -m {version_name} -p /kaggle/src-{framework_name}/

no_exp = 0 
passed = 0

counter = 0 

if __name__ == "__main__":
    rem_time = 10 * 60 * 60  # Run it for 3 hours
    expected_time =  0.0001 * 60 * 60 # expected one hour to finish one

    exp_type ="auto" # "auto" # "auto" "custom"
    repeat = [284, 277, 278, 279, 282]

    Timer(rem_time, expected_time, exp_type, repeat)

    
# lgbmc f_base, f_max, f_min, f_avg, f_last [] fold20 optimize_on 15 with_gpu = True