import os
import sys
from utils import *
import pandas as pd
import ast 

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def update_table_function(
    Table, exp_no, pblb_single_seed=None, pblb_all_seed=None, pblb_all_fold=None
):
    # pblb_single_seed	pblb_all_seed	pblb_all_fold
    Table.loc[exp_no, "pblb_single_seed"] = pblb_single_seed
    Table.loc[exp_no, "pblb_all_seed"] = pblb_all_seed
    Table.loc[exp_no, "pblb_all_fold"] = [[i for i in pblb_all_fold]]
    return Table

def update_score(
    exp_no=-1, pblb_single_seed=None, pblb_all_seed=None, pblb_all_fold=None
):
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    # load
    Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")

    if exp_no == -1:
        # pick the last one
        exp_no = Table.iloc[-1, :].exp_no

    # before
    print(
        Table[Table.exp_no == exp_no][
            ["exp_no", "pblb_single_seed", "pblb_all_seed", "pblb_all_fold"]
        ]
    )
    print()

    # -> Sanity check:
    if pblb_single_seed != None:
        assert Table.loc[exp_no, "pblb_single_seed"] == None
    if pblb_all_seed != None:
        assert Table.loc[exp_no, "pblb_all_seed"] == None
    if pblb_all_fold != None:
        assert Table.loc[exp_no, "pblb_all_fold"] == [] # None

    # update
    Table = update_table_function(
        Table, exp_no, pblb_single_seed, pblb_all_seed, pblb_all_fold
    )

    # after
    print(
        Table[Table.exp_no == exp_no][
            ["exp_no", "pblb_single_seed", "pblb_all_seed", "pblb_all_fold"]
        ]
    )

    print("=" * 40)
    print("Type Y/y to update, N/n to reject")
    text = input()
    if text.upper() == "Y":
        # Dump back
        save_pickle(f"../configs/configs-{comp_name}/Table.pkl", Table)
        print("Updated!")
    else:
        print("Cancelled Update!")

def change_table(exp_no, feature_name, value):
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    # load
    Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")
 
    old_value= Table.loc[exp_no, feature_name] 
    print(f"Do you want to replace \n{old_value} \n\nwith: \n{value} \nfor:")
    print(f"Exp no: {exp_no} and col: {feature_name}")
    print("=" * 40)
    print("Type Y/y to update, N/n to reject:")
    text = input()
    if text.upper() == "Y":
        # Dump back
        # old : Table.loc[exp_no, feature_name]  = value
        Table = get_table(Table, exp_no, feature_name, value)
        save_pickle(f"../configs/configs-{comp_name}/Table.pkl", Table)
        print("Updated!")
    else:
        print("Cancelled Update!")

def get_table(Table, exp_no, feature_name, value):
    raw_row = Table.loc[exp_no, :].copy()
    raw_row[feature_name] = value
    Table.loc[exp_no, :] = raw_row.values
    return Table    

def change_table_custom():
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    # load
    Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")
    Table.exp_no = Table.exp_no.astype(int)
    Table.level_no = Table.level_no.astype(int)
    Table.no_iterations = Table.no_iterations.astype(int)
    Table.random_state = Table.random_state.astype(int)
    print(Table.head(2))
    print("=" * 40)
    print("Type Y/y to update, N/n to reject:")
    text = input()
    if text.upper() == "Y":
        # Dump back
        save_pickle(f"../configs/configs-{comp_name}/Table.pkl", Table)
        print("Updated!")
    else:
        print("Cancelled Update!")

if __name__ == "__main__":
    # exp_no = -1  # keep it as default
    exp_no = 70
    pblb_single_seed = None
    pblb_all_seed = None
    pblb_all_fold = [83.00393] #83.22706
    update_score(exp_no, pblb_single_seed, pblb_all_seed, pblb_all_fold)

    # exp_no = 269 
    # feature_name = "no_iterations"
    # value = 30
    # change_table(exp_no, feature_name, value)
    # exp_no = 18 
    # feature_name = "bp"
    # value = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'dart', 'learning_rate': 0.010230005490999927, 'seed': 241, 'num_leaves': 105, 'feature_fraction': 0.18956831457766554, 'bagging_freq': 12, 'bagging_fraction': 0.5115517394581794, 'n_jobs': -1, 'lambda_l2': 1, 'min_data_in_leaf': 35}
    # change_table(exp_no, feature_name, value)
    #change_table_custom()


    # {'n_estimators': 500, 'learning_rate': 0.27981914604335584, 'max_depth': 7, 'loss': 'squared_error', 'criterion': 'mse', 'max_features': 'auto', 'min_sample_split': 0.11249062083311395, 'subsample': 0.95}
    # exp_no = 17
    # feature_name = "bp"
    # value = {'n_estimators': 500, 'learning_rate': 0.27981914604335584, 'max_depth': 7, 'loss': 'squared_error', 'criterion': 'mse', 'max_features': 'auto', 'min_samples_split': 0.11249062083311395, 'subsample': 0.95}
    # change_table(exp_no, feature_name, value)
    