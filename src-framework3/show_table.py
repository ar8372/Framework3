import os
import sys
from utils import *
import pandas as pd

#pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def update_table_function(
    Table, exp_no, pblb_single_seed=None, pblb_all_seed=None, pblb_all_fold=None
):
    # pblb_single_seed	pblb_all_seed	pblb_all_fold
    Table.loc[exp_no, "pblb_single_seed"] = pblb_single_seed
    Table.loc[exp_no, "pblb_all_seed"] = pblb_all_seed
    Table.loc[exp_no, "pblb_all_fold"] = pblb_all_fold
    return Table


def show_table(exp_list,col, is_sorted, which_table= ["base_table"]):
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    bv_t = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")

    # load
    for t in which_table:
        print()
        print("*--"*40)
        if t.split("_")[0] == "base":
            Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")
            if exp_list != []:
                Table= Table[Table.exp_no.isin(exp_list)]
            if is_sorted:
                Table = Table.sort_values(by=["bv"], axis=0, ascending=False)
                #Table = Table[Table.fold_mean is list]
                #Table = Table.head(5)
                #print(Table)
                Table = Table.sort_values(by= ["fold_mean"], axis=0, ascending=False)
            # to compress 
            if col == []:
                Table.rename(columns={'pblb_single_seed': 'single', 'pblb_all_seed': 'all', 'pblb_all_fold': 'fold', "no_iterations": "#iter"}, inplace=True)
                print(Table.set_index('exp_no'))
            else:
                if len(col) == 1:
                    # asking for single value 
                    print(Table[col[0]].values[0])
                else:
                    Table= Table[Table.bv > 0.75][col]
                    #Table.rename(columns={'pblb_single_seed': 'single', 'pblb_all_seed': 'all', 'pblb_all_fold': 'fold', "no_iterations": "#iter"}, inplace=True)
                    print(Table)
            #print(self.Table[])
        else:
            submissions = ["seed_mean", "fold_mean","pblb_single_seed","pblb_all_seed","pblb_all_fold"]
            Table = pd.read_csv(f"../configs/configs-{comp_name}/auto_exp_tables/auto_exp_table_{t.split('_')[0]}.csv")
            Table = pd.merge(Table, bv_t[["exp_no","bv"]+ submissions], on="exp_no",how="left")
            if exp_list != []:
                Table= Table[Table.exp_no.isin(exp_list)]
            if False: #is_sorted:
                Table = Table.sort_values(by=["bv"], axis=0, ascending=False)
            #print(tabulate(Table[Table.bv > 0.75].set_index('exp_no')))
            #print(Table[Table.bv > 0.75].set_index('exp_no').to_markdown())
            Table.rename(columns={'pblb_single_seed': 'single',"seed_mean":"s_mean", "fold_mean":"f_mean", "feature_names":"f_names", 'pblb_all_seed': 'all', 'pblb_all_fold': 'fold', "no_iterations": "#iter","optimize_on":"opt_on"}, inplace=True)
            print(Table[Table.bv > 0.75].set_index('exp_no'))
            #print(self.Table[])
def change_name(old_name, new_name):
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    # load
    Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")   
    Table = Table.rename(columns = {old_name : new_name}) 
    print(Table.head(2))
    save_pickle(f"../configs/configs-{comp_name}/Table.pkl", Table)

if __name__ == "__main__":
    col = [
        "exp_no",
        "model_name",
        "bv",
        "bp",
        "random_state",
        "with_gpu",
        "aug_type",
        "_dataset",
        "use_cutmix",
        "features_list",
        "level_no",          
        "oof_fold_name",
        "opt_fold_name",
        "fold_no",
        "no_iterations",
        "prep_list",
        #    'metrics_name',
        #    'seed_mean',
        #    'seed_std',
        #    'fold_mean',
        #    'fold_std',
        #    'pblb_single_seed',
        #    'pblb_all_seed',
        #    'pblb_all_fold',
        #    'notes',
    ]
    submissions = ["seed_mean", "seed_std","fold_mean","fold_std","pblb_single_seed","pblb_all_seed","pblb_all_fold"]
    general = ["prep_list","opt_fold_name","oof_fold_name", "fold_no","no_iterations"]
    base = ['exp_no',"bv"]

    explore_range = ["pblb_all_fold", "pblb_single_seed","pblb_all_seed"]
    pblb_cols = ["pblb_all_fold","pblb_single_seed","pblb_all_seed"]
    cv_cols = ["oof_fold_name", "fold_mean","fold_std", "seed_mean", "seed_std"]
    opt_cols = ["bv","no_iterations","prep_list","opt_fold_name","fold_no", "with_gpu"] #, "bp", "random_state", "metrics_name", "feaures_list" # things to reproduce prediction from an experiment


    exp_list = [17] #[435, 416]
    exp_list = [240]
    exp_list = []
    which_table = [
        "base_table",
        #"xgbr_table",
        #"xgb_table",
        #"cbc_table",
        #"lgbmc_table",
        #"lgb_table",
    ]

    # don't change col
    col = ["exp_no","model_name","no_iterations","bv"] + explore_range + ["fold_mean", "oof_fold_name", "opt_fold_name", "fold_no", "seed_mean", "fold_std", "seed_std"]
    #col = ['bp']
    #col = ['features_list']
    #col = []
    is_sorted = True
    show_table(exp_list, col, is_sorted, which_table)
    # old_name = 'callbacks_listfeatures_list'
    # new_name = 'features_list'
    # change_name(old_name, new_name)



