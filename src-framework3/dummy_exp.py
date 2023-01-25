import os ,sys
import pandas as pd 
import numpy as np 
from utils import load_pickle, save_pickle 

class dummy:
    def __init__(self):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.comp_name = comp_name # put it here for the mkdir of lgb callback
        self.locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
        self.current_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl"
        )        
        self.Table = load_pickle(f"../configs/configs-{self.locker['comp_name']}/Table.pkl")
        self.Table = pd.DataFrame(self.Table)

        # update it with the lates values
        self.get_exp_no()

        print("=" * 30)
        print(f"Current Exp no: {self.current_exp_no}")
        print("=" * 30)

    def get_exp_no(self):
        # exp_no, current_level
        self.current_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl"
        )
        self.current_exp_no = int(self.current_dict["current_exp_no"])
    
    def get_row(self, pull_exp_no):
        if pull_exp_no == -1:
            return self.Table.loc[self.Table.shape[0]-1, :].copy()
        else:
            # since this table stores all the experiment so index no corresponds to exp_no
            return self.Table.loc[pull_exp_no, :].copy()
        

    def insert_row(self, changer, pull_exp_no):

        raw_row = self.get_row(pull_exp_no)
        print("Original row")
        print(raw_row)
        print() 
        for key,value in changer.items():
            if value != "--|--": 
                # to change it 
                raw_row[key] = value
        raw_row['exp_no'] = self.current_exp_no
        if pull_exp_no == -1:
            raw_row["notes"] = f"dummy_{self.Table.shape[0]-1}" 
        else:
            raw_row["notes"] = f"dummy_{pull_exp_no}" 

        print("Modified row")
        print(raw_row)        
        self.Table.loc[self.Table.shape[0], :] = raw_row.values 
        print()
        print(self.Table.tail(3))

        m = input("Do you want to insert this row!!, Type Y/y to proceed, else type any other key.\n: ")
        if m.lower() == "y":
            self._save_models()
            print("Updated!!")
        else:
            print("Aborted!!")

    def _save_models(self):
        self.current_exp_no += 1
        # --------------- dump experiment no
        self.current_dict["current_exp_no"] = self.current_exp_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl",
            self.current_dict,
        )

        #---------------- sanity check table 
        self.Table.exp_no = self.Table.exp_no.astype(int)
        self.Table.level_no = self.Table.level_no.astype(int)
        self.Table.no_iterations = self.Table.no_iterations.astype(int)
        self.Table.random_state = self.Table.random_state.astype(int)
        # ---------------- dump table
        save_pickle(f"../configs/configs-{self.locker['comp_name']}/Table.pkl", self.Table)

    def show_variables(self):
        print()
        for i, (k, v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>", v)
        print()

if __name__ == "__main__":
    d = dummy()

    changer= {
    "exp_no":  "--|--",     
    "model_name": "--|--",
    "bv": 100, # to keep it on top
    "bp": {'learning_rate': 0.010821262164314453, 'max_depth': 16, 'min_child_weight': 5, 'subsample': 0.4521783648128741, 'n_estimators': 500, 'objective': 'reg:squarederror', 'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': 'gpu_predictor'},
    
    "random_state": "--|--",
    "with_gpu": "--|--",
    "aug_type": "--|--",
    "_dataset": "--|--",
    "use_cutmix": "--|--",
    "callbacks_list": "--|--",
    "features_list": "--|--",
    "level_no": "--|--",
    "oof_fold_name": [],
    "opt_fold_name": "--|--",
    "no_iterations": "--|--",
    "prep_list": "--|--",
    "metrics_name": "--|--",
    # Below and some of the above things shold be empty when creating a new row
    "seed_mean": None,
    "seed_std": None,
    "fold_mean": [],
    "fold_std": [],
    "pblb_single_seed": None,
    "pblb_all_seed": None,
    "pblb_all_fold": [],
    "notes": "--|--",
    }
    pull_exp_no = 7
    d.insert_row(changer, pull_exp_no)