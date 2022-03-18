from optuna_search import OptunaOptimizer
from feature_generator import features 
from feature_picker import Picker 
import os 
import sys
import pickle
import pandas as pd

class Agent:
    def __init__(self,useful_features, model_name, comp_type,metrics_name,n_trials, prep_list, optimize_on):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.locker = self.load_pickle(f"../models_{comp_name}/locker.pkl")
        self.current_dict = self.load_pickle(f"../models_{comp_name}/current_dict.pkl")
        #----------------------------------------------------------
        self.useful_features = useful_features
        self.model_name = model_name 
        self.comp_type = comp_type 
        self.metrics_name = metrics_name 
        self.n_trials = n_trials 
        self.prep_list = prep_list
        self.optimize_on = optimize_on 
        print(self.locker)

    def save_pickle(self, path, to_dump):
        with open(path, "wb") as f:
            pickle.dump(to_dump, f)

    def load_pickle(self, path):
        with open(path, "rb") as f:
            o = pickle.load(f)
        return o
    
    def run(self):
        my_folds= pd.read_csv(f"../models_{self.locker['comp_name']}/my_folds.csv")
        opt = OptunaOptimizer( model_name= self.model_name,comp_type=self.comp_type,
                            metrics_name=self.metrics_name,n_trials=self.n_trials, prep_list= self.prep_list, optimize_on=self.optimize_on)
        study, log_table= opt.run( my_folds, self.useful_features)
        self.save_models(study,log_table)

    def get_exp_no(self):
        # exp_no, current_level, current_feature_no
        self.current_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/current_dict.pkl"
        )
        self.current_exp_no = int(self.current_dict["current_exp_no"])


    def save_models(self,study,log_table):
        Table = self.load_pickle(f"../models{a['comp_name']}/Table.pkl")

        # what unifies it 
        self.get_exp_no()
        # ExpNo- self.current_exp_no 
        self.current_exp_no += 1 
        self.Table.loc[Table.shape[0],:] = [self.current_exp_no, self.model_name, study.best_trial.value, study.best_trial.params, 
                                            self.useful_features, self.current_dict["current_level"], self.optimize_on, self.n_trials, 
                                            self.prep_list, self.metrics_name, log_table]


        #--------------- dump experiment no 
        self.current_dict['current_exp_no'] = self.current_exp_no
        self.current_dict = self.load_pickle(
            f"../models_{self.locker['comp_name']}/current_dict.pkl"
        )
        #---------------- dump table 
        self.save_pickle(f"../models{a['comp_name']}/Table.pkl", Table)

    def display(self,exp_list= [0]):
        """
        exp_no", "model_name", "bv", "bp", "features_list", "level_no", "fold_no", "no_iterations", "prep_list" "metrics_name" "exp_log"           
        exp_log: it will be a table
        """
        Table_Temp = self.load_pickle(f"../models{a['comp_name']}/Table.pkl")
        print(Table_Temp[Table_Temp.exp_no.isin(exp_list)])

if __name__ == "__main__":
    #==========================================================
    list_levels = ["1"]    #---------------> ["1","2"]
    list_features = ["0","1","2"]  #---------------> ["0","1","2"]
    list_feat_title = ["create_statistical_features"]   #---------------->["base", "create_statistical_features"]
    #---------------------------------------------------------
    p = Picker()
    useful_features = p.find_features(list_levels=list_levels, list_features=list_features, list_feat_title=list_feat_title)
    #==========================================================
    model_name = "lgr"   #--------> ["lgr","lir","xgbc","xgbr"]
    comp_type = "2class" #-------->["regression", "2class","multi_class", "multi_label"]
    metrics_name ="accuracy" #--------->["accuracy","f1","recall","precision", "auc", "logloss","auc_tf","mae","mse","rmse","msle","rmsle","r2"]
    n_trials = 5          #------------> no of times to run optuna
    prep_list = ["Sd"]        #------> ["SiMe", "SiMd", "SiMo", "Mi", "Ro", "Sd", "Lg"] <= _prep_list
    optimize_on= 0       # fold on which optimize
    #-----------------------------------------------------------

    e = Agent(useful_features=useful_features,model_name= model_name,comp_type=comp_type,metrics_name=metrics_name,n_trials=n_trials,prep_list= prep_list, optimize_on=optimize_on)
    print("="*40)
    print("Useful_features:", useful_features)
    
    
    
    e.run() 

    #-------------------------------------------------------------
    exp_list = [0] #----------------> [0,1,2]
    #e.display(exp_list)
    
