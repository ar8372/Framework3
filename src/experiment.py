from optuna_search import OptunaOptimizer
from feature_generator import features 
from feature_picker import Picker 


if __name__ == "__main__":
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
        a = pickle.load(f)
    #----------------------------------------------------------
    model_name = "lgr"   #--------> ["lgr","lir","xgbc","xgbr"]
    comp_type = "2class" #-------->["regression", "2class","multi_class", "multi_label"]
    metrics_name ="accuracy" #--------->["accuracy","f1","recall","precision", "auc", "logloss","auc_tf","mae","mse","rmse","msle","rmsle","r2"]
    n_trials = 5          #------------> no of times to run optuna
    prep_set = []        #------> ["SiMe", "SiMd", "SiMo", "Mi", "Ro", "Sd", "Lg"] 
    optimize_on= 0       # fold on which optimize
    #-----------------------------------------------------------
    ft = Picker()
    useful_features = 
    
    my_folds= pd.read_csv(f"../input_{comp_name}/train_folds.csv")
    opt = OptunaOptimizer()
    study= opt.run(self, my_folds, useful_features, prep_set= "--|--", optimize_on="--|--"):