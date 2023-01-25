import numpy as np 
import pandas as pd 
import os 
import sys 
import gc 
from utils import * 
from metrics import * 
from collections import defaultdict

class Ensembler:
    def __init__(self):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.comp_name = comp_name 
        self.locker = load_pickle(f"../configs/configs-{self.comp_name}/locker.pkl")
        self.current_dict = load_pickle(f"../configs/configs-{self.comp_name}/current_dict.pkl")
        self.current_ens_no = self.current_dict['current_ens_no']
        print("="*40)
        print("Current ens no:", self.current_ens_no)
        print("="*40)
        self.cv_score = [] # of fold3, all, single
        self.pblb_score = [] # corresponding
        self.Table = load_pickle(f"../configs/configs-{self.comp_name}/Table.pkl")
        #self.my_folds = pd.read_parquet(f"../input/input-{self.comp_name}/my_folds.parquet")
        self.id_folds_target = pd.read_parquet(f"../input/input-{self.comp_name}/id_folds_target.parquet")
        self.target =  self.id_folds_target[self.locker['target_name']]   #self.my_folds[self.locker["target_name"]].values
        self.id = self.id_folds_target[self.locker["id_name"]].values 
        self.sample = pd.read_parquet(f"../input/input-{self.comp_name}/sample.parquet")

        #del self.my_folds 
        gc.collect()

    def access_predictions(self, submission_list):
        self.submission_list = submission_list 
        self.train_list = []
        self.test_list = []

        for i,j in submission_list: 
            # access it  
            
            if j.startswith("fold"):
                # it is a prediction 
                self.train_list.append( load_pickle(f"../configs/configs-{self.comp_name}/oof_preds/oof_pred_e_{i}_{j}.pkl").reshape(-1,))
                self.test_list.append( load_pickle(f"../configs/configs-{self.comp_name}/test_preds/test_pred_e_{i}_{j}.pkl").reshape(-1,))
                # exp_no and index match for table so works
                names = self.Table.loc[i, "oof_fold_name"]
                ind = names.index(j)
                print()
                print("exp_no",i)
                try:
                    self.pblb_score.append(self.Table.loc[i, 'pblb_all_fold'][ind])
                    print(f"pblb: {self.Table.loc[i, 'pblb_all_fold'][ind]}")
                except:
                    print("pblb_score not found")
                try:
                    self.cv_score.append(self.Table.loc[i, 'fold_mean'][ind])
                    print(f"cv: {self.Table.loc[i, 'fold_mean'][ind]}")
                except:
                    print("cv_score is not found")
            elif j in ["all", "single"]:
                raise Exception("Need to work on it")
            elif j.startswith("feat"):
                # these are features 
                raise Exception("Need to work on  it")
                self.column_names.append(f"feat_l_1_e_{i}_feat")
            elif j.startswith("../working"):
                # picked some public kernel 
                assert j.endswith(".csv") 
                # must be a csv 
                # [[0.7977, 0.799], "../working/.."]
                assert len(i) == 2
                self.cv_score.append(i[0])
                self.pblb_score.append(i[1])
                print(f"cv: {i[0]} , pblb: {i[1]}")
                # no training 
                self.train_list.append( np.ones((self.id_folds_target.shape[0], )) )
                self.test_list.append( pd.read_csv(j)[self.locker['target_name']])

        # got train and test 
        #self.train_list = np.array()
        # draw correlation plot
        col_names = [str(i[0]) for i in submission_list]
        train_corr = (pd.DataFrame(np.array(self.train_list + [self.id_folds_target[self.locker['target_name']]]).T, columns= col_names + [self.locker['target_name']])).corr()
        test_corr = (pd.DataFrame(np.array(self.test_list).T, columns= col_names)).corr()
        print("-"*40)
        print("Train corr")
        print(train_corr)
        print("-"*40)
        print("Test corr")
        print(test_corr)
        print("-"*40)


    def save_process(self, save_dict):
        # save submissions_list a
        # technique # no need to save seperately contained in submissions_list name
        # update current_dict 
        save_json(f"../configs/configs-{self.comp_name}/ensemble_logs/ens_{self.current_ens_no}.json", save_dict)

        self.current_ens_no += 1 
        self.current_dict['current_ens_no'] = self.current_ens_no 
        save_pickle(f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl", self.current_dict)

    def combine(self, technique):
        self.technique = technique
        if technique == "power_averaging_basic":
            # self.train_list is a list where each element is a list of submissions , [0] picking the first since not multi-class problem
            train_pred =  [ np.sum([0.2 * i for i in f], axis=0) for f in [self.train_list]][0]
            test_pred =  [ np.sum([0.2 * i for i in f], axis=0) for f in [self.test_list]][0]
        elif technique == "weight_averaging_by_pblb":
            train_pred =  [ np.sum([i*j for i,j in zip(f,self.pblb_score) ], axis=0) for f in [self.train_list]][0]
            test_pred =  [ np.sum([i*j for i,j in zip(f,self.pblb_score)], axis=0) for f in [self.test_list]][0]
        elif technique == "weight_average_by_cv":
            train_pred =  [ np.sum([i*j for i,j in zip(f,self.cv_score) ], axis=0) for f in [self.train_list]][0]
            test_pred =  [ np.sum([i*j for i,j in zip(f,self.cv_score)], axis=0) for f in [self.test_list]][0]
            print(test_pred)
            train_pred = np.array(train_pred)/sum(self.cv_score)
            test_pred = np.array(test_pred)/sum(self.cv_score)
        elif technique == "median":
            train_pred =  [ np.median(f) for f in [self.train_list]][0]
            test_pred =  [ np.median(f) for f in [self.test_list]][0]



        print(train_pred.shape, test_pred.shape)



        # save oof predictions
        #score = amex_metric(self.target, train_pred)
        score = getaroom_metrics(self.target, train_pred)
        print("train score: ",score)
        
        save_dict = defaultdict()
        save_dict["submission_list"] = self.submission_list 
        save_dict["cv_score"] = score 
        save_dict["technique"] = technique 


        self.sample[self.locker['target_name']] = test_pred
        print() 
        print(self.sample.head(3))

        # Now working withing same comp because folds hold for same datset and not amex, amex2
        input("Want to proceed!")
        self.sample.to_parquet(f"../working/{self.comp_name}_ens_{self.current_ens_no}.parquet")
        self.save_process(save_dict)






if __name__ == "__main__":
    # For now works only for fold prediction
    # submission_list = [
    #     [215, "fold5"],
    #     [1, "fold5"],
    #     [265, "fold5"],
    #     [49, "fold5"],
    #     [138, "fold5"]
    # ]    
    submission_list = [
        # 1,8,20,18, 47, 14, 54, 43, 56, 42, 35, 3, 34, 6, 41
        # [294,'fold5'],
        # [297,'fold5'],
        # [273,'fold5'],
        # [254,'fold5'],
        [[0.7977, 0.799], "../working/rr_mean_submission.csv"],
        [[0.7977, 0.799], "../working/rr_submission.csv"],
        [[0.7977, 0.799], "../working/rr_submission1.csv"]


        #[1, "fold5"],
        #[8, "fold5"],
        #[20, "fold5"],
        # [18, "fold5"],
        # [47, "fold5"],
        # [14, "fold5"],
        # [54, "fold5"],
        #[61, "fold5"],
        # [43, "fold5"],
        # [56, "fold5"],
        # [42, "fold5"],
        # [35, "fold5"],
        # [3, "fold5"],
        # [34, "fold5"],
        # [6, "fold5"],
        # [41, "fold5"],

        # [8, "fold5"],
        # [20, "fold5"],
        # [18, "fold5"],
        # [35, "fold5"],
        # [29, "fold5"],
        # [1, "fold5"],
        # [45, "fold5"],
        # [54, "fold5"],
        # [56, "fold5"],
        # [14, "fold5"],
        # [43, "fold5"],
        # [34, "fold5"],
        # [47, "fold5"],
        # [27, "fold5"],
        # [3, "fold5"],
        # [55, "fold5"],
        # [6, "fold5"],
        # [42, "fold5"],
        # [41, "fold5"],
        # [7, "fold5"],
        # [2, "fold5"],
        # [32, "fold5"],
        # [15, "fold5"],
        # [44, "fold5"],
        # [80, "fold5"],
        #[43, "fold5"],
        #[15, "fold5"],
        # [1, "fold5"],
        # [8, "fold5"],
        # [14, "fold5"],
        # [15, "fold5"],
        # [17, "fold5"],
        # [45, "fold5"],
        #[79, "fold5"],
        #[98, 'fold5'],
        #[111, 'fold5'],
        #[1, "fold5"],
        #[44, "fold5"],
        #[15, "fold5"],
        #[[0.798, 0.799], "../working/mean_submission.csv"],
        #[[0.7977, 0.799], "../working/test_lgbm_baseline_5fold_seed_blend.csv"],
        #[112, "fold5"],
    ]   
    # "median", "mean" , "weighted_mean", "best", "rank", "weigh_by_cv", "weight_averaging_by_pblb", "weigh_by_cv_pblb_jump", "power_averaging_basic"
    technique = "weight_average_by_cv" #"weight_average_by_cv" #"power_averaging_basic" # "mean" # "weighted_mean" , "best" , "mean"

    e = Ensembler()
    e.access_predictions(submission_list)
    e.combine(technique)

"""
0.795 215 fold5, 1 fold5, 265 fold5 , 
0.796 49 fold5, 138 fold5

"""
# some ideas
# larger the jump b/w cv and pblb score better is the model pblb > cv not cv < pblb