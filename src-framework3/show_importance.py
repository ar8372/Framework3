from utils import *
import os  
pd.set_option("display.max_rows", None)
import matplotlib.pyplot as plt 
import seaborn as sns


class Importance:
    def __init__(self, exp_no):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.comp_name = comp_name 
        self.feature_importance_table = None 
        self.log_table = None
        assert exp_no >= 0

        # pull data 
        try:
            self.feature_importance_table = load_pickle(f"../configs/configs-{self.comp_name}/feature_importance/feature_importance_e_{exp_no}.pkl")
        except:
            try:
                self.feature_importance_table = load_pickle(f"../configs/configs-{self.comp_name}/feature_importance/feature_importance_e_{exp_no}_fold.pkl")
                print(self.feature_importance_table.head(3))
                print("got")
            except:
                raise Exception("Feature importance is not saved!!")
        # fill nan 
        self.feature_importance_table.fillna(0,inplace=True)  

        try:
            self.log_table = load_pickle(f"../configs/configs-{self.comp_name}/logs/log_exp_{exp_no}.pkl")
        except:
            raise Exception("Log is not saved!!")


        self.scores = list(self.log_table.trial_score.astype(float))


        print("Trial Scores:- ")
        print(self.scores)
        print("---+"*30)
        print("Log Table")
        print(self.log_table)
        print("---+"*30)
        print("Feature Importance")
        print(self.feature_importance_table.head(2))

    def show(self,technique = "weighted_mean", top= 20, threshold=100, direction="maximize", pick = None, type_importance="opt", base_features=[]):
        
        filtered_trials = list(self.log_table.sort_values(by=['trial_score'], axis=0, ascending = False).index.values)
        # if pick is not none 
        if pick is not None:
            if technique == "weighted_mean":
                # time to pick trials 
                filtered_trials = list(self.log_table.sort_values(by=['trial_score'], axis=0, ascending = False).head(pick).index.values)
                # remove extra trials
                self.log_table = self.log_table.loc[filtered_trials]
                self.scores = list(self.log_table.trial_score.astype(float))
                self.feature_importance_table = self.feature_importance_table.iloc[:,[0]+ [i+1 for i in filtered_trials]]
            elif type_importance=="fold" and technique == "mean":
                filtered_trials = [0,1,2,3,4]
                self.feature_importance_table = self.feature_importance_table.iloc[:,[0]+ [i+1 for i in filtered_trials]]

        if technique == "mean":
            self.feature_importance_table[technique]= self.feature_importance_table.drop("feature", axis=1).mean(axis=1)
            if direction == "minimize":
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = True)
            else:
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = False)
            print()
            print(self.feature_importance_table.head(10))
        
        if technique == "bagging":
            L = self.feature_importance_table.drop("feature", axis=1)
            L = L<0
            L = L.sum(axis=1)
            self.feature_importance_table[technique] = L
            self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = False)
            print(self.feature_importance_table.head(3))
            
            #self.feature_importance_table[technique] = self.feature_importance_table[L>= 1]
        if technique == "weighted_mean":
            val = []
            for col,w in zip(self.feature_importance_table.drop("feature", axis=1).columns, self.scores):
                val.append(self.feature_importance_table[col]*w/sum(self.scores) )

            self.feature_importance_table[technique] = np.sum(np.array(val), axis=0)
            if direction == "minimize":
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = True)
            else:
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = False)
        if technique == "best":
            ind = self.scores.index(max(self.scores))
            useful_cols = self.feature_importance_table.drop("feature", axis=1).columns.values 
            best_col = useful_cols[ind]
            print("best feature:", best_col)
            val = self.feature_importance_table[best_col]
            self.feature_importance_table[technique] = val
            if direction == "minimize":
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = True)
            else:
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = False)
        try:
            val= self.feature_importance_table[["feature", technique]]
        except:
            print("couldn't")
            val= self.feature_importance_table[["feature", technique]]
        
        print("---+"*30)
        print("VALUE")
        #print(val)
        print()
        print([i.round(2) for i in val[technique].values])
        print(f"originally {val.shape[0]} features.")
        
        print("---+"*30)
        if top is not None:
            print(f"Top {top} features")
            print()
            if base_features is not None:
                filtered_feat = list(set(list(val.iloc[:top,:].feature))- set(base_features))
                print("Original length:", len(list(set(list(val.iloc[:top,:].feature)))))
                print("After filter:", len(filtered_feat))
                print(filtered_feat )
            else:
                print(list(val.iloc[:top,:].feature))
        elif threshold is not None:
            # threshold 
            print(f"Threshold of {threshold} features")
            if technique == "bagging":
                l = list(val.feature)
                val = val[val[technique] >= threshold]
                print([i.round(4) for i in val[technique].values])
                print(f"after filter: {val.shape[0]} features.")
                print(f"left features:", list(set(l)- set(val.feature)))
            elif direction == "maximize":
                val = val[val[technique]>= threshold]
                print([i.round(4) for i in val[technique].values])
                print(f"after filter: {val.shape[0]} features.")
            else:
                # minimize
                val = val[val[technique]<= threshold]
                print([i.round(4) for i in val[technique].values])
                print(f"after filter {val.shape[0]} features.")
            if base_features is not None:
                filtered_feat = list(set(val.feature)- set(base_features))
                print("Original length:", len(val.feature))
                print("After filter:", len(filtered_feat))
                print(filtered_feat )
            else:
                print(list(val.feature))
        else:
            if base_features is not None:
                filtered_feat = list(set(val.feature)- set(base_features))
                print("Original length:", len(val.feature))
                print("After filter:", len(filtered_feat))
                print(filtered_feat )
            else:
                print(list(val.feature))
        print("---+"*30)

    def give(self,technique = "weighted_mean", top= 20, threshold=100, direction="maximize", pick = None, type_importance="opt", base_features=[]):
        
        filtered_trials = list(self.log_table.sort_values(by=['trial_score'], axis=0, ascending = False).index.values)
        # if pick is not none 
        if pick is not None:
            if technique == "weighted_mean":
                # time to pick trials 
                filtered_trials = list(self.log_table.sort_values(by=['trial_score'], axis=0, ascending = False).head(pick).index.values)
                # remove extra trials
                self.log_table = self.log_table.loc[filtered_trials]
                self.scores = list(self.log_table.trial_score.astype(float))
                self.feature_importance_table = self.feature_importance_table.iloc[:,[0]+ [i+1 for i in filtered_trials]]
            elif type_importance=="fold" and technique == "mean":
                filtered_trials = [0,1,2,3,4]
                self.feature_importance_table = self.feature_importance_table.iloc[:,[0]+ [i+1 for i in filtered_trials]]

        if technique == "mean":
            self.feature_importance_table[technique]= self.feature_importance_table.drop("feature", axis=1).mean(axis=1)
            if direction == "minimize":
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = True)
            else:
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = False)
            print()
            print(self.feature_importance_table.head(10))
        
        if technique == "bagging":
            L = self.feature_importance_table.drop("feature", axis=1)
            L = L<0
            L = L.sum(axis=1)
            self.feature_importance_table[technique] = L
            self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = False)
            print(self.feature_importance_table.head(3))
            
            #self.feature_importance_table[technique] = self.feature_importance_table[L>= 1]
        if technique == "weighted_mean":
            val = []
            for col,w in zip(self.feature_importance_table.drop("feature", axis=1).columns, self.scores):
                val.append(self.feature_importance_table[col]*w/sum(self.scores) )

            self.feature_importance_table[technique] = np.sum(np.array(val), axis=0)
            if direction == "minimize":
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = True)
            else:
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = False)
        if technique == "best":
            ind = self.scores.index(max(self.scores))
            useful_cols = self.feature_importance_table.drop("feature", axis=1).columns.values 
            best_col = useful_cols[ind]
            print("best feature:", best_col)
            val = self.feature_importance_table[best_col]
            self.feature_importance_table[technique] = val
            if direction == "minimize":
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = True)
            else:
                self.feature_importance_table= self.feature_importance_table.sort_values(by=[technique], axis=0, ascending = False)
        try:
            val= self.feature_importance_table[["feature", technique]]
        except:
            print("couldn't")
            val= self.feature_importance_table[["feature", technique]]
        
        print("---+"*30)
        print("VALUE")
        #print(val)
        print()
        print([i.round(2) for i in val[technique].values])
        print(f"originally {val.shape[0]} features.")
        
        print("---+"*30)
        if top is not None:
            print(f"Top {top} features")
            print()
            if base_features is not None:
                filtered_feat = list(set(list(val.iloc[:top,:].feature))- set(base_features))
                print("Original length:", len(list(set(list(val.iloc[:top,:].feature)))))
                print("After filter:", len(filtered_feat))
                print(filtered_feat )
                return filtered_feat
            else:
                print(list(val.iloc[:top,:].feature))
                return list(val.iloc[:top,:].feature)
        elif threshold is not None:
            # threshold 
            print(f"Threshold of {threshold} features")
            if technique == "bagging":
                l = list(val.feature)
                val = val[val[technique] >= threshold]
                print([i.round(4) for i in val[technique].values])
                print(f"after filter: {val.shape[0]} features.")
                print(f"left features:", list(set(l)- set(val.feature)))
            elif direction == "maximize":
                val = val[val[technique]>= threshold]
                print([i.round(4) for i in val[technique].values])
                print(f"after filter: {val.shape[0]} features.")
            else:
                # minimize
                val = val[val[technique]<= threshold]
                print([i.round(4) for i in val[technique].values])
                print(f"after filter {val.shape[0]} features.")
            if base_features is not None:
                filtered_feat = list(set(val.feature)- set(base_features))
                print("Original length:", len(val.feature))
                print("After filter:", len(filtered_feat))
                print(filtered_feat )
                return filtered_feat
            else:
                print(list(val.feature))
                return list(val.feature)
        else:
            if base_features is not None:
                filtered_feat = list(set(val.feature)- set(base_features))
                print("Original length:", len(val.feature))
                print("After filter:", len(filtered_feat))
                print(filtered_feat )
                return filtered_feat
            else:
                print(list(val.feature))
                return list(val.feature)
        print("---+"*30)

from settings import *




if __name__ == "__main__":
    """
    best: picke features from beast trial
    mean: take simple mean of all the trials
    weighted mean: take weighted mean based on the score

    top: 20 filters top 20 
    threshold: 121 removes all whoose value is less than 121
    """


    exp_no = 142 #-1  120, 122, 127
    direction = "minimize"

    technique = "bagging" # "weighted_mean" , "best" , "mean", "top50", "bagging"

    f = Importance(exp_no=exp_no)
    # helps when doing weighted mean

  
    base_features = None

    type_importance = "fold" #"fold", "opt"
    pick = None # pick top 2 trials out of 5
    top = None
    threshold = 1
    f.show(technique= technique, top=top, threshold=threshold, direction=direction, pick = pick, type_importance= type_importance, base_features=base_features)
    
"""
# mean
['D_39_last', 'P_2_last', 'P_2_last_2round2', 'S_3_mean', 'B_4_last_mean_diff', 'D_39_last_mean_diff', 'D_41_last_mean_diff', 'B_1_last', 'S_3_last', 'B_4_last', 'D_43_last', 'D_42_mean', 'D_39_std', 'B_4_std', 'D_39_max', 'B_3_last', 'B_5_last', 'D_42_min', 'D_43_mean', 'D_43_last_2round2']
# best
['D_39_last_mean_diff', 'P_2_last', 'D_39_last', 'B_1_last', 'B_4_last', 'D_43_last', 'D_43_last_2round2', 'S_3_last', 'D_39_std', 'S_3_mean', 'P_2_last_2round2', 'D_42_mean', 'P_2_mean', 'D_43_mean', 'D_41_last_mean_diff', 'S_3_min', 'S_8_mean', 'D_39_max', 'B_3_last', 'B_3_last_2round2']
# weighted
['D_39_last', 'P_2_last', 'P_2_last_2round2', 'S_3_mean', 'B_4_last_mean_diff', 'D_39_last_mean_diff', 'D_41_last_mean_diff', 'B_1_last', 'S_3_last', 'B_4_last', 'D_43_last', 'D_42_mean', 'D_39_std', 'B_4_std', 'D_39_max', 'B_3_last', 'B_5_last', 'D_42_min', 'D_43_mean', 'D_43_last_2round2']
"""