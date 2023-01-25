"""
This does nothing but submits one prediction.
"""
from utils import * 

#pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

class Moderator:
    def __init__(self, comp_full_name, explore_range= ["pblb_all_fold"], sort_by_col = "bv", mode="auto"):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
        self.Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")
        self.comp_full_name = comp_full_name
        self.explore_range = explore_range 
        self.sort_by_col = sort_by_col
        self.mode = mode

        self.ascending = False
        if sort_by_col in ["fold_std",  "seed_std"]:
            self.ascending = True

        # filter by             THRESHOLD
        # bv 
        self.Table = self.Table[self.Table.bv > 0.75]
        # no_iterations 
        #self.Table = self.Table[self.Table.no_iterations > 5]
        # fold_mean 
        #self.Table = self.Table[self.Table.fold_mean > 0.75]
        # fold_std 
        #self.Table = self.Table[self.Table.fold_std < 0.001]
        # seed_mean 
        #self.Table = self.Table[self.Table.seed_mean > 0.79]
        # seed_std 
        #self.Table = self.Table[self.Table.seed_std < 0.005]


        # Sort by:
        self.Table = self.Table.sort_values(by=[sort_by_col], ascending=self.ascending)

        self.show()
    
    def query(self):
        print("Type:")
        print("sub: to submit in this exp")
        print("down: to move to next row")
        text = input()
        if text.lower() == "down":
            print()
            return 0 
        elif text.lower() == "sub":
            print("+.."*40)
            # submit this
            print("Type prediction type for which you want to make submission!: ")
            print("single, all, fold3, fold5, fold10, fold20")
            fold_type = input()
            # fold_type : "all", "single", "fold3", "fold10"
            from submit import submit_pred
            submit_pred(self.comp_full_name, self.target_exp_no, fold_type, self.mode) 
            self.submitted = True
            return 1 
        raise Exception("Should never reach here, Not a valid input")

    def submit(self):
        self.submitted = False 
        pos = 0
        size = self.Table.shape[0]
        while self.submitted is False and pos< size :
            self.target_exp_no = self.Table.iloc[pos,0]
            oof_fold_name_list = self.Table.loc[self.target_exp_no, "oof_fold_name"] # ["fold5", "fold3", "fold10"]

            print("="*40)
            print("Experiment no:",self.target_exp_no)
            explore_range = ["pblb_all_fold", "pblb_single_seed","pblb_all_seed"]
            cv_cols = ["oof_fold_name", "fold_mean","fold_std", "seed_mean", "seed_std"]
            print(self.Table[self.Table.exp_no== self.target_exp_no][["bv"] +explore_range+ cv_cols])

            if self.mode == "manual":
                no= self.query() 
                if no == 0:
                    pos += 1
                    continue 
                elif no == 1:
                    break 

            #submitted = True
            for feat in self.explore_range:
                # "pblb_all_fold", "pblb_single_seed","pblb_all_seed"
                val= self.Table.loc[self.target_exp_no, feat]
                if isinstance(val, list) and None in val: # so we need to repeat it for that
                    # [ 0.23, None, None, 0.22,0.56,None]
                    for i in range(len(val)):
                        if val[i] is None:
                            # automatically submit this as not submitted before
                            from submit import submit_pred
                            submit_pred(self.comp_full_name, self.target_exp_no, fold_type, self.mode) 
                            # stopping execution since submitted
                            return 
                         

                if val in [None, []]: # at present it deals with only [] oof_fold_name .i.e no prediction is made yet [None, 0.9, None] won't work for now
                    # None, []
                    if feat.split("_")[-1] == "fold":
                        # no prediction is made yet
                        print("Type prediction fold_name for which you want to make submission!: ")
                        print("fold3, fold5, fold10, fold20")
                        fold_type = input()
                        # fold_type : "all", "single", "fold3", "fold10"
                        from submit import submit_pred
                        submit_pred(self.comp_full_name, self.target_exp_no, fold_type, self.mode) 
                        self.submitted = True
                        break
                    fold_type = feat.split("_")[1] # "single", "all" :-> seed 

                    # automatically submit this as not submitted before
                    from submit import submit_pred
                    submit_pred(self.comp_full_name, self.target_exp_no, fold_type, self.mode) 
                    # stopping execution since submitted
                    return 
            pos += 1

    def show(self):
        # anything extra added will cause it to move below: "seed_std",  "fold_std"
        print(self.Table[["exp_no","model_name","no_iterations",self.sort_by_col] + self.explore_range + ["fold_mean", "oof_fold_name", "seed_mean"]].set_index('exp_no'))

submissions = ["seed_mean", "seed_std","fold_mean","fold_std","pblb_single_seed","pblb_all_seed","pblb_all_fold"]
general = ["prep_list","opt_fold_name","oof_fold_name", "fold_no","no_iterations"]
base = ['exp_no',"bv"]


pblb_cols = ["pblb_all_fold","pblb_single_seed","pblb_all_seed"]
cv_cols = ["oof_fold_name", "fold_mean","fold_std", "seed_mean", "seed_std"]
opt_cols = ["bv","no_iterations","prep_list","opt_fold_name","fold_no", "with_gpu"] #, "bp", "random_state", "metrics_name", "feaures_list" # things to reproduce prediction from an experiment

if __name__ == "__main__":
    comp_full_name = "amex-default-prediction"
    # explore range searches for what we have to submit
    # if set only "fold" It will look if fold public score is there or not
    # If not it will prompt/auto submit.
    explore_range = ["pblb_all_fold", "pblb_single_seed","pblb_all_seed"] 
    #["fold", "seed_single", "seed_all"]

    sort_by_col = "bv" # "bv", "", "fold_mean", "fold_std", "seed_mean", "seed_std"
    # if it is sort by "bv" then no issue 
    # but if is is by fold/seed then we may have not done "predict.py"/"seed_it.py"

    mode = "auto" # "auto", "manual"
    # auto: looks for the first None and sends it for submission , It asks only once when we encounter [], which fold_name to predict 
    # manual: asks at each step what to select
    m = Moderator(comp_full_name, explore_range, sort_by_col, mode)

    m.submit()