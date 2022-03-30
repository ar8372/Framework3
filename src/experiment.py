from optuna_search import OptunaOptimizer
from feature_generator import features
from feature_picker import Picker
import os
import sys
import pickle
import pandas as pd
#from custom_models import UModel
#from custom_models import *
from utils import *

class Agent:
    def __init__(
        self,
        useful_features=[],
        model_name="",
        comp_type="2class",
        metrics_name="accuracy",
        n_trials=5,
        prep_list=[],
        optimize_on=0,
        save_models=True,
        with_gpu=False,
    ):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.locker = load_pickle(f"../models_{comp_name}/locker.pkl")
        self.current_dict = load_pickle(f"../models_{comp_name}/current_dict.pkl")
        # ----------------------------------------------------------
        self.useful_features = useful_features
        self.model_name = model_name
        self.comp_type = comp_type
        self.metrics_name = metrics_name
        self.n_trials = n_trials
        self.prep_list = prep_list
        self.optimize_on = optimize_on
        self.save_models = True
        self.with_gpu = True

    def sanity_check(self):
        if "--|--" in [
            self.useful_features,
            self.model_name,
            self.comp_type,
            self.metrics_name,
            self.n_trials,
            self.prep_list,
            self.optimize_on,
            self.save_models,
        ]:
            raise Exception("Found --|--- while sanity check!")

    def run(
        self,
        useful_features="--|--",
        model_name="--|--",
        comp_type="--|--",
        metrics_name="--|--",
        n_trials="--|--",
        prep_list="--|--",
        optimize_on="--|--",
        save_models="--|--",
        with_gpu="--|--",
    ):
        if useful_features != "--|--":
            self.useful_features = useful_features
        if model_name != "--|--":
            self.model_name = model_name
        if comp_type != "--|--":
            self.comp_type = comp_type
        if metrics_name != "--|--":
            self.metrics_name = metrics_name
        if n_trials != "--|--":
            self.n_trials = n_trials
        if prep_list != "--|--":
            self.prep_list = prep_list
        if optimize_on != "--|--":
            self.optimize_on = optimize_on
        if save_models != "--|--":
            self.save_models = save_models
        if with_gpu != "--|--":
            self.with_gpu = with_gpu

        self.sanity_check()
        my_folds = pd.read_csv(f"../models_{self.locker['comp_name']}/my_folds.csv")
        opt = OptunaOptimizer(
            model_name=self.model_name,
            comp_type=self.comp_type,
            metrics_name=self.metrics_name,
            n_trials=self.n_trials,
            prep_list=self.prep_list,
            optimize_on=self.optimize_on,
            with_gpu=self.with_gpu,
            save_models=self.save_models,
        )
        self.study, random_state, seed_mean, seed_std = opt.run(
            my_folds, self.useful_features
        )
        if self.save_models == True:
            self._save_models(self.study, random_state, seed_mean, seed_std)

        # Let's make perdiction on Test Set:
        # self._seed_it()

    def get_exp_no(self):
        # exp_no, current_level
        self.current_dict = load_pickle(
            f"../models_{self.locker['comp_name']}/current_dict.pkl"
        )
        self.current_exp_no = int(self.current_dict["current_exp_no"])

    def _save_models(self, study, random_state, seed_mean, seed_std):
        Table = load_pickle(f"../models_{self.locker['comp_name']}/Table.pkl")
        Table = pd.DataFrame(Table)
        # what unifies it
        self.get_exp_no()
        # ExpNo- self.current_exp_no
        self.current_exp_no += 1
        Table.loc[Table.shape[0], :] = [
            self.current_exp_no,
            self.model_name,
            study.best_trial.value,
            study.best_trial.params,
            random_state,
            self.with_gpu,
            self.useful_features,
            self.current_dict["current_level"],
            self.optimize_on,
            self.n_trials,
            self.prep_list,
            self.metrics_name,
            seed_mean,
            seed_std,
            None,
            None,
            None,
            None,
            None
        ]

        # --------------- dump experiment no
        self.current_dict["current_exp_no"] = self.current_exp_no
        save_pickle(
            f"../models_{self.locker['comp_name']}/current_dict.pkl", self.current_dict
        )
        # ---------------- dump table
        save_pickle(f"../models_{self.locker['comp_name']}/Table.pkl", Table)

    def show_variables(self):
        print()
        for i, (k, v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>", v)
        print()


if __name__ == "__main__":
    # ==========================================================
    list_levels = ["1"]  # ---------------> ["1","2"]
    list_features = ["0", "1", "2"]  # ---------------> ["0","1","2"]
    list_feat_title = [
        "create_statistical_features"
    ]  # ---------------->["base", "create_statistical_features"]
    # ---------------------------------------------------------
    p = Picker()
    useful_features = p.find_features(
        list_levels=list_levels,
        list_features=list_features,
        list_feat_title=list_feat_title,
    )
    useful_features = ["id"]  # ["SibSp", "Parch", "Pclass"]
    # ==========================================================
    model_name = "tez1"  # -------->["lgr","lir","xgbc","xgbr","cbc","mlpc", "rg", "ls","knnc", "dtc", "adbc", "gbmc" ,"hgbc", "lgbmc", "lgbmr", "rfc" , "k1", "k2", "k3" ]
    # --------------->["tez1"]
    comp_type = (
        "multi_class"  # -------->["regression", "2class","multi_class", "multi_label"]
    )
    metrics_name = "accuracy"  # --------->["accuracy","f1","recall","precision", "auc", "logloss","auc_tf","mae","mse","rmse","msle","rmsle","r2"]
    n_trials = 2  # ------------> no of times to run optuna
    prep_list = [
        "Sd",
    ]  # ------> ["SiMe", "SiMd", "SiMo", "Mi", "Ro", "Sd", "Lg"] <= _prep_list
    optimize_on = 0  # fold on which optimize
    with_gpu = True
    # -----------------------------------------------------------

    e = Agent(
        useful_features=useful_features,
        model_name=model_name,
        comp_type=comp_type,
        metrics_name=metrics_name,
        n_trials=n_trials,
        prep_list=prep_list,
        optimize_on=optimize_on,
        with_gpu=with_gpu,
    )
    print("=" * 40)
    print("Useful_features:", useful_features)

    e.run()

    # -------------------------------------------------------------
    # exp_list = ["1"]  # ----------------> [1,2,3,4]
    # e.show(exp_list)
