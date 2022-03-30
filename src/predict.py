from optuna_search import OptunaOptimizer
from utils import *
from custom_models import *
from custom_classes import *
from utils import *
import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

"""
Inference file
"""


class predictor(OptunaOptimizer):
    def __init__(self, exp_no):
        self.exp_no = exp_no
        # initialize rest
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                self.comp_name = i
        x.close()
        self.Table = load_pickle(f"../models_{self.comp_name}/Table.pkl")
        self.locker = load_pickle(f"../models_{self.comp_name}/locker.pkl")

        row_e = self.Table[self.Table.exp_no == self.exp_no]
        self.model_name = row_e.model_name.values[0]
        self.params = row_e.bp.values[0]
        self.random_state = row_e.random_state.values[0]
        self.with_gpu = row_e.with_gpu.values[0]
        self.features_list = row_e.features_list.values[0]
        self.prep_list = row_e.prep_list.values[0]
        self.metrics_name = row_e.metrics_name.values[0]
        self.level_no = row_e.level_no.values[0]
        self.useful_features = row_e.features_list.values[0]

        super().__init__(
            model_name=self.model_name,
            comp_type=self.locker["comp_type"],
            metrics_name=self.metrics_name,
            prep_list=self.prep_list,
            with_gpu=self.with_gpu,
        )

    def run_folds(self):
        self._state = "fold"
        image_path = f'../input_{self.locker["comp_name"]}/' + "train_img/"
        my_folds = pd.read_csv(f"../models_{self.comp_name}/my_folds.csv")
        test = pd.read_csv(f"../models_{self.comp_name}/test.csv")
        scores = []
        oof_prediction = {}
        test_predictions = []

        for fold in range(5):
            # select data: xtrain xvalid etc
            self.run(my_folds, self.useful_features)
            scores.append(self.obj("--|--"))
            oof_prediction.update(dict(zip(self.val_idx, self.valid_preds)))  # oof
            test_predictions.append(self.test_preds)
        # save oof predictions
        temp_valid_predictions = pd.DataFrame.from_dict(
            oof_prediction, orient="index"
        ).reset_index()
        temp_valid_predictions.columns = [
            f"{self.locker['id_name']}",
            f"pred_l_{self.current_dict['current_level']}_e_{self.exp_no}",
        ]
        my_folds[
            f"pred_l_{self.current_dict['current_level']}_e_{self.exp_no}"
        ] = temp_valid_predictions[
            f"pred_l_{self.current_dict['current_level']}_e_{self.exp_no}"
        ]
        my_folds.to_csv(
            f"../models_{self.locker['comp_name']}/my_folds.csv", index=False
        )
        # save temp predictions
        test[
            f"pred_l_{self.current_dict['current_level']}_e_{self.exp_no}"
        ] = stats.mode(np.column_stack(test_predictions), axis=1)[0]
        test.to_csv(f"../models_{self.comp_name}/test.csv", index=False)

        # ---------------
        new_features = [f"pred_l_{self.current_dict['current_level']}_e_{self.exp_no}"]
        useful_features = self.useful_features
        # -----------------------------update current dict
        self.current_dict["current_feature_no"] = (
            self.current_dict["current_feature_no"] + 1
        )
        feat_no = self.current_dict["current_feature_no"]
        level_no = self.current_dict["current_level"]
        save_pickle(
            f"../models_{self.locker['comp_name']}/current_dict.pkl", self.current_dict
        )
        # -----------------------------dump feature dictionary
        feat_dict = load_pickle(
            f"../models_{self.locker['comp_name']}/features_dict.pkl"
        )
        feat_dict[f"l_{level_no}_f_{feat_no}"] = [
            new_features,
            useful_features,
            f"exp_{self.exp_no}",
        ]
        save_pickle(
            f"../models_{self.locker['comp_name']}/features_dict.pkl", feat_dict
        )
        # -----------------------
        print("New features create:- ")
        print(new_features)
        # -----------------------------
        print("scores: ")
        print(scores)

        # ---- update table
        self.Table.loc[self.Table.exp_no == self.exp_no, "fold_mean"] = np.mean(scores)
        self.Table.loc[self.Table.exp_no == self.exp_no, "fold_std"] = np.std(scores)
        self.Table.loc[self.Table.exp_no == self.exp_no, "pblb_single_seed"] = None
        self.Table.loc[self.Table.exp_no == self.exp_no, "pblb_all_seed"] = None
        self.Table.loc[self.Table.exp_no == self.exp_no, "pblb_all_fold"] = None
        # pblb to be updated mannually
        # ---------------- dump table
        save_pickle(f"../models_{self.locker['comp_name']}/Table.pkl", self.Table)


if __name__ == "__main__":
    p = predictor(exp_no=1)
    p.run_folds()
