from collections import defaultdict
import pickle
import os
import sys


class KeyMaker:
    def __init__(
        self,
        random_state=21,
        target_name="Survived",
        id_name="PassengerId",
        comp_type="2class",
        metrics_name="accuracy",
        no_folds=5,
    ):
        #
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self._comp_name = comp_name

        self.metrics_list = [
            "accuracy",
            "f1",
            "recall",
            "precision",
            "auc",
            "logloss",
            "auc_tf",
            "mae",
            "mse",
            "rmse",
            "msle",
            "rmsle",
            "r2",
        ]
        self.comp_list = ["regression", "2class", "multi_class", "multi_label"]
        self.random_state = random_state
        self.target_name = target_name
        self.id_name = id_name
        self.comp_type = comp_type
        self.metrics_name = metrics_name
        self.no_folds = no_folds
        self.locker = defaultdict()

        self.sanity_check()  # --> sanity check
        self.update()  # dumps files as pickel

    def sanity_check(self):
        if self.comp_type not in self.comp_list:
            raise Exception(f"{self.comp_type} not in the list {self.comp_list}")
        if self.metrics_name not in self.metrics_list:
            raise Exception(f"{self.metrics_name} not in the list {self.metrics_name}")

    def help(self):
        print("comp_type:=> ", [comp for i, comp in enumerate(self.comp_list)])
        print("metrics_index:=>", [mt for i, mt in enumerate(self.metrics_list)])

    def __call__(
        self,
        random_state="--|--",
        target_name="--|--",
        id_name="--|--",
        comp_type="--|--",
        metrics_name="--|--",
        no_folds="--|--",
    ):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
            a = pickle.load(f)
        self.random_state = a["random_state"]
        self.target_name = a["target_name"]
        self.id_name = a["id_name"]
        self.comp_type = a["comp_type"]
        self.metrics_name = a["metrics_name"]
        self.no_folds = a["no_folds"]

        if random_state != "--|--":
            # updated
            self.random_state = random_state
        if target_name != "--|--":
            # updated
            self.target_name = target_name
        if id_name != "--|--":
            # updated
            self.id_name = id_name
        if comp_type != "--|--":
            self.comp_type = comp_type
        if metrics_name != "--|--":
            self.metrics_name = metrics_name
        if no_folds != "--|--":
            self.no_folds = no_folds

        self.sanity_check()
        self.update()  # dump files to pickel

    def update(self):
        # updates the locker
        a = self.locker
        self.locker["comp_name"] = self._comp_name
        self.locker["random_state"] = self.random_state
        self.locker["target_name"] = self.target_name
        self.locker["id_name"] = self.id_name
        self.locker["comp_type"] = self.comp_type
        self.locker["no_folds"] = self.no_folds

        with open(f"../models_{a['comp_name']}/locker.pkl", "wb") as f:
            pickle.dump(self.locker, f)

    def show_keys(self):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../models_{comp_name}/locker.pkl", "rb") as f:
            a = pickle.load(f)
        for k, v in a.items():
            print(f"{k}:", v)


if __name__ == "__main__":
    x = KeyMaker()
