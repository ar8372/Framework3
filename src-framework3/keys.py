from collections import defaultdict
import pickle
import os
import sys

"""
generates keys and stores it in models-ultramnist
"""


class KeyMaker:
    def __init__(
        self,
        random_state=21,
        target_name="Survived",
        id_name="PassengerId",
        comp_type="2class",
        metrics_name="accuracy",
        fold_dict={'fold5':5},
        data_type="image",  # ["image", "tabular", "text"]
    ):
        #
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self._comp_name = comp_name
        self.data_type = data_type
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
            "amex_metric",
        ]
        self.data_list = ["tabular", "image", "text"]
        self.comp_list = ["regression", "2class", "multi_class", "multi_label"]
        self.random_state = random_state
        self.target_name = target_name
        self.id_name = id_name
        self.comp_type = comp_type
        self.metrics_name = metrics_name
        self.fold_dict = fold_dict
        self.locker = defaultdict()

        self.sanity_check()  # --> sanity check
        self.update()  # dumps files as pickel

    def sanity_check(self):
        if self.comp_type not in self.comp_list:
            raise Exception(f"{self.comp_type} not in the list {self.comp_list}")
        if self.metrics_name not in self.metrics_list:
            raise Exception(f"{self.metrics_name} not in the list {self.metrics_name}")
        if self.data_type not in self.data_list:
            raise Exception(f"{self.data_type} not in the list {self.data_type}")

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
        fold_dict="--|--",
        data_type="--|--",
    ):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../configs/configs-{comp_name}/locker.pkl", "rb") as f:
            a = pickle.load(f)
        self.random_state = a["random_state"]
        self.target_name = a["target_name"]
        self.id_name = a["id_name"]
        self.comp_type = a["comp_type"]
        self.metrics_name = a["metrics_name"]
        self.fold_dict = a["fold_dict"]
        self.data_type = a["data_type"]

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
        if fold_dict != "--|--":
            self.fold_dict = fold_dict
        if data_type != "--|--":
            self.data_type = data_type

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
        self.locker["fold_dict"] = self.fold_dict
        self.locker["data_type"] = self.data_type

        with open(f"../configs/configs-{a['comp_name']}/locker.pkl", "wb") as f:
            pickle.dump(self.locker, f)

    def show_stored_keys(self):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        with open(f"../configs/configs-{comp_name}/locker.pkl", "rb") as f:
            a = pickle.load(f)
        for k, v in a.items():
            print(f"{k}:", v)

    def show_variables(self):
        print()
        for i, (k, v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>", v)
        print()


if __name__ == "__main__":
    x = KeyMaker()
    # # ultramnist
    # x.id_name = "id"
    # x.target_name = "digit_sum"
    # x.comp_type = "multi_class"
    # x.no_folds = 5
    # x.data_type = "image_path"

    ## mnist
    # x.id_name = "ImageId"
    # x.target_name = "Label"
    # x.comp_type = "multi_class"
    # x.no_folds = 5
    # x.data_type = "image_df"  # image_path, image_df, image_folder

    # twistmnist
    # x.id_name = "image_id"
    # x.target_name = "label"
    # x.comp_type = "multi_class"
    # x.no_folds = 5
    # x.data_type = "image_df"

    # bengaliai
    # x.id_name = "image_id"
    # x.target_name = ['grapheme_root','vowel_diacritic','consonant_diacritic']
    # x.comp_type = "multi_label"
    # x.no_folds = 5
    # x.data_type = "image_path"

    # # tmay
    # x.id_name = "id"
    # x.target_name = "target"
    # x.comp_type = "2class"
    # x.no_folds = 5
    # x.data_type = "tabular"

    # # # amex
    # x.id_name = "customer_ID"
    # x.target_name = "prediction"
    # x.comp_type = "2class"
    # x.fold_dict =    {           # no_folds : replace it
    #     "fold3": 3,
    #     "fold5": 5,
    #     "fold10": 10,
    #     "fold20": 20
    # }                                          # 10
    # x.data_type = "tabular"

    # amex5
    # x.id_name = "customer_ID"
    # x.target_name = "prediction"
    # x.comp_type = "2class"
    # x.no_folds = 5
    # x.data_type = "tabular"


    # # getaroom
    # x.id_name = "Property_ID"
    # x.target_name = "Habitability_score"
    # x.comp_type = "regression"
    # x.fold_dict =    {           # no_folds : replace it
    #     "fold3": 3,
    #     "fold5": 5,
    #     "fold10": 10,
    #     "fold20": 20
    # }                                          # 10
    # x.data_type = "tabular"


    x.id_name = "ID"
    x.target_name = "Time_taken"
    x.comp_type = "regression"
    x.fold_dict =    {           # no_folds : replace it
        "fold3": 3,
        "fold5": 5,
        "fold10": 10,
        "fold20": 20
    }                                          # 10
    x.data_type = "tabular"

    x.update()
    # # x.show_stored_keys()
