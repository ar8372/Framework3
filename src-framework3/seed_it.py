from optuna_search import OptunaOptimizer
from utils import *
from custom_models import *
from custom_classes import *
from utils import *
import os
import gc
import sys
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.preprocessing import RobustScaler

"""
Seed file
Trains of full dataset for n different seeds and 
Outputs two files
Single seed ( output based on one of the seed)
all seed (ensemble of all the seeds)
"""


class seeds(OptunaOptimizer):
    def __init__(self, exp_no):
        self.exp_no = exp_no
        # initialize rest
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                self.comp_name = i
        x.close()
        self.Table = load_pickle(f"../configs/configs-{self.comp_name}/Table.pkl")
        self.locker = load_pickle(f"../configs/configs-{self.comp_name}/locker.pkl")

        if self.exp_no == -1:
            row_e = self.Table[self.Table.exp_no == list(self.Table.exp_no.values)[-1]]
            self.exp_no = row_e.exp_no.values[0]
        else:
            row_e = self.Table[self.Table.exp_no == self.exp_no]
        self.model_name = row_e.model_name.values[0]
        self.params = row_e.bp.values[0]
        self.bv = row_e.bv.values[0]  # confirming we are predcting correct experiment:
        print(f"Predicting Exp No {self.exp_no}, whoose bv is {self.bv}")
        if self.model_name == "lgr":
            del self.params["c"]
        self._random_state = row_e.random_state.values[0] # This value will be changed
        self.with_gpu = row_e.with_gpu.values[0]
        self.features_list = row_e.features_list.values[0]
        self.prep_list = row_e.prep_list.values[0]
        self.metrics_name = row_e.metrics_name.values[0]
        self.level_no = row_e.level_no.values[0]
        self.useful_features = row_e.features_list.values[0]
        self.aug_type = row_e.aug_type.values[0]
        self._dataset = row_e._dataset.values[0]
        self.use_cutmix = row_e.use_cutmix.values[0]

        super().__init__(
            model_name=self.model_name,
            comp_type=self.locker["comp_type"],
            metrics_name=self.metrics_name,
            prep_list=self.prep_list,
            with_gpu=self.with_gpu,
            aug_type=self.aug_type,
            _dataset=self._dataset,
            use_cutmix=self.use_cutmix,
        )
        # When we call super() It is like calling their init 
        # so all the default initialization of parent class is made here
        # So we must manually change it here after doing super() 
        # if we did it before calling super() it will be overwritten by parent init
        # Overrite exp_no of OptunaOptimizer since it takes exp_no from current_dict 
        self.exp_no = exp_no 
        if self.exp_no == -1:
            row_e = self.Table[self.Table.exp_no == list(self.Table.exp_no.values)[-1]]
            self.exp_no = row_e.exp_no.values[0]        
        # --- sanity check [new_feat, old_feat, feat_title]
        # ---------------
        self.feat_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/features_dict.pkl"
        )
        useful_features = self.useful_features

        # It's ok to run seed any no of time but it's not ok to run predict 
        # so sanity check in predict but not in seed_it

    def run_seeds(self):
        check_memory_usage("run seeds started", self, 0)
        ######################################
        #         Memory uage                #
        ######################################
        tracemalloc.start()
        # -------------------------------------->
        print("SEEDING")
        self._state = "seed"
        self.generate_random_no()
        no_seeds = 5 # 20 #3
        random_list = np.random.randint(1, 1000, no_seeds)  # 100
        print(f"Running {no_seeds} seeds!")
        """
        Use full train set and test set. call it train and valid
        """
        # --> test set
        # read only what is necessary
        # self.test = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/" + "test.csv")[self.useful_features + [ self.locker["id_name"]] ]
        # self.test = pd.read_parquet(
        #     f"../input/input-{self.locker['comp_name']}/" + "test.parquet",
        #     columns=self.useful_features + [self.locker["id_name"]],
        # )  # [self.useful_features + [ self.locker["id_name"]] ]
        # self.test[self.locker["target_name"]] = 0.0

        # # Create Folds since deleted in run
        # # self.my_folds = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/my_folds.csv")[ [ self.locker["id_name"], self.locker["target_name"]] + self.useful_features ]
        # self.my_folds = pd.read_parquet(
        #     f"../input/input-{self.locker['comp_name']}/my_folds.parquet",
        #     columns=[self.locker["id_name"], self.locker["target_name"]]
        #     + self.useful_features,
        # )  # [ [ self.locker["id_name"], self.locker["target_name"]] + self.useful_features ]

        # if not multi_label
        # keep sample to input
        # self.sample = pd.read_csv(
        #     f"../input/input-{self.locker['comp_name']}/" + "sample.csv"
        # )
#        # self.sample = pd.read_parquet(
        #     f"../input/input-{self.locker['comp_name']}/" + "sample.parquet"
        # )
        # if self.locker["comp_type"] == "multi_label":
        #     self.sample = self.test.copy() # temp
        # else:
        #     self.sample = pd.read_csv(
        #         f"../input/input-{self.locker['comp_name']}/" + "sample.csv"
        #     )
        # BOTTLENECK 
        return_type = "numpy_array"
        self.optimize_on = None # just to make sure it is not called 
        fold_name = "fold_check"
        self.val_idx, self.xtrain, self.xvalid, self.ytrain, self.yvalid, self.ordered_list_train = bottleneck(self.locker['comp_name'],self.useful_features, fold_name, self.optimize_on, self._state, return_type)
        self.xvalid = None 
        self.yvalid = None 
        self.val_idx = None 
        
        self.xtest, self.ordered_list_test = bottleneck_test(self.locker['comp_name'], self.useful_features, return_type)
        # sanity check: 
        for i,j in zip(self.ordered_list_test, self.ordered_list_train):
            if i != j:
                raise Exception(f"Features don't correspond in test - train {i},{j}")
        self.ordered_list_test = None 
        self.ordered_list_train = None 

        # print("self.xtrain.shape, self.ytrain.shape, self.xtest.shape")
        # print(self.xtrain.shape, self.ytrain.shape, self.xtest.shape)

        if self.locker["data_type"] == "image_path":
            image_path = f"../input/input-{self.locker['comp_name']}/" + "train_img/"
            # test_path = f"../input/input-{self.locker['comp_name']}/" + "test_img/"
            if self.model_name in ["tez1", "tez2", "pretrained"]:
                # now implemented for pytorch
                print("one")
                # use pytorch
                self.train_image_paths = [
                    os.path.join(image_path, str(x))
                    for x in self.my_folds[self.locker["id_name"]].values  # =>
                ]
                print("two")
                self.valid_image_paths = [
                    os.path.join(image_path, str(x))
                    for x in self.my_folds[self.locker["id_name"]].values  # =>
                ]
                print("three")
                self.ytrain = self.my_folds[self.locker["target_name"]].values  # =>
                self.yvalid = self.my_folds[self.locker["target_name"]].values  # =>
                # ------------------  prep test dataset
                # self.test_image_paths = [
                #     os.path.join(
                #         test_path, str(x)
                #     )  # f"../input/input-{self.locker['comp_name']}/" + "test_img/" + x
                #     for x in self.sample[self.locker["id_name"]].values
                # ]
                print("four")
                # fake targets
                # self.test_targets = self.sample[
                #     self.locker["target_name"]
                # ].values  # dfx_te.digit_sum.values
                print("five")
                if self._dataset in [
                    "BengaliDataset",
                ]:
                    self.train_dataset = BengaliDataset(  # train_dataset
                        image_paths=self.train_image_paths,
                        targets=self.ytrain,
                        img_height=128,
                        img_width=128,
                        transform=self.train_aug,
                    )
                    print("six")
                    self.valid_dataset = BengaliDataset(  # train_dataset
                        image_paths=self.valid_image_paths,
                        targets=self.yvalid,
                        img_height=128,
                        img_width=128,
                        transform=self.valid_aug,
                    )
                    print("seven")
                    # already defined
                    # self.test_dataset = BengaliDataset(  # train_dataset
                    #     image_paths=self.test_image_paths,
                    #     targets=self.test_targets,
                    #     img_height = 128,
                    #     img_width = 128,
                    #     transform=self.valid_aug,
                    # )
                    print("eight")
                    # now implemented for pytorch
                    # Can make our own custom dataset.. Note tez has dataloader inside the model so don't make
                else:
                    # imageDataset
                    self.train_dataset = ImageDataset(  # train_dataset
                        image_paths=self.train_image_paths,
                        targets=self.ytrain,
                        augmentations=self.aug,
                    )

                    self.valid_dataset = ImageDataset(  # valid_dataset
                        image_paths=self.valid_image_paths,
                        targets=self.yvalid,
                        augmentations=self.aug,
                    )

                    # self.test_dataset = ImageDataset(
                    #     image_paths=self.test_image_paths,
                    #     targets=self.test_targets,
                    #     augmentations=self.aug,
                    # )
            elif self.model_name in ["k1", "k2", "k3"]:
                # now implemented for keras
                # use keras flow_from_dataframe
                train_datagen = ImageDataGenerator(rescale=1.0 / 255)
                valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

                if self.use_cutmix != True:
                    self.train_dataset = train_datagen.flow_from_dataframe(
                        dataframe=self.my_folds,
                        directory=image_path,
                        target_size=(28, 28),  # images are resized to (28,28)
                        x_col=self.locker["id_name"],
                        y_col=self.locker["target_name"],
                        batch_size=32,
                        seed=42,
                        shuffle=True,
                        class_mode="categorical",  # "binary"
                    )
                elif self.use_cutmix == True:
                    train_datagen1 = train_datagen.flow_from_dataframe(
                        dataframe=self.my_folds,
                        directory=image_path,
                        target_size=(28, 28),  # images are resized to (28,28)
                        x_col=self.locker["id_name"],
                        y_col=self.locker["target_name"],
                        batch_size=32,
                        seed=42,
                        shuffle=True,  # Required for cutmix
                        class_mode="categorical",  # "binary"
                    )
                    train_datagen2 = train_datagen.flow_from_dataframe(
                        dataframe=self.my_folds,
                        directory=image_path,
                        target_size=(28, 28),  # images are resized to (28,28)
                        x_col=self.locker["id_name"],
                        y_col=self.locker["target_name"],
                        batch_size=32,
                        seed=42,
                        shuffle=True,  # Required for cutmix
                        class_mode="categorical",  # "binary"
                    )
                    self.train_dataset = CutMixImageDataGenerator(
                        generator1=train_generator1,
                        generator2=train_generator2,
                        img_size=(28, 28),
                        batch_size=32,
                    )
                self.valid_dataset = valid_datagen.flow_from_dataframe(
                    dataframe=self.my_folds,
                    directory=image_path,
                    target_size=(28, 28),  # images are resized to (28,28)
                    x_col=self.locker["id_name"],
                    y_col=self.locker["target_name"],
                    batch_size=32,
                    seed=42,
                    shuffle=True,
                    class_mode="categorical",  # "binary"
                )

                test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
                test_generator = test_datagen.flow_from_dataframe(
                    dataframe=self.test,
                    directory=test_path,
                    target_size=(28, 28),  # images are resized to (28,28)
                    x_col=self.locker["id_name"],
                    y_col=None,
                    batch_size=32,
                    seed=42,
                    shuffle=True,
                    class_mode="None",  # "binary"
                )

        elif self.locker["data_type"] == "image_df":
            # here we create filtered_features from useful_features
            # self.test = pd.read_csv(
            #     f"../configs/configs-{self.locker['comp_name']}/" + "test.csv"
            # )
            self.test = pd.read_parquet(
                f"../input/input-{self.locker['comp_name']}/" + "test.parquet"
            )
            self.yvalid = self.my_folds[self.locker["target_name"]]
            self.ytrain = self.my_folds[self.locker["target_name"]]
            # create fake labels
            self.test[self.locker["target_name"]] = 0.0
            if self._dataset == "DigitRecognizerDataset":
                # DigitRecognizerDataset
                self.train_dataset = DigitRecognizerDataset(
                    df=self.my_folds[
                        self.filtered_features + [self.locker["target_name"]]
                    ],
                    augmentations=self.train_aug,
                    model_name=self.model_name,
                )

                self.valid_dataset = DigitRecognizerDataset(
                    df=self.my_folds[
                        self.filtered_features + [self.locker["target_name"]]
                    ],
                    augmentations=self.valid_aug,
                    model_name=self.model_name,
                )
                self.test_dataset = DigitRecognizerDataset(
                    df=self.test[self.filtered_features + [self.locker["target_name"]]],
                    augmentations=self.valid_aug,
                    model_name=self.model_name,
                )

            elif self._dataset == "BengaliDataset":
                # now implemented for pytorch
                # Can make our own custom dataset.. Note tez has dataloader inside the model so don't make
                self.train_dataset = BengaliDataset(  # train_dataset
                    csv=self.my_folds[
                        self.filtered_features + [self.locker["target_name"]]
                    ],
                    img_height=28,
                    img_width=28,
                    transform=self.train_aug,
                )

                self.valid_dataset = BengaliDataset(  # valid_dataset
                    csv=self.my_folds[
                        self.filtered_features + [self.locker["target_name"]]
                    ],
                    img_height=28,
                    img_width=28,
                    transform=self.valid_aug,
                )
                self.test_dataset = BengaliDataset(
                    df=self.test[self.filtered_features + [self.locker["target_name"]]],
                    img_height=28,
                    img_width=28,
                    augmentations=self.valid_aug,
                )

        elif self.locker["data_type"] == "image_folder":
            # folders of train test
            pass
            # use keras flow_from_directory don't use for now because it looks for subfolders with folder name as different targets like horses/humans

        elif self.locker["data_type"] == "tabular":
            # concept of useful feature don't make sense for image problem
            # self.xtrain = self.my_folds[self.useful_features]
            # self.xvalid = self.my_folds[self.useful_features]
            # self.yvalid = self.my_folds[self.locker["target_name"]]
            # self.ytrain = self.my_folds[self.locker["target_name"]]

            # self.xtest = self.test[self.useful_features]

            # del self.test
            # del self.my_folds
            # gc.collect()

            prep_dict = {
                "SiMe": SimpleImputer(strategy="mean"),
                "SiMd": SimpleImputer(strategy="median"),
                "SiMo": SimpleImputer(strategy="mode"),
                "Ro": RobustScaler(),
                "Sd": StandardScaler(),
                "Mi": MinMaxScaler(),
            }
            for f in self.prep_list:
                if f in list(prep_dict.keys()):
                    sc = prep_dict[f]
                    self.xtrain = sc.fit_transform(self.xtrain)
                    if self._state != "seed": # understand carefully it is correct
                        self.xvalid = sc.transform(self.xvalid)
                    if self._state != "opt":
                        self.xtest = sc.transform(self.xtest)

                elif f == "Lg":
                    self.xtrain = pd.DataFrame(
                        self.xtrain, columns=self.useful_features
                    )
                    self.xvalid = pd.DataFrame(
                        self.xvalid, columns=self.useful_features
                    )
                    self.xtest = pd.DataFrame(self.xtest, columns=self.useful_features)
                    # xtest = pd.DataFrame(xtest, columns=useful_features)
                    for col in self.useful_features:
                        self.xtrain[col] = np.log1p(self.xtrain[col])
                        self.xvalid[col] = np.log1p(self.xvalid[col])

                        self.xtest[col] = np.log1p(self.xtest[col])
                        # xtest[col] = np.log1p(xtest[col])
                        gc.collect()
                else:
                    raise Exception(f"scaler {f} is invalid!")
                gc.collect()

            # create instances
            if self.model_name.startswith("k") and self.comp_type != "2class":
                ## to one hot
                self.ytrain = np_utils.to_categorical(self.ytrain)
                self.yvalid = np_utils.to_categorical(self.yvalid)

        scores = []
        if self.locker["comp_type"] == "multi_label":
            final_test_predictions = [[], [], []]
        else:
            final_test_predictions = [[]]
        for i, rn in enumerate(random_list):
            print()
            print(f"Seed no: {i}, seed: {rn}")
            self._random_state = rn
            # run an algorithm for 100 times
            scores.append(self.obj("--no-trial--"))
            for i, f in enumerate(final_test_predictions):
                final_test_predictions[i].append(self.test_preds[i])
                gc.collect()
            gc.collect()
        
        # if regression problem then rank it
        if self.locker["comp_type"] in [
            "regression",
            "2class",
        ] and self.metrics_name in [
            "auc",
            "auc_tf",
        ]:  # auc takes rank s
            # no oof while creating seed submissions
            # temp_valid_prediction[f"pred_l_{self.current_dict['current_level']}_e_{self.exp_no}"] = [stats.rankdata(f) for f in temp_valid_prediction[f"pred_l_{self.current_dict['current_level']}_e_{self.exp_no}"]]
            # [[p1, p2, p3]]
            for i, f in enumerate(final_test_predictions):
                final_test_predictions[i] = [
                    stats.rankdata(f) for f in final_test_predictions[i]
                ]
                gc.collect()

        if self.locker["comp_type"] == "multi_label":
            # convert multi column target to single column
            # input=> 3 columns , output=> 1 column
            single_column = [np.array(f[0]) for f in final_test_predictions]
            multiple_columns = [
                stats.mode(np.column_stack(f), axis=1)[0]
                for f in final_test_predictions
            ]

            # keep sample to input
            # sample_real = pd.read_csv(f"../input/input-{self.locker['comp_name']}/sample.csv")
            sample_real = pd.read_parquet(
                f"../input/input-{self.locker['comp_name']}/sample.parquet"
            )
            sample_real["target"] = coln_3_1(single_column)
            # sample_real.to_csv(
            #     f"../configs/configs-{self.locker['comp_name']}/sub_seed_exp_{self.current_dict['current_exp_no']}_l_{self.current_dict['current_level']}_single.csv",
            #     index=False,
            # )
            sample_real.to_parquet(
                f"../working/{self.locker['comp_name']}_sub_e_{int(self.exp_no)}_single.parquet",
                index=False,
            )

            sample_real["target"] = coln_3_1(multiple_columns)
            # sample_real.to_csv(
            #     f"../configs/configs-{self.locker['comp_name']}/sub_seed_exp_{self.current_dict['current_exp_no']}_l_{self.current_dict['current_level']}_all.csv",
            #     index=False,
            # )
            sample_real.to_parquet(
                f"../working/{self.locker['comp_name']}_sub_e_{int(self.exp_no)}_all.parquet",
                index=False,
            )
        else:
            self.sample = pd.read_parquet(f"../input/input-{self.locker['comp_name']}/" + "sample.parquet")

            self.sample[self.locker["target_name"]] = [
                np.array(f[0]) for f in final_test_predictions
            ][0]
            # self.sample.to_csv(
            #     f"../configs/configs-{self.locker['comp_name']}/sub_seed_exp_{self.current_dict['current_exp_no']}_l_{self.current_dict['current_level']}_single.csv",
            #     index=False,
            # )
            self.sample.to_parquet(
                f"../working/{self.locker['comp_name']}_sub_e_{int(self.exp_no)}_single.parquet",
                index=False,
            )
            # mode is good for classification proble but not for regression problem
            if self.locker["comp_type"] in ["regression", "2class"]:
                # so we will use regression methods
                for i, f in enumerate(final_test_predictions):
                    final_test_predictions[i] = [
                        0.2 * f for f in final_test_predictions[i]
                    ]
                    gc.collect()
                self.sample[self.locker["target_name"]] = np.sum(
                    np.array(final_test_predictions[0]), axis=0
                )
            else:
                self.sample[self.locker["target_name"]] = [
                    stats.mode(np.column_stack(f), axis=1)[0]
                    for f in final_test_predictions
                ][0]

            # self.sample.to_csv(
            #     f"../configs/configs-{self.locker['comp_name']}/sub_seed_exp_{self.current_dict['current_exp_no']}_l_{self.current_dict['current_level']}_all.csv",
            #     index=False,
            # )
            self.sample.to_parquet(
                f"../working/{self.locker['comp_name']}_sub_e_{int(self.exp_no)}_all.parquet",
                index=False,
            )

        check_memory_usage("seed_it", self, 0)
        # ---- update table
        # self.Table.loc[self.Table.exp_no == self.exp_no, "fold_mean"] = np.mean(scores)
        # self.Table.loc[self.Table.exp_no == self.exp_no, "fold_std"] = np.std(scores)
        self.Table.loc[self.Table.exp_no == self.exp_no, "seed_mean"] = np.mean(scores)
        self.Table.loc[self.Table.exp_no == self.exp_no, "seed_std"] = np.std(scores)
        # pblb to be updated mannually
        # ---------------- dump table
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/Table.pkl", self.Table
        )

        gc.collect()
        check_memory_usage("run seed stop", self)
        tracemalloc.stop()

    def isRepetition(self, gen_features, old_features, feat_title):
        # self.curr
        for key, value in self.feat_dict.items():
            f1, f2, ft = value

            if f1 == gen_features and f2 == old_features and ft == feat_title:
                raise Exception("This feature is already present in my_folds!")

            gen_features_modified = gen_features
            old_features_modified = old_features
            f1_modified = f1
            f2_modified = f2

            if f2 == 0:
                # from base
                pass
            elif len(f1[0].split("_")[0]) < 5 or (
                f1[0].split("_")[0][0] == "l" and f1[0].split("_")[0][2] == "f"
            ):
                # originate from base so f2 can't be split
                f1_modified = ["_".join(f.split("_")[2:]) for f in f1_modified]
                gen_features_modified = [
                    "_".join(f.split("_")[2:]) for f in gen_features_modified
                ]
            else:
                f2_modified = ["_".join(f.split("_")[2:]) for f in f2_modified]
                old_features_modified = [
                    "_".join(f.split("_")[2:]) for f in old_features_modified
                ]
                f1_modified = ["_".join(f.split("_")[2:]) for f in f1_modified]
                gen_features_modified = [
                    "_".join(f.split("_")[2:]) for f in gen_features_modified
                ]
            if (
                f1_modified == gen_features_modified
                and f2_modified == old_features_modified
                and ft == feat_title
            ):
                raise Exception("This feature is already present in my_folds!")
            gc.collect()


if __name__ == "__main__":
    #s = seeds(exp_no=-1)  # last exp
    s = seeds(exp_no=44)
    
    s.run_seeds()
    del s
    # p = predictor(exp_no=3)  # exp_4
    # p.run_folds()
