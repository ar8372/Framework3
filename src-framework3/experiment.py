from optuna_search import OptunaOptimizer
from feature_generator import features
from feature_picker import Picker
import os
import sys
import gc
import pickle
import pandas as pd
import tracemalloc

# from custom_models import UModel
# from custom_models import *
from utils import *
from settings import *


class Agent:
    def __init__(
        self,
        useful_features=[],
        model_name="",
        comp_type="2class",
        metrics_name="accuracy",
        n_trials=5,
        prep_list=[],
        fold_name = 'fold5',
        optimize_on=0,
        save_models=True,
        with_gpu=False,
        aug_type="Aug1",
        _dataset="ImageDataset",
        use_cutmix=True,
        callbacks_list=[],
        note="---",
    ):
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
        self.current_dict = load_pickle(
            f"../configs/configs-{comp_name}/current_dict.pkl"
        )
        print("=" * 30)
        print(f"Current Exp no: {self.current_dict['current_exp_no']}")
        print("=" * 30)
        # ----------------------------------------------------------
        self.useful_features = useful_features
        self.model_name = model_name
        self.comp_type = comp_type
        self.metrics_name = metrics_name
        self.n_trials = n_trials
        self.prep_list = prep_list
        self.fold_name = fold_name 
        self.optimize_on = optimize_on
        self.save_models = True
        self.with_gpu = with_gpu
        self.aug_type = aug_type
        self._dataset = _dataset
        self.use_cutmix = use_cutmix
        self.callbacks_list = callbacks_list
        self.note = note

    def sanity_check(self):
        if "--|--" in [
            self.useful_features,
            self.model_name,
            self.comp_type,
            self.metrics_name,
            self.n_trials,
            self.prep_list,
            self.fold_name,
            self.optimize_on,
            self.save_models,
            self.with_gpu,
            self.aug_type,
            self._dataset,
            self.note,
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
        fold_name = "--|--",
        optimize_on="--|--",
        save_models="--|--",
        with_gpu="--|--",
        aug_type="--|--",
        _dataset="--|--",
    ):

        tracemalloc.start()
        check_memory_usage("Experiment started", self, 0)
        ######################################
        #         Memory uage                #
        ######################################

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
        if fold_name != "--|--":
            self.fold_name = fold_name
        if optimize_on != "--|--":
            self.optimize_on = optimize_on
        if save_models != "--|--":
            self.save_models = save_models
        if with_gpu != "--|--":
            self.with_gpu = with_gpu
        if aug_type != "--|--":
            self.aug_type = aug_type
        if _dataset != "--|--":
            self._dataset = _dataset

        self.sanity_check()

        # # my_folds = pd.read_csv(f"../configs/configs-{self.locker['comp_name']}/my_folds.csv")[self.useful_features + [self.locker["target_name"] , self.locker["id_name"], "fold"] ]
        # my_folds = pd.read_parquet(
        #     f"../input/input-{self.locker['comp_name']}/my_folds.parquet",
        #     columns=self.useful_features
        #     + [self.locker["target_name"], self.locker["id_name"], "fold"],
        # )  # [self.useful_features + [self.locker["target_name"] , self.locker["id_name"], "fold"] ]
        # # print(my_folds.head(2))
        # # taking only what is needed to reduce memory issue

        opt = OptunaOptimizer(
            model_name=self.model_name,
            comp_type=self.comp_type,
            metrics_name=self.metrics_name,
            n_trials=self.n_trials,
            prep_list=self.prep_list,
            fold_name = self.fold_name,
            optimize_on=self.optimize_on,
            with_gpu=self.with_gpu,
            save_models=self.save_models,
            aug_type=self.aug_type,
            _dataset=self._dataset,
            use_cutmix=self.use_cutmix,
            callbacks_list=self.callbacks_list,
        )

        print(f"Total no of trials: {self.n_trials}")
        self.study, random_state = opt.run( self.useful_features)

        del opt  # delete object
        gc.collect()

        if self.save_models == True:
            self._save_models(self.study, random_state)

        # Let's make perdiction on Test Set:
        # self._seed_it()

        check_memory_usage("run experiment Ends", self)
        tracemalloc.stop()

    def get_exp_no(self):
        # exp_no, current_level
        self.current_dict = load_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl"
        )
        self.current_exp_no = int(self.current_dict["current_exp_no"])

    def _save_models(self, study, random_state):
        Table = load_pickle(f"../configs/configs-{self.locker['comp_name']}/Table.pkl")
        Table = pd.DataFrame(Table)
        # what unifies it
        self.get_exp_no()
        # ExpNo- self.current_exp_no
        print("=" * 30)
        print(f"Current Exp no: {self.current_exp_no}")
        print("=" * 30)

        # # temp # whenever need to add new column
        # Table["callbacks_list"] = None
        # Table = Table[
        #     [
        #     "exp_no",
        #     "model_name",
        #     "bv",
        #     "bp",
        #     "random_state",
        #     "with_gpu",
        #     "aug_type",
        #     "_dataset",
        #     "use_cutmix",
        #     "callbacks_list",
        #     "features_list",
        #     "level_no",
        #     "oof_fold_name"
        #     "opt_fold_name"
        #     ###############"fold_no",
        #     "no_iterations",
        #     "prep_list",
        #     "metrics_name",
        #     "seed_mean",
        #     "seed_std",  # ---\
        #     "fold_mean", []
        #     "fold_std", []
        #     "pblb_single_seed",
        #     "pblb_all_seed",
        #     "pblb_all_fold", []
        #     "notes",
        #     ]
        # ]
        # print(Table.columns)
        # input("What you see:")
        # # temp
        # Rule 
        # initialize int/str feature with None and list with []
        Table.loc[Table.shape[0], :] = [
            self.current_exp_no,
            self.model_name,
            study.best_trial.value,
            study.best_trial.params,
            random_state,
            self.with_gpu,
            self.aug_type,
            self._dataset,
            self.use_cutmix,
            self.callbacks_list,
            self.useful_features,
            1, #self.current_dict["current_level"],
            [],  # oof on fold name
            self.fold_name, # opt on fold name
            self.optimize_on,
            self.n_trials,
            self.prep_list,
            self.metrics_name,
            None,
            None,
            [],
            [],
            None,
            None,
            [],
            self.note,
        ]

        self.current_exp_no += 1
        # --------------- dump experiment no
        self.current_dict["current_exp_no"] = self.current_exp_no
        save_pickle(
            f"../configs/configs-{self.locker['comp_name']}/current_dict.pkl",
            self.current_dict,
        )

        #---------------- sanity check table 
        Table.exp_no = Table.exp_no.astype(int)
        Table.level_no = Table.level_no.astype(int)
        Table.no_iterations = Table.no_iterations.astype(int)
        # ---------------- dump table
        save_pickle(f"../configs/configs-{self.locker['comp_name']}/Table.pkl", Table)

    def show_variables(self):
        print()
        for i, (k, v) in enumerate(self.__dict__.items()):
            print(f"{i}. {k} :=======>", v)
        print()

if __name__ == "__main__":
    # '100_165', 
    # ['100_165', '98_166', '93_168', '91_171', '90_172', '89_173', '88_182', '87_183', '85_185', '84_188', '83_195', '82_186', '78_188', '75_191', '74_194', '72_195', '71_196', '70_201', '69_205', '68_208', '66_209', '65_216', '64_220', '63_229', '62_235', '61_242', '60_244', '59_245', '58_248', '52_234', '51_235', '50_237', '49_238', '48_239', '47_241', '44_242', '43_247', '42_258', '40_263', '39_266', '38_268', '37_270', 'filter35_165', 'filter34_168', 'filter33_186', 'filter32_265', 'filter31_256', 'filter30_234', 'filter29_273', 'filter28_313', 'filter27_275', 'filter26_317', 'filter25_319', 'filter24_324', 'filter23_336', 'filter22_340', 'filter21_347', 'filter20_355', 'filter19_368', 'filter18_375', 'filter17_377', 'filter16_386', 'filter15_430', 'filter14_448', 'filter13_481', 'filter12_527', 'filter11_537', 'filter10_580', 'filter9_594', 'filter8_569', 'filter7_1364', 'filter6.1_43', 'filter6_37', 'filter5_38', 'filter4_45', 'filter2_49', 'filter1_54']:
    for jd in [ '93_168', '91_171',  '88_182', '83_195', '78_188', '70_201',  '64_220', '52_234','filter23_336','filter7_1364', 'filter6.1_43']:
    #for jd in ['100_165', '98_166', '93_168', '91_171', '90_172', '89_173', '88_182', '87_183', '85_185', '84_188', '83_195', '82_186', '78_188', '75_191', '74_194', '72_195', '71_196', '70_201', '69_205', '68_208', '66_209', '65_216', '64_220', '63_229', '62_235', '61_242', '60_244', '59_245', '58_248', '52_234', '51_235', '50_237', '49_238', '48_239', '47_241', '44_242', '43_247', '42_258', '40_263', '39_266', '38_268', '37_270', 'filter35_165', 'filter34_168', 'filter33_186', 'filter32_265', 'filter31_256', 'filter30_234', 'filter29_273', 'filter28_313', 'filter27_275', 'filter26_317', 'filter25_319', 'filter24_324', 'filter23_336', 'filter22_340', 'filter21_347', 'filter20_355', 'filter19_368', 'filter18_375', 'filter17_377', 'filter16_386', 'filter15_430', 'filter14_448', 'filter13_481', 'filter12_527', 'filter11_537', 'filter10_580', 'filter9_594', 'filter8_569', 'filter7_1364', 'filter6.1_43', 'filter6_37', 'filter5_38', 'filter4_45', 'filter2_49', 'filter1_54']:
        # ==========================================================

        
        useful_features = []
        # auto_features =['ver2_statistical']
        # for f in auto_features:
        #     useful_features += amzcomp1_settings().feature_dict[f]
        auto_features = [jd]
        for f in auto_features:
            useful_features += amzcomp1_settings().auto_filtered_features[f]
        useful_features = list(set(useful_features))
        # exp 20 features added
        #useful_features += ['Water_Supply_Once in two days', 'Dust_and_Noise_Medium', 'Property_Area', 'Dust_and_Noise_Low', 'Crime_Rate_Well above average', 'Property_Type_Bungalow', 'Traffic_Density_Score', 'Crime_Rate_Slightly below average', 'Number_of_Windows', 'Property_Type_Single-family home', 'Power_Backup_Yes', 'Air_Quality_Index', 'Frequency_of_Powercuts', 'Crime_Rate_Well below average', 'Neighborhood_Review', 'Property_Type_Apartment', 'Water_Supply_Once in a day - Evening', 'Furnishing_Semi_Furnished', 'Water_Supply_Once in a day - Morning', 'Property_Type_Container Home', 'Property_Type_Duplex', 'Water_Supply_NOT MENTIONED', 'Number_of_Doors', 'Power_Backup_No', 'Furnishing_Unfurnished']

        # ==========================================================
        model_name = "xgbr"  # -----s--->
        """
            [
                "lgr",
                "lir",
                "xgb",
                "xgbc",
                "xgbr",
                "cbc",
                "cbr",
                "mlpc", 
                "rg", 
                "ls",
                "knnc", 
                "dtc", 
                "adbc", 
                "gbmc" ,
                "gbmr,
                "hgbc", 
                "lgb", 
                "lgbmc", 
                "lgbmr", 
                "rfc" ,
                "rfr",
        # --------------->["tabnetr", "tabnetc", "k1", "k2", "k3", "tez1", "tez2", "p1" ,"pretrained"]
        """
        comp_type = (
            "2class"  # -------->["regression", "2class","multi_class", "multi_label"]
        )
        metrics_name = "amzcomp1_metrics"  # --------->["getaroom_metrics", "amex_metric","amex_metric_mod", "accuracy","f1","recall","precision", "auc", "logloss","auc_tf","mae","mse","rmse","msle","rmsle","r2"]
        n_trials = 100 #0 #30  # ------------> no of times to run optuna
        prep_list = [
            "Sd",
        ]  # ------> ["SiMe", "SiMd",~ "SiMo", "Mi", "Ro", "Sd", "Lg"] <= _prep_list
        prep_list = []
        fold_name = "fold5"  # ['fold3', 'fold5', 'fold10', 'fold20']
        optimize_on = [random.choice(range(5))] # [0]  # fold on which optimize # 0,1,2,4
        with_gpu = True

        aug_type = "aug2"  # "aug1", "aug2", "aug3", "aug4"
        _dataset = "DigitRecognizerDataset"  # "BengaliDataset", "ImageDataset", "DigitRecognizerDataset", "DigitRecognizerDatasetTez2"
        use_cutmix = False

        # CALLBACKS
        # lgbmClassifiers callback: 
        # https://lightgbm.readthedocs.io/en/latest/Python-API.html#callbacks
        """
        ############################ change learning rate
        custom schedulers
        :=>cosine_decay , exponential_decay, simple_decay 
        built in scheduler
        :=>"ReduceLROnPlateau" # EarlyStopping, 

        ############################# save model
        :=>chk_pt : whether to use checkpoint or not

        ############################### stop training
        :=>terminate_on_NaN 
        >Callbacks monitor a particular varaibable and stops exectuion when it crosses fixed value , 
        >early_top also monitors a particular value but it stops when it stops improving it has some patience 
        builtin callback
        :=>early_stopping 
        custom callback 
        :=>myCallback1
        """
        # don't use early_stopping with cyclic decay lr because model gets good and bad periodically and it doesn't mean we should terminate.
        # "swa", "cosine_decay", "exponential_decay", "simple_decay", "ReduceLROnPlateau", "chk_pt", "terminate_on_NaN", "early_stopping", "myCallback1"
        callbacks_list = ["terminate_on_NaN"] # ["exponential_decay", "terminate_on_NaN"] # [swa,early_stop]
        # -----------------------------------------------------------
        note = "ragnar"
        e = Agent(
            useful_features=useful_features,
            model_name=model_name,
            comp_type=comp_type,
            metrics_name=metrics_name,
            n_trials=n_trials,
            prep_list=prep_list,
            fold_name = fold_name,
            optimize_on=optimize_on,
            with_gpu=with_gpu,
            aug_type=aug_type,
            _dataset=_dataset,
            use_cutmix=use_cutmix,
            callbacks_list=callbacks_list,
            note=note,
        )
        print("=" * 40)
        print("Useful_features:", useful_features)

        e.run()
        del e
        # -------------------------------------------------------------
        # exp_list = ["1"]  # ----------------> [1,2,3,4]
        # e.show(exp_list)

        """
        {'learning_rate': 0.010821262164314453, 'max_depth': 16, 'min_child_weight': 5, 'subsample': 0.4521783648128741, 'n_estimators': 500, 'objective': 'reg:squarederror', 'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': 'gpu_predictor'}
        
        """

        # Make prediction also 
        os.system(f"python predict.py")


        #break
