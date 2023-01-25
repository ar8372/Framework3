import pandas as pd 
import numpy as np 
from auto_exp import * 
from collections import defaultdict

def split_base():
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")

    target_name = locker['target_name']
    id_name = locker['id_name']
    fold_list = ['fold3', 'fold5', 'fold10', 'fold20']

    my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
    print("my_folds shape is:", my_folds.shape)
    test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")    


    
    all_columns = list(test.drop(id_name, axis=1).columns)

    my_dict = defaultdict()
    demo_set = set(all_columns)
    # step 1
    feat = [id_name] + fold_list + [target_name]
    d  = my_folds[feat]
    d.to_parquet(f"../input/input-{comp_name}/id_folds_target.parquet")
    my_dict['id_folds_target'] = feat

    # step 2
    feat = all_columns
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_original.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_original.parquet", index=False)
    my_dict['original'] = feat
    demo_set = demo_set - set(feat)

    assert demo_set == set()

    save_pickle(f"../input/input-{comp_name}/input_dict.pkl", my_dict)


def split():
    comp_name = "amexdummy"
    target_name = "prediction"
    id_name = "customer_ID"
    fold_list = ["fold3", "fold5", "fold10", "fold20"]

    my_folds = pd.read_parquet(f"../input/input-{comp_name}/my_folds.parquet")
    print("my_folds shape is:", my_folds.shape)
    test = pd.read_parquet(f"../input/input-{comp_name}/test.parquet")
    #print(my_folds.columns)
    # cat_features = [
    #     "B_30",
    #     "B_38",
    #     "D_114",
    #     "D_116",
    #     "D_117",
    #     "D_120",
    #     "D_126",
    #     "D_63",
    #     "D_64",
    #     "D_66",
    #     "D_68"
    # ]
    # num_columns = ['P_2', 'D_39', 'B_1', 'B_2', 'R_1', 'S_3', 'D_41', 'B_3', 'D_42', 'D_43', 'D_44', 'B_4', 'D_45', 'B_5', 'R_2', 'D_46', 'D_47', 'D_48', 'D_49', 'B_6', 'B_7', 'B_8', 'D_50', 'D_51', 'B_9', 'R_3', 'D_52', 'P_3', 'B_10', 'D_53', 'S_5', 'B_11', 'S_6', 'D_54', 'R_4', 'S_7', 'B_12', 'S_8', 'D_55', 'D_56', 'B_13', 'R_5', 'D_58', 'S_9', 'B_14', 'D_59', 'D_60', 'D_61', 'B_15', 'S_11', 'D_62', 'D_65', 'B_16', 'B_17', 'B_18', 'B_19', 'B_20', 'S_12', 'R_6', 'S_13', 'B_21', 'D_69', 'B_22', 'D_70', 'D_71', 'D_72', 'S_15', 'B_23', 'D_73', 'P_4', 'D_74', 'D_75', 'D_76', 'B_24', 'R_7', 'D_77', 'B_25', 'B_26', 'D_78', 'D_79', 'R_8', 'R_9', 'S_16', 'D_80', 'R_10', 'R_11', 'B_27', 'D_81', 'D_82', 'S_17', 'R_12', 'B_28', 'R_13', 'D_83', 'R_14', 'R_15', 'D_84', 'R_16', 'B_29', 'S_18', 'D_86', 'D_87', 'R_17', 'R_18', 'D_88', 'B_31', 'S_19', 'R_19', 'B_32', 'S_20', 'R_20', 'R_21', 'B_33', 'D_89', 'R_22', 'R_23', 'D_91', 'D_92', 'D_93', 'D_94', 'R_24', 'R_25', 'D_96', 'S_22', 'S_23', 'S_24', 'S_25', 'S_26', 'D_102', 'D_103', 'D_104', 'D_105', 'D_106', 'D_107', 'B_36', 'B_37', 'R_26', 'R_27', 'D_108', 'D_109', 'D_110', 'D_111', 'B_39', 'D_112', 'B_40', 'S_27', 'D_113', 'D_115', 'D_118', 'D_119', 'D_121', 'D_122', 'D_123', 'D_124', 'D_125', 'D_127', 'D_128', 'D_129', 'B_41', 'B_42', 'D_130', 'D_131', 'D_132', 'D_133', 'R_28', 'D_134', 'D_135', 'D_136', 'D_137', 'D_138', 'D_139', 'D_140', 'D_141', 'D_142', 'D_143', 'D_144', 'D_145']
    
    # # 11 + 177 = 188
    all_columns = list(test.drop('customer_ID', axis=1).columns)
    # #base_columns =[i[:-6] for i in  list(filter(lambda x:x.endswith("_first"), all_columns))]
    # base_columns = cat_features + num_columns
    # #num_columns = [c for c in base_columns if c not in cat_features]

    # print(base_columns)
    # print(cat_features)
    # print(num_columns)

    # print(len(cat_features))
    # print(len(num_columns))
    # print(len(base_columns))

    demo_set = set(all_columns)
    print("all columns", len(all_columns), len(list(demo_set)))
    #print(all_columns)
    print()
    my_dict = defaultdict()
    
    # fold+target : id_fold_target
    feat = [id_name] + fold_list + [target_name]
    d  = my_folds[feat]
    d.to_parquet(f"../input/input-{comp_name}/id_folds_target.parquet")
    my_dict['id_folds_target'] = feat
    # print(d.head(2))
    # print(d.shape)
    # first : train_first, test_first
    feat = ['year_first', 'year_mean', 'year_std', 'year_min', 'year_max', 'year_last', 'year_nunique', 'year_count', 'month_first', 'month_mean', 'month_std', 'month_min', 'month_max', 'month_last', 'month_nunique', 'month_count', 'day_first', 'day_mean', 'day_std', 'day_min', 'day_max', 'day_last', 'day_nunique', 'day_count', 'dayofweek_first', 'dayofweek_mean', 'dayofweek_std', 'dayofweek_min', 'dayofweek_max', 'dayofweek_last', 'dayofweek_nunique', 'dayofweek_count']
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_date.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_date.parquet", index=False)
    my_dict['date'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)

    """
    feat = [i+"_first" for i in base_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_first.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_first.parquet", index=False)
    my_dict['first'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    
    # last : train_last, test_last
    feat = [i+"_last" for i in base_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_last.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_last.parquet", index=False)
    my_dict['last'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # min
    feat = [i+"_min" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[[i+"_min" for i in num_columns]]
    d1.to_parquet(f"../input/input-{comp_name}/train_min.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_min.parquet", index=False)
    my_dict['min'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # max
    feat = [i+"_max" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_max.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_max.parquet", index=False)
    my_dict['max'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # mean
    feat = [i+"_mean" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_mean.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_mean.parquet", index=False)
    my_dict['mean'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # std
    feat = [i+"_std" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_std.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_std.parquet", index=False)
    my_dict['std'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # count
    feat = [i+"_count" for i in cat_features]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_count.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_count.parquet", index=False)
    my_dict['count'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # nunique
    feat = [i+"_nunique" for i in cat_features]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_nunique.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_nunique.parquet", index=False)
    my_dict['nunique'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # lag_sq
    feat = [i+"_last_lag_sq" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_lag_sq.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_lag_sq.parquet", index=False)
    my_dict['lag_sq'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # lag_cb
    feat = [i+"_last_lag_cb" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_lag_cb.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_lag_cb.parquet", index=False)
    my_dict['lag_cb'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # lag_div
    feat = [i+"_last_lag_div" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_lag_div.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_lag_div.parquet", index=False)
    my_dict['lag_div'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # lag_sub
    feat = [i+"_last_lag_sub" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_lag_sub.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_lag_sub.parquet", index=False)
    my_dict['lag_sub'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)

    # round2 Round lall ast float features to 2 decimal place
    # This is the one which keeps changing
    feat = list(filter(lambda x:x.endswith("_2round2"), all_columns))
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_2round2.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_2round2.parquet", index=False)
    my_dict['2round2'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)

    # lag_mmsub max-min
    feat = [i+"_max_lag_mmsub" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_lag_mmsub.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_lag_mmsub.parquet", index=False)
    my_dict['lag_mmsub'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # lag_mmsub max-min
    feat = [i+"_max_lag_mmsq" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_lag_mmsq.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_lag_mmsq.parquet", index=False)
    my_dict['lag_mmsq'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # lag_mmcb max-min
    feat = [i+"_max_lag_mmcb" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_lag_mmcb.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_lag_mmcb.parquet", index=False)
    my_dict['lag_mmcb'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # last_mean_diff
    feat = [i+"_last_mean_diff" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_last_mean_diff.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_last_mean_diff.parquet", index=False)
    my_dict['last_mean_diff'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)
    # _diff1
    feat = [i+"_diff1" for i in num_columns]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_diff1.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_diff1.parquet", index=False)
    my_dict['diff1'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)

    # count_nunique_diff
    feat = [i+"_count_nunique_diff" for i in cat_features]
    d1 = my_folds[feat]
    d2 = test[feat]
    d1.to_parquet(f"../input/input-{comp_name}/train_count_nunique_diff.parquet", index=False)
    d2.to_parquet(f"../input/input-{comp_name}/test_count_nunique_diff.parquet", index=False)
    my_dict['count_nunique_diff'] = feat
    print( len(feat))
    #print(feat)
    print()
    demo_set = demo_set - set(feat)

    print("Done")
    print(demo_set)
    print(len(list(demo_set)))
    """
    save_pickle(f"../input/input-{comp_name}/input_dict.pkl", my_dict)
    

#split_base()
#split()
#print(11*3+177*13+188*2+642)
d = load_pickle(f"../input/input-amzcomp1/input_dict.pkl")
print(d)
# 2633
# s = [188,
# 188,
# 177,
# 177,
# 177,
# 177,
# 11,
# 11,
# 177,
# 177,
# 177,
# 177,
# 642,
# 177]
# print(s)
# print(sum(s))
# verified
# amex = amex3_settings()
# feature_keys = ['first','last','min', 'max' , 'mean', 'std', 'count', 'nunique', 'lag_sq', 'lag_cb','lag_div', 'lag_sub', 'round2', 'lag_mmsub']
# feature_keys2 = ['all_cat', 'all_num']
# all_cols = []
# for f in feature_keys:
#     all_cols += amex.feature_dict[f]



# print(len(all_cols))
# assert len(all_cols) == len(set(all_cols))

# all_cols2 = []
# for r in feature_keys2:
#     all_cols2 += amex.feature_dict[r]
# print(len(all_cols2))
# assert len(all_cols2) == len(set(all_cols2))