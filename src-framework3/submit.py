import pandas as pd 
import numpy as np 
import os
import sys 
import pickle 
from utils import * 
from output import * 
from predict import * 
from seed_it import * 
import time

# for animation
import itertools
import threading
import time
import sys

import global_variables

import warnings
warnings.filterwarnings("ignore")

def update_it( comp_name, exp_no, sub_type, pb_score):
    # to update a table that is all we need
    # comp_name , exp_no, sub_table 
    
    Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")
    updated = False
    col_name = {
        "all" : "pblb_all_seed",
        "single" : "pblb_single_seed"
    }
    if sub_type.startswith("e"):
        # ensemble
        raise Exception("Ensemble should not try to update Table!!")
    elif sub_type.startswith("f"):
        # public fold submission 
        # so first find index of fold
        full_row = Table.loc[Table.exp_no == exp_no]
        try:
            oof_fold_name_list = Table.loc[Table.exp_no == exp_no, "oof_fold_name"].values[0]
            # sanity check
            if oof_fold_name_list == []:
                # our oof_fold_name is empty : No prediction is done yet  
                # that is not possible because we entered here only after some prediction has been done as "fold5/fold10/..."
                raise Exception("Already made prediction but Table is showing oof_fold_name empty!!")
            index_no= Table.loc[Table.exp_no == exp_no, "oof_fold_name"].values[0].index(sub_type)
            pblb_all_fold_list = Table.loc[Table.exp_no == exp_no, "pblb_all_fold"].values[0]
            # first correct it 
            if pblb_all_fold_list == []:
                # it is first time 
                pblb_all_fold_list = [None for i in range(len(oof_fold_name_list))]
            if pblb_all_fold_list[index_no] is  None:
                # so this is a new submission so update 
                
                print("Before updating")
                print(Table.loc[Table.exp_no == exp_no, ["exp_no","oof_fold_name", "pblb_all_fold"]])
                pblb_all_fold_list[index_no] = pb_score
                full_row.pblb_all_fold = [[i for i in pblb_all_fold_list]]
                
                print("updating...")
                Table.loc[Table.exp_no == exp_no, :] = full_row.copy()
                print("After updating")
                print(Table.loc[Table.exp_no == exp_no, ["exp_no","oof_fold_name", "pblb_all_fold"]])
                print("="*40)
                print()
                updated = True

            else:
                # already updated
                print("Already updated")
                print()
                pass
        except:
            raise Exception(f"Fold name {sub_type} is not found in Table") 


    elif sub_type in ["single", "all"]:
        # all seeds
        if Table.loc[Table.exp_no == exp_no, col_name[sub_type]].values[0] is None:
            # first time
            print("Before updating")
            print(Table.loc[Table.exp_no == exp_no, ["exp_no","pblb_single_seed", "pblb_all_seed"]])
            print("updating...")
            Table.loc[Table.exp_no == exp_no, col_name[sub_type]] = pb_score
            print("After updating")
            print(Table.loc[Table.exp_no == exp_no, ["exp_no","pblb_single_seed", "pblb_all_seed"]])
            print("="*40)
            print()
            updated = True
        else:
            # no need to rename it already updated
            print("Already updated")
            print()
            pass
    else:
        raise Exception(f"sub_type {sub_type} is not valid name!")
    
    if updated:
        # since Table is updated 
        save_pickle(f"../configs/configs-{comp_name}/Table.pkl", Table)





def update_table_with_submission_log(comp_full_name, current_update_only = False):
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    comp_full_name = "amex-default-prediction"   
    log = pd.read_csv(f"../working/{comp_full_name}_submission_logs.csv") 
    print(log)
    log = log[log.fileName == "submission.csv"] # ignore old 
    if current_update_only is True:
        # it is asking to update only the current submission no need to go through the whole logs 
        log = log.iloc[[0]]
        print("Only update single row!!")
        print()
        print(log)
        print()
    
    for row in log.iterrows():
        items = row[1]
        if items[3] != "complete":
            # score is not updated yet
            continue
        pb_score = float(items[4])
        # items[2] :=> message 
        #  f"{comp_name}_exp_{exp_no}_{pred_type}" 
        #  f"{comp_name}_ens_{exp_no}" 
        comp_name = items[2].split("_")[0]
        marker = items[2].split("_")[1]
        exp_no = int(items[2].split("_")[2])
        if marker != "exp":
            # ensemble case 
            update_log_ens(comp_name, exp_no, pb_score)
        else:
            sub_type = items[2].split("_")[3] # fold5, all, single
            print(comp_name, marker, exp_no, sub_type, pb_score)
            update_it( comp_name, exp_no, sub_type, pb_score)

def update_log_ens(comp_name, exp_no, pb_score):
    # update the public score
    ensemble_dict = load_json(f"../configs/configs-{comp_name}/ensemble_logs/ens_{exp_no}.json")
    ensemble_dict["pb_score"] = pb_score 
    print(ensemble_dict)
    save_json(f"../configs/configs-{comp_name}/ensemble_logs/ens_{exp_no}.json", ensemble_dict)


"""
submit.py 
    kaggle competitions list
    kaggle competitions leaderboard amex-default-prediction --show | --download
    kaggle competitions submissions amex-default-prediction 
    kaggle competitions submit ventilator-pressure-prediction -f submission.csv -m "exp_{}_fold/single/all" #submit your submission.csv
    kaggle competitions submit -c [COMPETITION] -f [FILE] -m [MESSAGE]
"""
def submit_pred(comp_full_name, exp_no, pred_type, mode):
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    locker = load_pickle(f"../configs/configs-{comp_name}/locker.pkl")
    Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")
    if exp_no == -1:
        row_e = Table[Table.exp_no == list(Table.exp_no.values)[-1]]
        exp_no = row_e.exp_no.values[0]
    if pred_type == "ens":
        # pred type is good for exp as can be multiple fold but here each ensemble is unique so keep minimalistic name
        message = f"{comp_name}_ens_{exp_no}" 
    else:
        message = f"{comp_name}_exp_{exp_no}_{pred_type}" 
    ##########################################
    #              Sanity check              # 
    ##########################################
    # checks whether it is already submitted 
    log = pd.read_csv(f"../working/{comp_full_name}_submission_logs.csv")
    log = log[log.fileName == "submission.csv"] # ignore old 
    if message in list(log.description.values):
        raise Exception(f"{message} already submitted!!!")
    #########################################
    
    if pred_type == "ens":
        # it's an ensemble prediction 
        try:
            sample = pd.read_parquet(f"../working/{comp_name}_ens_{exp_no}.parquet")
        except:
            raise Exception(f"Not created ensemble of ens no: {exp_no}")
        sample.to_csv(f"../working/submission.csv",index=False)
        print(sample.head(3))
        if mode == "manual":
            val = input(f"""You are about to make submission to {comp_name} with: \nFile named: {comp_name}_sub_e_{exp_no}_single.parquet
            Press Y/y to continue or press N/n to prevent submission!: """)
            if val.upper() == "Y":
                os.system(f"kaggle competitions submit {comp_full_name} -f ../working/submission.csv -m {message}") # #submit your submission.csv")
        else:
            time.sleep(1)
            os.system(f"kaggle competitions submit {comp_full_name} -f ../working/submission.csv -m {message}") # #submit your submission.csv")

    if pred_type == "single":
        try:
            sample= pd.read_parquet(f"../working/{comp_name}_sub_e_{exp_no}_single.parquet")
        except:
            # It is not predicted yet
            # run seed_it script 
            s = seeds(exp_no=exp_no)  # last exp
            s.run_seeds()
            del s
            # It must be true now
            sample= pd.read_parquet(f"../working/{comp_name}_sub_e_{exp_no}_single.parquet")


        sample.to_csv(f"../working/submission.csv",index=False) # {comp_name}_sub_e_{exp_no}_single.csv") # we 
        print(sample.head(3))
        if mode == "manual":
            val = input(f"""You are about to make submission to {comp_name} with: \nFile named: {comp_name}_sub_e_{exp_no}_single.parquet
            Press Y/y to continue or press N/n to prevent submission!: """)
            if val.upper() == "Y":
                os.system(f"kaggle competitions submit {comp_full_name} -f ../working/submission.csv -m {message}") # #submit your submission.csv")
        else:
            time.sleep(1)
            os.system(f"kaggle competitions submit {comp_full_name} -f ../working/submission.csv -m {message}") # #submit your submission.csv")

    if pred_type == "all":
        try:
            sample= pd.read_parquet(f"../working/{comp_name}_sub_e_{exp_no}_all.parquet")
        except:
            # It is not predicted yet
            # run seed_it script 
            s = seeds(exp_no=exp_no)  # last exp
            s.run_seeds()
            del s
            # It must be true now
            sample= pd.read_parquet(f"../working/{comp_name}_sub_e_{exp_no}_all.parquet")

        sample.to_csv(f"../working/submission.csv",index=False) # {comp_name}_sub_e_{exp_no}_all.csv") # we 
        print(sample.head(3))
        if mode == "manual":
            val = input(f"""You are about to make submission to {comp_name} with: \nFile named: {comp_name}_sub_e_{exp_no}_all.parquet
            Press Y/y to continue or press N/n to prevent submission!: """)
            if val.upper() == "Y":
                os.system(f"kaggle competitions submit {comp_full_name} -f ../working/submission.csv -m {message}") # #submit your submission.csv")
        else:
            time.sleep(1)
            os.system(f"kaggle competitions submit {comp_full_name} -f ../working/submission.csv -m {message}") # #submit your submission.csv")

    if pred_type.startswith("f"): #"fold3", "fold5"
        
        try:
            sample= pd.read_parquet(f"../working/{comp_name}_sub_e_{exp_no}_{pred_type}.parquet")
        except:
            # It is not output yet yet
            try:
                dummy = load_pickle(f"../configs/configs-{comp_name}/test_preds/test_pred_e_{exp_no}_{pred_type}.pkl")
                del dummy 
                gc.collect()
            except:
                # It is also not predicted yet
                # predict it 
                p = predictor(exp_no=exp_no)  # last exp
                fold_name = pred_type #fold5" #"fold5" # "fold3" , "fold5", "fold10" , "fold20", ""
                p.run_folds(fold_name)
                del p

            # output it
            # need to add which fold prediction to output
            file_type = "parquet"  # "csv"
            o = out( pred_type , exp_no, file_type)
            o.dump()
            del o 

        # It must be true now
        sample= pd.read_parquet(f"../working/{comp_name}_sub_e_{exp_no}_{pred_type}.parquet")

        sample.to_csv(f"../working/submission.csv",index=False) # {comp_name}_sub_e_{exp_no}_ingle.csv") # we 
        print(sample.head(3))
        if mode == "manual":
            val = input(f"""You are about to make submission to {comp_name} with: \nFile named: {comp_name}_sub_e_{exp_no}_{pred_type}.parquet
            Press Y/y to continue or press N/n to prevent submission!: """)
            if val.upper() == "Y":
                score= os.system(f"kaggle competitions submit {comp_full_name} -f ../working/submission.csv -m {message}") # #submit your submission.csv")
        else:
            time.sleep(1)
            os.system(f"kaggle competitions submit {comp_full_name} -f ../working/submission.csv -m {message}") # #submit your submission.csv")


    ######################################################
    #          Makes sure submission score is out        #
    ######################################################
    global_variables.done = False
    t = threading.Thread(target=animate)
    t.start()
    #long process here
    time.sleep(5)
    # save logs
    # update submission_log
    # https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file#:~:text=the%20shortcut%20is%20Ctrl%20%2B%20Shift,or%20as%20HTML%20including%20colors!
    #!kaggle competitions submissions amex-default-prediction  --csv > ../working/amex-default-prediction_submission_logs.csv
    os.system(f"kaggle competitions submissions {comp_full_name}  --csv > ../working/{comp_full_name}_submission_logs.csv") 
    log =  pd.read_csv(f"../working/{comp_full_name}_submission_logs.csv")
    latest_message = log.loc[0, 'description']
    status = log.loc[0, "status"]
    while latest_message != message or status !=  "complete":
        time.sleep(3)
        os.system(f"kaggle competitions submissions {comp_full_name}  --csv > ../working/{comp_full_name}_submission_logs.csv") 
        log = pd.read_csv(f"../working/{comp_full_name}_submission_logs.csv")
        latest_message = log.loc[0, 'description']
        status = log.loc[0, "status"]
    global_variables.done = True    
    # display submission_log
    os.system(f"kaggle competitions submissions {comp_full_name}") 
    # update the table
    #update_table_with_submission_log(comp_full_name)
    current_update_only = True
    update_table_with_submission_log(comp_full_name, current_update_only)

if __name__ == "__main__":
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()
    comp_full_name = "amex-default-prediction"
    #exp_no =  -1  #0,1,2 
    exp_no = 44
    pred_type = "single" #"fold10" #"fold5" # "all" "fold" "fold3", "ens"
      
    mode = "auto" # "manual"
    # manual asks for a prompt just before submitting
    # auto don't asks for a prompt 
    # both can make prediction file if already not created

    submit_pred(comp_full_name, exp_no, pred_type, mode) 
    

    # if calling from here update, itterate through whole log file 
    current_update_only = False
    #update_table_with_submission_log(comp_full_name, current_update_only)

    #e138
    #[0.7984387250317737, 0.794395022818941, 0.7928167917014163, 0.797858209410043, 0.7950284894125992]
    #
    # [0.7978218889227393, 0.7940426440210139, 0.7932858172582967, 0.7982026580647962, 0.7945527414758347] --> 0.97
