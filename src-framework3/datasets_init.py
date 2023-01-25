import os
import sys
import json


def initialize_folders():
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    # -------------------CREATE-FOLDERS----------------------------------#
    # --------------------configs----------------------------------------#
    try:
        os.mkdir(f"../configs/configs-{comp_name}/")
        os.mkdir(f"../configs/configs-{comp_name}/logs/")
        os.mkdir(f"../configs/configs-{comp_name}/oof_preds/")
        os.mkdir(f"../configs/configs-{comp_name}/test_preds/")
        os.mkdir(f"../configs/configs-{comp_name}/test_feats/")
        os.mkdir(f"../configs/configs-{comp_name}/train_feats/")
        os.mkdir(f"../configs/configs-{comp_name}/ensemble_logs/")
        os.mkdir(f"../configs/configs-{comp_name}/feature_importance/")
        os.mkdir(f"../configs/configs-{comp_name}/auto_exp_tables/")
        
        print(f"configs-{comp_name} folder and subfolders logs/oof_preds/test_preds created.")
    except:
        print(f"configs-{comp_name} and subfolders logs/oof_preds/test_preds already exists.")
    # --------------------models----------------------------------------#
    try:
        os.mkdir(f"../models/models-{comp_name}/")
        print(f"models-{comp_name} folder created.")
    except:
        print(f"models-{comp_name} already exists.")
    # --------------------input----------------------------------------#
    try:
        os.mkdir(f"../input/input-{comp_name}/")
        print(f"input-{comp_name} folder created.")
    except:
        print(f"input-{comp_name} already exists.")
    # -------------------ADD-META DATA----------------------------------#
    # --------------------configs----------------------------------------#
    print("=" * 40)
    # --------------------meta data create
    os.system(f"kaggle datasets init -p ../configs/{'configs-'+ comp_name}/")
    os.system(f"kaggle datasets init -p ../models/{'models-'+ comp_name}/")
    os.system(f"kaggle datasets init -p ../input/{'input-'+ comp_name}/")

    # read the json file
    with open(f"../configs/configs-{comp_name}/dataset-metadata.json") as f:
        dataset_meta = json.load(f)

    try:
        dataset_meta["id"] = f"raj401/configs-{comp_name}"
        dataset_meta["title"] = f"configs-{comp_name}"
        with open(
            f"../configs/configs-{comp_name}/dataset-metadata.json", "w"
        ) as outfile:
            json.dump(dataset_meta, outfile)
        print(outfile)
        with open(f"../configs/configs-{comp_name}/jj.txt", "w") as x:
            x.write("demo")
        x.close()
        print(
            f"configs-{comp_name} folder meta-data added."
        )
    except:
        print(f"configs-{comp_name} meta-data already exists.")
    # --------------------models----------------------------------------#
    print("=" * 40)
    try:
        dataset_meta["id"] = f"raj401/models-{comp_name}"
        dataset_meta["title"] = f"models-{comp_name}"
        with open(
            f"../models/models-{comp_name}/dataset-metadata.json", "w"
        ) as outfile:
            json.dump(dataset_meta, outfile)
        print(outfile)
        with open(f"../models/models-{comp_name}/jj.txt", "w") as x:
            x.write("demo")
        x.close()
        print(f"models-{comp_name} folder meta-data added.")
    except:
        print(f"models-{comp_name} meta-data already exists.")
    # --------------------input----------------------------------------#
    print("=" * 40)
    try:
        dataset_meta["id"] = f"raj401/input-{comp_name}"
        dataset_meta["title"] = f"input-{comp_name}"
        with open(f"../input/input-{comp_name}/dataset-metadata.json", "w") as outfile:
            json.dump(dataset_meta, outfile)
        print(outfile)
        with open(f"../input/input-{comp_name}/jj.txt", "w") as x:
            x.write("demo")
        x.close()
        print(f"input-{comp_name} folder meta-data added.")
    except:
        print(f"input-{comp_name} meta-data already exists.")
    print("=" * 40)


if __name__ == "__main__":
    #[ CAN CALL IT MULTIPLE TIMES, no harm will be done in calling]
    # each time it will reset .json file
    """
    It Creates 3 folders: configs-{}, models-{}, input-{}
    It then initializes their metadata
    It then modifies their meta data accordingly sets the name and rug
    It then puts a demo txt file jj
    """
    initialize_folders()
    print("Done")
