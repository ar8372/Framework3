import os
import sys
import json


def create_datasets():
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    # -------------------CREATE-FOLDERS----------------------------------#


    try:
        os.system(f"kaggle datasets create -p ../configs/{'configs-'+ comp_name}/")
        print(
            f"configs-{comp_name} dataset created on kaggle."
        )
    except:
        print(f"configs-{comp_name} dataset already created on kaggle.")
    # --------------------models----------------------------------------#
    print("=" * 40)
    try:
        os.system(f"kaggle datasets create -p ../models/{'models-'+ comp_name}/")
        print(f"models-{comp_name} dataset created on kaggle.")
    except:
        print(f"models-{comp_name} dataset already created on kaggle.")
    # --------------------input----------------------------------------#
    print("=" * 40)
    try:
        os.system(f"kaggle datasets create -p ../input/{'input-'+ comp_name}/")
        print(f"input-{comp_name} dataset created on kaggle.")
    except:
        print(f"input-{comp_name} dataset already created on kaggle.")
    print("=" * 40)


if __name__ == "__main__":
    # [ IF RUN 2nd time will throw error]
    # CALL IT ONLY ONCE after init_folders.py from next time just call push.py 

    create_datasets()
    print("Done")
