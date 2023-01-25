import os
import sys
import pandas as pd
import json


def sync_(configs, models, src, input_):
    framework_name = "framework3"
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    print("comp name:", comp_name)

    # download data
    if configs:
        # configs
        #!kaggle datasets list
        try:
            os.mkdir(f"../configs/configs-{comp_name}/")
            print(f"configs-{comp_name} folder created.")
        except:
            print(f"configs-{comp_name} already exists.")

        os.system(
            f"kaggle datasets download raj401/configs-{comp_name} -p ../configs/configs-{comp_name}/ --unzip"
        )
        print(f"configs-{comp_name} downloaded")

    if models:
        try:
            os.mkdir(f"../models/models-{comp_name}/")
            print(f"models-{comp_name} folder created.")
        except:
            print(f"models-{comp_name} already exists.")
        # models
        #!kaggle datasets download raj401/models-tmay -p ../models/models-tmay/ --unzip
        os.system(
            f"kaggle datasets download raj401/models-{comp_name} -p ../models/models-{comp_name}/ --unzip"
        )
        print(f"models-{comp_name} downloaded")

    if src:
        try:
            os.mkdir(f"../src-{framework_name}/")
            print(f"src-{framework_name} folder created.")
        except:
            print(f"src-{framework_name} already exists.")
        # src
        # !kaggle datasets download raj401/src-{framework_name} -p ../src-{framework_name}/ --unzip
        os.system(
            f"kaggle datasets download raj401/src-{framework_name} -p ../src-{framework_name}/ --unzip"
        )
        print(f"src-{framework_name} downloaded")

    if input_:
        try:
            os.mkdir(f"../input/input-{comp_name}/")
            print(f"input-{comp_name} folder created.")
        except:
            print(f"input-{comp_name} already exists.")
        # input
        #!kaggle datasets download raj401/input-tmay -p ../input/input-tmay/ --unzip
        os.system(
            f"kaggle datasets download raj401/input-{comp_name} -p ../input/input-{comp_name}/ --unzip"
        )
        print(f"input-{comp_name} downloaded")


if __name__ == "__main__":
    configs = True  # configs
    models = False  # models
    src = False  # src_framework3
    input_ = False

    sync_(configs, models, src, input_)
    print("Done")
