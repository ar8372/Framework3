import os
import sys
import pandas as pd
import json
from datetime import datetime


def sync_(configs, models, src, working, input_, comment=""):
    framework_name = "framework3"
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    print("comp name:", comp_name)

    version_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if comment != "":
        version_name += "-" + comment
    print(f"Versioning at {version_name}")
    print("=" * 40)

    # upload data
    if configs:
        # configs
        #!kaggle datasets list
        try:
            os.system(
                f"kaggle datasets version -m {version_name} -p ../configs/configs-{comp_name}/ -r zip" 
            )
            print(f"configs-{comp_name} uploaded")
        except:
            print(f"configs-{comp_name} doesn't exists.")

    if models:
        # models
        try:
            os.system(
                f"kaggle datasets version -m {version_name} -p ../models/models-{comp_name}/ -r zip -q"
            )
            print(f"models-{comp_name} uploaded")
        except:
            print(f"models-{comp_name} doesn't exists.")

    if src:
        # src
        try:
            os.system(
                f"kaggle datasets version -m {version_name} -p ../src-{framework_name}/"
            )
            print(f"src-{framework_name} uploaded")
        except:
            print(f"src-{framework_name} doesn't exists.")

    if working:
        # src
        try:
            os.system(f"kaggle datasets version -m {version_name} -p ../working/")
            print(f"working uploaded")
        except:
            print(f"working doesn't exists.")

    if input_:
        try:
            os.system(
                f"kaggle datasets version -m {version_name} -p ../input/input-{comp_name}/ -r zip -q"
            )
            print(f"input-{comp_name} uploaded")
        except:
            print(f"input-{comp_name} doesn't exists.")


if __name__ == "__main__":
    configs = False  # configs
    models = False  # models
    src = True  # src_framework3
    working = False  # working
    input_ = False  # we never push input_ twice [already pushed once]
    comment = "amz_reg_tabnet_src"  # don't keep space

    sync_(configs, models, src, working, input_, comment)
    print("Done")
