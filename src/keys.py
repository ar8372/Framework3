from collections import defaultdict
import pickle
import os
import sys


class KeyMaker:
    def __init__(
        self,
        random_state=21,
        target_name="Survived",
        id_name="Passenger_ID",
        comp_index=0
    ):
        #
        with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
            for i in x:
                comp_name = i
        x.close()
        self.comp_list = ["regression", "2class","multi_class", "multi_label"]
        self._comp_name = comp_name
        self.random_state = random_state
        self.target_name = target_name
        self.id_name = id_name
        self.comp_type = self.comp_list[comp_index]
        self.locker = defaultdict()
        self.update()  # dumps files as pickel

    def help(self):
        print("comp_index:=> ","0: regression", "1: 2class","2: multi_class", "3: multi_label")

    def __call__(self, random_state="--|--", target_name="--|--", id_name="--|--", comp_type="--|--"):
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

        self.update()  # dump files to pickel

    def update(self):
        # updates the locker
        a = self.locker
        self.locker["comp_name"] = self._comp_name
        self.locker["random_state"] = self.random_state
        self.locker["target_name"] = self.target_name
        self.locker["id_name"] = self.id_name
        self.locker["comp_type"] = self.comp_type
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
