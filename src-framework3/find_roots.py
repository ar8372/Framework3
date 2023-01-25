import os
import sys
from utils import * 



def roots(exp_no):
    with open(os.path.join(sys.path[0], "ref.txt"), "r") as x:
        for i in x:
            comp_name = i
    x.close()

    Table = load_pickle(f"../configs/configs-{comp_name}/Table.pkl")
    base = load_pickle(f"../configs/configs-{comp_name}/useful_features_l_1.pkl")

    Features = []
    Exp = []
    exp_list = [exp_no]

    while True:
        current_features = []
        temp = []
        for e in exp_list:
            if e == 'base':
                temp.append((e, 'NA'))
                continue
            else:
                e = int(e)
                current_features += Table.loc[Table.exp_no == e, 'features_list'][e]
                temp.append((e, Table.loc[Table.exp_no == e, 'model_name'][e]))
        Exp.append(temp)
        # print(exp_list)
        # print()
        Features.append(current_features)
        entered = False
        exp_list = []
        for feat in current_features:
            # 
            if feat in base:
                exp_list.append('base')
                pass 
            else:
                entered = True
                exp_list.append(feat.split("_")[2])
        exp_list= list(sorted(set(exp_list)))
        
        if entered is False:
            # all base features 
            # so break 
            break  
    # for f in Features:
    #     print(f)
    #     print()
    for i,j in enumerate(Exp[::-1]):
        print(f"LEVEL {i}:", j)
        print()


    




if __name__ == '__main__':

    exp_no = 240
    roots(exp_no)