import os
import numpy as np
import pandas as pd
import starrotate as sr
import calc_statistics as cs


def test_load_and_process():
    i = 0
    mc1 = pd.read_csv("../data/Table_1_Periodic.txt")
    kplr_path = "/Users/rangus/.kplr/data/lightcurves"
    path_to_light_curve = os.path.join(kplr_path,
                                       str(int(mc1.iloc[i].kepid)).zfill(9))
    x, y, yerr = cs.load_and_process(path_to_light_curve)


if __name__ == "__main__":
    test_load_and_process()
