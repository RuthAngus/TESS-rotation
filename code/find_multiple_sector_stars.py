import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import lightcurvestuff as lcs
import starspot as ss
from tqdm import trange
import starspot.rotation_tools as rt
import re
import starspot.stitch as sps


def extract_ticid(fn):
    """
    Read TIC id from the file name.

    """
    return int(re.findall(r'\d+', fn)[2])


def find_multiple_sector_stars(ids):
    tics = []
    path = "/Users/rangus/projects/TESSlightcurves/"
    for i in range(len(ids)):
        str_ticid = str(int(ids[i])).zfill(16)
        tfile = "tess?????????????-s????-{}-????-s_lc.fits".format(str_ticid)
        ticpath = os.path.join(path, tfile)
        fnames = glob.glob(ticpath)
        if len(fnames) > 10:
            print(ids[i], len(fnames))
            tics.append(ids[i])
        if i % 100 == 0:
            print(i, "of", len(ids), ids[i])
    return tics


if __name__ == "__main__":
    df = pd.read_csv("ticids.csv")
    ticids = df.ticid.values

    tics = find_multiple_sector_stars(ticids)
    df = pd.DataFrame(dict({"ticid": tics}))
    df.to_csv("stars_with_10_or_more_sectors.csv")
