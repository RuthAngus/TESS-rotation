import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tess_rotation as tr
import starspot as ss
from tess_stars2px import tess_stars2px_function_entry


def download_make_and_stitch(ticid):
    tr.download_tess_cuts(ticid, upper_sector_limit=16)

    # Create CPM light curve
    tesscut_path="/Users/rangus/projects/TESS-rotation/data/TESScut"
    time_cpm, flux_cpm, sectors = tr.CPM_recover(ticid, tesscut_path,
                                                 upper_sector_limit=16)

    # Stitch light curve
    time, flux, flux_err = tr.stitch_light_curve(ticid, sectors, time_cpm,
                                                flux_cpm)

    plt.errorbar(time, flux, yerr=flux_err, fmt="k.", alpha=.5)
    plt.savefig("plots/{}_cpm_lc.png".format(ticid))

    return time, flux, flux_err


# df = pd.read_csv("~/Downloads/Kepler_Tess.csv")
# df = df.drop_duplicates(subset="TIC")

# # Find slow-ish-but-not-too-slow rotators
# m_long = (df.Prot.values < 20) & (df.Prot.values > 5)
# df = df.iloc[m_long]

# ras = df.ra.values
# decs = df.dec.values
# ticids = df.TIC.values

# # Find stars observed in sectors 14 and 15.
# double_inds = []
# for i in range(len(ras)):
#     outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, outColPix, \
#         outRowPix, scinfo = tess_stars2px_function_entry(ticids[i], ras[i],
#                                                          decs[i])
#     if len(outSec) > 1:
#         if outSec[0] == 14 and outSec[1] == 15:
#             print(outSec, outID[0])
#             double_inds.append(i)
# df = df.iloc[double_inds]
# df.to_csv("../kepler_stars_in_14_and_15.csv")

# ticid = "164668179"
# ticid = 48504458
# ticid = 270608461

df = pd.read_csv("../kepler_stars_in_14_and_15.csv")
for ticid in df.TIC.values:

    time, flux, flux_err = download_make_and_stitch(ticid)
    save_df = pd.DataFrame(dict({"time": time,
                                "flux": flux,
                                "flux_err": flux_err
                                }))
    save_df.to_csv("../data/{}_lc.csv".format(ticid))
