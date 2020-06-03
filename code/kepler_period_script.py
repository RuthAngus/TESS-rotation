import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tess_rotation as tr
import starspot as ss
from tess_stars2px import tess_stars2px_function_entry


df = pd.read_csv("../kepler_stars_in_14_and_15.csv")
N = 4

cpm_periods = np.zeros(len(df))
for i, ticid in enumerate(df.TIC.values[:N]):
    print("Measuring period for", ticid)

    if os.path.exists("../data/{}_lc.csv".format(ticid)):
        lc = pd.read_csv("../data/{}_lc.csv".format(ticid))
        time, flux, flux_err = lc.time*1, lc.flux*1, lc.flux_err*1

        # Take off a straight line
        p = np.polyfit(time, flux, 2)
        flux -= np.polyval(p, time)

        star = ss.RotationModel(time, flux, flux_err)
        ls_period = star.ls_rotation(max_period=50)
        cpm_periods[i] = ls_period

        fig = star.big_plot(["ls"], method_xlim=(0, 50));
        fig.savefig("plots/{}_bigplot".format(ticid))
        plt.close()
    else:
        print("No light curve found")


xs = np.linspace(min(df.Prot.values[:N]), max(df.Prot.values[:N]), 100)
plt.plot(df.Prot.values[:N], cpm_periods[:N], ".")
plt.plot(xs, xs)
plt.savefig("kepler_period_comparison")


df["cpm_period"] = np.array(cpm_periods)
df.to_csv("../data/kepler_cpm_periods.csv")
