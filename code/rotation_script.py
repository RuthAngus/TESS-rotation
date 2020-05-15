"""
Measure rotation periods using plots.

"""

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
    path = "/Users/rangus/projects/TESSlightcurves/"
    for ticid in ids:
        str_ticid = str(int(ticid)).zfill(16)
        tfile = "tess?????????????-s????-{}-????-s_lc.fits".format(str_ticid)
        ticpath = os.path.join(path, tfile)
        fnames = sorted(glob.glob(ticpath))
        print(ticid, len(fnames))


def find_unique_ids(fn):
    ids = []
    for f in fn:
        ticid = extract_ticid(f)
        if len(str(ticid)) > 8:
            ids.append(ticid)
    id_array = np.array(ids)
    return np.unique(id_array)


def load_sectors(ticid, flux_type="sap_flux"):
    path = "/Users/rangus/projects/TESSlightcurves/"
    str_ticid = str(int(ticid)).zfill(16)
    tfile = "tess?????????????-s????-{}-????-s_lc.fits".format(str_ticid)

    ticpath = os.path.join(path, tfile)
    fnames = sorted(glob.glob(ticpath))
    time, flux, flux_err = lcs.tools.load_and_split_TESS(
        fnames, flux_type="SAP_FLUX")
    return time, flux, flux_err


def sigma_clip_TESS_sector(x, y, yerr):
    x, y, yerr = np.array(x), np.array(y), np.array(yerr)

    # Initial removal of extreme outliers.
    m = rt.sigma_clip(y, nsigma=7)
    x, y, yerr = x[m], y[m], yerr[m]

    # Remove outliers using Sav-Gol filter
    smooth, mask = rt.filter_sigma_clip(x, y)
    resids = y - smooth
    stdev = np.std(resids)
    return x[mask], y[mask], yerr[mask], stdev


def clip_and_join(time, flux, flux_err):
    x, y, yerr, std = [], [], [], []
    for i in trange(len(time)):
        t, f, ferr, stdev = sigma_clip_TESS_sector(time[i], flux[i],
                                                   flux_err[i])
        x.append(t)
        y.append(f)
        yerr.append(ferr)
        std.append(stdev)
    return x, y, yerr, std


def find_gap_times(time):

    start_times, stop_times = [], []
    for i in range(len(time)):
        start_times.append(min(time[i]))
        stop_times.append(max(time[i]))
    return start_times[1:]


def find_steps(y):
    # Meds
    meds = []
    for i in range(len(y)):
        meds.append(np.median(y[i]))
    return meds[1:]


def format_lc(t, y, yerr, subsample=10):

    # turn into arrays
    t = np.array([i for j in t for i in j], dtype="float64")
    y = np.array([i for j in y for i in j], dtype="float64")
    yerr = np.array([i for j in yerr for i in j], dtype="float64")

    # Remove NaNs
    m = np.isfinite(t) * np.isfinite(y) * np.isfinite(yerr)
    t, y, yerr = t[m], y[m], yerr[m]

    # Subsample
    t, y, yerr = t[::subsample], y[::subsample], yerr[::subsample]

    # Sort by time
    inds = np.argsort(t)
    t, y, yerr = t[inds], y[inds], yerr[inds]

    return t, y, yerr


def process_light_curve(ticid):

    # Load sectors
    time, flux, flux_err = load_sectors(ticid)
    nsectors = len(time)
    print("star:", ticid, nsectors, "sectors")

    print("Sigma clip and join light curves...")
    x, y, yerr, std = clip_and_join(time, flux, flux_err)

    # Find the start times of the sectors.
    gap_times = find_gap_times(x)

    # Subtract off the median.
    original_flux_median = np.median(y[0])
    y -= original_flux_median

    # Find initial step values
    steps = find_steps(y)

    # Format the light curve
    t, y, yerr = format_lc(x, y, yerr)

    # Start at time 0
    gap_times -= t[0]
    t -= t[0]

    print("Infer the offset values...")
    star = sps.StitchModel(t, y, yerr, gap_times, steps, 2.0)
    star.model_offsets()
    map_soln, success = star.find_optimum()
    print("SUCCESS: ", success, int(success))

    # Correct the light curve
    best_fit_mu = sps.step_model(t, gap_times, steps)
    corrected_y = y - best_fit_mu

    # Median normalize the light curve.
    corrected_y /= original_flux_median

    # Save to file
    print("Saving light curve")
    df = pd.DataFrame(dict({"time": t, "flux": corrected_y,
                            "flux_err": yerr, "success": int(success)}))
    df.to_csv("results/{}_SAP_lc.csv".format(ticid))

    return t, corrected_y, yerr, success


def measure_period(ticid, t, corrected_y, yerr):

    print("Measure the rotation period...")
    rotate = ss.RotationModel(t, corrected_y, yerr)
    ls_period = rotate.ls_rotation()
    acf_period = rotate.acf_rotation(interval=0.00138889, cutoff=1.5,
                                    window_length=1999)

    period_grid = np.arange(.5, 3*ls_period, 30./1000.)
    if ls_period > 10:
        period_grid = np.linspace(.5, 30, 1000.)
    pdm_period = rotate.pdm_rotation(period_grid)

    # Make the figure
    fig = rotate.big_plot(methods=["pdm", "acf", "ls"], xlim=(0, 100),
                            method_xlim=(-1, 50))
    rvar = np.log10(rotate.Rvar*1e6)
    print("log10(Rvar) = ", rvar, "ppm")
    plt.title("log10(Rvar) = {0:.2f}".format(rvar), fontsize=20)
    fig.savefig("plots/{}_plot".format(ticid))

    # Get peak heights
    ls_peak_pos, ls_peak_height = rt.get_peak_statistics(
        1./rotate.freq, rotate.power)
    acf_peak_pos, acf_peak_height = rt.get_peak_statistics(
        rotate.lags, rotate.acf)
    pdm_peak_pos, pdm_peak_height = rt.get_peak_statistics(
        rotate.phase, -rotate.phis)


    return rvar, ls_period, acf_period, pdm_period[0], pdm_period[1], \
        ls_peak_height[:3], ls_peak_pos[:3], acf_peak_height[:3], \
        acf_peak_pos[:3], pdm_peak_height[:3], pdm_peak_pos[:3]


if __name__ == "__main__":
    # # Load the light curve names
    # fn = glob.glob("/Users/rangus/projects/TESSlightcurves/*fits")
    # ids = find_unique_ids(fn)
    # df = pd.DataFrame(dict({"ticid": ids}))
    # df.to_csv("ticids.csv")

    # df = pd.read_csv("ticids.csv")

    assert os.path.exists("results.csv") == False, \
        "WARNING: results.csv file already exists! Delete/rename this file."

    df = pd.read_csv("stars_with_10_or_more_sectors.csv")
    ticids = df.ticid.values

    clobber = False

    for i in range(2, 3):  # len(ticids)):
        ticid = ticids[i]
        print(ticid, i, "of", len(ticids))
        try:

            # Stitch light curve.
            if not os.path.exists("results/{}_SAP_lc.csv"
                                    .format(ticid)) or clobber:
                t, y, yerr, success = process_light_curve(ticid)

            # Load saved version
            else:
                print("Loading saved light curve")
                # Load light curve
                df = pd.read_csv("results/{}_lc.csv".format(ticid))
                t = df.time.values
                y = df.flux.values
                yerr = df.flux_err.values
                success = df.success.values[0]

            fig = plt.figure(figsize=(16, 9))
            ax1 = fig.add_subplot(311)
            ax1.plot(t, y, "k.", alpha=.5)
            ax2 = fig.add_subplot(312)
            ax2.plot(t, y, "k.", alpha=.5)
            ax2.set_xlim(0, 100)
            ax3 = fig.add_subplot(313)
            ax3.plot(t, y, "k.", alpha=.5)
            ax3.set_xlim(0, 30)
            plt.savefig("plots/{}_SAP_lc".format(ticid))
            plt.close

            # Measure period
            rvar, ls_period, acf_period, pdm_period, err, ls_peak_heights, \
                ls_peak_pos, acf_peak_heights, acf_peak_pos, \
                pdm_peak_heights, pdm_peak_pos = \
                measure_period(ticid, t, y, yerr)

            # Save to file
            df = pd.DataFrame(dict({"ticid": ticid,
                                    "ls_period": ls_period,
                                    "acf_period": acf_period,
                                    "pdm_period": pdm_period,
                                    "period_err": err,
                                    "Rvar": rvar,
                                    "ls_peak_height1": ls_peak_heights[0],
                                    "ls_peak_height2": ls_peak_heights[1],
                                    "ls_peak_height3": ls_peak_heights[2],
                                    "ls_peak_pos1": ls_peak_pos[0],
                                    "ls_peak_pos2": ls_peak_pos[1],
                                    "ls_peak_pos3": ls_peak_pos[2],
                                    "acf_peak_height1": acf_peak_heights[0],
                                    "acf_peak_height2": acf_peak_heights[1],
                                    "acf_peak_height3": acf_peak_heights[2],
                                    "acf_peak_pos1": acf_peak_pos[0],
                                    "acf_peak_pos2": acf_peak_pos[1],
                                    "acf_peak_pos3": acf_peak_pos[2],
                                    "pdm_peak_height1": pdm_peak_heights[0],
                                    "pdm_peak_height2": pdm_peak_heights[1],
                                    "pdm_peak_height3": pdm_peak_heights[2],
                                    "pdm_peak_pos1": pdm_peak_pos[0],
                                    "pdm_peak_pos2": pdm_peak_pos[1],
                                    "pdm_peak_pos3": pdm_peak_pos[2],
                                    "stitch_success": success}),
                            index=[i])

            with open("results_SAP.csv", 'a') as f:
                df.to_csv(f, mode='a', header=f.tell()==0)

        except:
            pass
