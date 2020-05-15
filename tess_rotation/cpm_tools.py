import tess_cpm
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd

plt.rcParams["figure.figsize"] = (14, 10)

def select_aperture(sector, collims, rowlims, fits_files, plot=True):
    dw = tess_cpm.Source(fits_files, remove_bad=True)
    dw.set_aperture(rowlims=rowlims, collims=collims)
    if plot:
        dw.plot_cutout(rowlims=[0, 64], collims=[0, 64], show_aperture=True);
    return collims, rowlims


def make_lc_single_sector(sector, collims, rowlims, fits_files,
                          save_to_file=True, plot=True):

    dw = tess_cpm.Source(fits_files, remove_bad=True)
    dw.set_aperture(rowlims=rowlims, collims=collims)
    dw.add_cpm_model(predictor_method='similar_brightness')

    dw.add_poly_model()
    dw.set_regs([0.001, 0.1])

    dw.holdout_fit_predict(k=100)
    aperture_normalized_flux = dw.get_aperture_lc(data_type="normalized_flux")
    aperture_cpm_prediction = dw.get_aperture_lc(data_type="cpm_prediction")
    detrended_lc = dw.get_aperture_lc(split=True, data_type="cpm_subtracted_flux")
    cpm_lc = dw.get_aperture_lc(data_type="cpm_subtracted_flux")

    if save_to_file:
        df = pd.DataFrame(dict({"time": dw.time, "flux": cpm_lc}))
        df.to_csv("sector_{}.csv".format(str(sector).zfill(2)))

    if plot:
        plt.figure(figsize=(16, 4), dpi=200)
        plt.plot(dw.time, cpm_lc, "k.")
        plt.xlabel("time")
        plt.ylabel("flux")

    return dw.time, cpm_lc
