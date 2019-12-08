from __future__ import print_function
import numpy as np
from astropy.io import fits
import glob
import os
from tqdm import trange


def load_and_join(LC_DIR):
    """
    load and join quarters together.
    Takes a list of fits file names for a given star.
    Returns the concatenated arrays of time, flux and flux_err
    """
    fnames = sorted(glob.glob(os.path.join(LC_DIR, "*fits")))
    hdulist = fits.open(fnames[0])
    t = hdulist[1].data
    time = t["TIME"]
    flux = t["PDCSAP_FLUX"]
    flux_err = t["PDCSAP_FLUX_ERR"]
    q = t["SAP_QUALITY"]
    m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
            (q == 0)
    x = time[m]
    med = np.median(flux[m])
    y = flux[m]/med - 1
    yerr = flux_err[m]/med
    for fname in fnames[1:]:
       hdulist = fits.open(fname)
       t = hdulist[1].data
       time = t["TIME"]
       flux = t["PDCSAP_FLUX"]
       flux_err = t["PDCSAP_FLUX_ERR"]
       q = t["SAP_QUALITY"]
       m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(flux_err) * \
               (q == 0)
       x = np.concatenate((x, time[m]))
       med = np.median(flux[m])
       y = np.concatenate((y, flux[m]/med - 1))
       yerr = np.concatenate((yerr, flux_err[m]/med))
    return x, y, yerr


def load_and_split(LC_DIR):
    """
    load individual quarters.
    Takes a list of fits file names for a given star.
    Returns a list of arrays of time, flux and flux_err

    """

    fnames = sorted(glob.glob(os.path.join(LC_DIR, "*fits")))
    time, flux, flux_err = [], [], []
    for i in range(len(fnames)):
        hdulist = fits.open(fnames[i])
        t = hdulist[1].data
        x, y, yerr = t["TIME"], t["PDCSAP_FLUX"], t["PDCSAP_FLUX_ERR"]
        m = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr) \
                        * (t["SAP_QUALITY"] == 0)

        time.append(x[m])
        med = np.median(y[m])
        flux.append(y[m]/med - 1)
        flux_err.append(yerr[m]/med)

    return time, flux, flux_err


def sigma_clip(x, nsigma=3):
    """
    Sigma clipping for 1D data.

    Args:
        x (array): The data array. Assumed to be Gaussian in 1D.
        nsigma (float): The number of sigma to clip on.

    Returns:
        newx (array): The clipped x array.
        mask (array): The mask used for clipping.
    """

    m = np.ones(len(x)) == 1
    newx = x*1
    oldm = np.array([False])
    i = 0
    while sum(oldm) != sum(m):
        oldm = m*1
        sigma = np.std(newx)
        m &= np.abs(np.median(newx) - x)/sigma < nsigma
        # m &= m
        newx = x[m]
        i += 1
    return x[m], m


def running_sigma_clip(interval, t, x, nsigma=3):
    """
    Sigma clipping using a running median.

    Args:
        interval (float): Time interval to bin on.
        t (array): The time array.
        x (array): The flux array.
        nsigma (float): The number of sigma to clip on.

    """

    nints = int((max(t) - min(t)) / interval)
    full_mask = []
    n = 0
    for i in trange(nints):
        m = (min(t) + i*interval <= t) * (t < min(t) + (i + 1)*interval)
        newx, clip = sigma_clip(x[m], nsigma=nsigma)
        full_mask.append(clip)
        n += sum(m)

    if len(full_mask) != len(x):
        m = (min(t) + (i + 1)*interval < t)
        newx, clip = sigma_clip(x[m], nsigma=nsigma)
        full_mask.append(clip)
        n += sum(m)

    full_mask = [i for j in full_mask for i in j]
    return x[full_mask], full_mask
