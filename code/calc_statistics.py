"""
This code calculates statistics for Lomb-Scargle and ACF methods.
"""

import numpy as np
import pandas as pd

import starrotate as sr
import kepler_data as kd
import sigma_clip as sc
from astropy.timeseries import LombScargle


def load_and_process(path_to_light_curve):
    """
    Load the light curve, sigma clip, join quarters, sort, and start at zero.

    Args:
        path_to_light_curve (str): The path to the light curve folder that
            contains a fits file for each quarter.

    Returns:
        x (array): The time array, starting from zero.
        y (array): The flux array.
        yerr (array): The flux uncertainty array
    """

    # Load the light curve
    x, y, yerr = kd.load_and_split(path_to_light_curve)

    # Sigma clip in each quarter
    clipped_x, clipped_y, clipped_yerr = [], [], []
    for i in range(len(x)):
        _, mask = sc.sigma_clip(y[i], nsigma=4)
        clipped_x.append(x[i][mask])
        clipped_y.append(y[i][mask])
        clipped_yerr.append(yerr[i][mask])

    x = np.array([i for j in clipped_x for i in j])
    y = np.array([i for j in clipped_y for i in j])
    yerr = np.array([i for j in clipped_yerr for i in j])

    # Sort arrays
    inds = np.argsort(x)
    x, y, yerr = x[inds], y[inds], yerr[inds]

    # Start at zero
    x -= x[0]

    return x, y, yerr


def get_peak_statistics(x, y):
    """
    Get the positions and height of peaks in an array.

    Args:
        x (array): the x array (e.g. period or lag).
        y (array): the y array (e.g. power or ACF).

    Returns:
        x_peaks1 (array): the peak positions, sorted in order of descending
            peak height.
        y_peaks1 (array): the peak heights, sorted in order of descending
            peak height.
        x_peaks2 (array): the peak positions, sorted in order of ascending
            x-position.
        y_peaks2 (array): the peak heights, sorted in order of ascending
            x-position.
    """

    # Array of peak indices
    peaks = np.array([i for i in range(1, len(y)-1) if y[i-1] <
                      y[i] and y[i+1] < y[i]])

    # extract peak values
    x_peaks = x[peaks]
    y_peaks = y[peaks]

    # sort by height
    inds = np.argsort(y_peaks)
    x_peaks1, y_peaks1 = x_peaks[inds][::-1], y_peaks[inds][::-1]

    # sort by position
    inds2 = np.argsort(x_peaks)
    x_peaks2, y_peaks2 = x_peaks[inds2], y_peaks[inds2]

    return x_peaks1, y_peaks1, x_peaks2, y_peaks2


def get_statistics(y, ps, power, lags, acf):
    """
    Calculate statistics of the LS periodogram and ACF.

    Args:
        y (array): the flux array.
        ps (array): the period array from the LS periodogram.
        power (array): the power array from the LS periodogram.
        lags (array): the lag array from the ACF.
        acf (array): the acf array from the ACF.

    Returns:
        highest 3 pgram peak heights (array).
        highest 3 pgram peak periods (array).
        highest 3 ACF peak heights (array).
        highest 3 ACF peak periods (array).
        1st 3 ACF peak heights (array).
        1st 3 ACF peak periods (array).
        highest 3 acf-pgram peak heights (array).
        highest 3 acf-pgram peak periods (array).
        MAD of the LS periodogram (float).
        RMS of the LS periodogram (float).
        MAD of the ACF (float).
        RMS of the ACF (float).
        Rvar (float).
    """


    # Highest 3 peak heights and positions in the LS periodogram
    peak_periods1, peak_powers1, _, _ = get_peak_statistics(ps, power)
    ls_mad = np.median(np.abs(power))
    ls_rms = np.sqrt(np.mean(power**2))

    # Highest and first 3 peak heights and positions in the LS periodogram
    lags1, acf1, lags2, acf2 = get_peak_statistics(lags, acf)

    # Calculate periodogram of ACF
    acf_freqs = np.linspace(1./100, 1./.1, 10000)
    acf_pgram = LombScargle(lags, acf).power(acf_freqs)
    acf_ls_periods, acf_ls_powers, _, _ = get_peak_statistics(1./acf_freqs,
                                                              acf_pgram)
    acf_ls_h3, acf_ls_p3 = acf_ls_powers[:3], acf_ls_periods[:3]

    acf_mad = np.median(np.abs(acf))
    acf_rms = np.sqrt(np.mean(acf**2))

    Rvar = np.percentile(y, 95) - np.percentile(y, 5)
    Rvar

    return peak_powers1[:3], peak_periods1[:3], acf1[:3], lags1[:3], \
        acf2[:3], lags2[:3], acf_ls_powers[:3], acf_ls_periods[:3], ls_mad, \
        ls_rms, acf_mad, acf_rms, Rvar, acf_freqs, acf_pgram
