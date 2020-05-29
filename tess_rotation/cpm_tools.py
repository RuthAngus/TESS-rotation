import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
import astropy.stats as aps

from astroquery.mast import Tesscut
import lightkurve as lk
import eleanor
import tess_cpm
import starspot as ss

import pkg_resources
st_file = pkg_resources.resource_filename(__name__, 'sector_times.csv')

from tess_stars2px import tess_stars2px_function_entry

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
    detrended_lc = dw.get_aperture_lc(split=True,
                                      data_type="cpm_subtracted_flux")
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


def download_tess_cuts(ticid, lower_sector_limit=0, upper_sector_limit=1000,
                       tesscut_path="/Users/rangus/projects/TESS-rotation/data/TESScut"):

    # Download light curves
    sectors, star = get_sectors(ticid, 14,
                                lower_sector_limit=lower_sector_limit,
                                upper_sector_limit=upper_sector_limit)
    path_to_tesscut = "{0}/astrocut_{1:12}_{2:13}_{3}x{4}px".format(
        tesscut_path, star.coords[0], star.coords[1], 68, 68)

    for sector in sectors:
        star = eleanor.Source(tic=ticid, sector=int(sector), tc=True)
        fits_only = "tess-s{0}-{1}-{2}_{3:.6f}_{4:.6f}_{5}x{6}_astrocut.fits" \
            .format(str(int(sector)).zfill(4), star.camera, star.chip,
                    star.coords[0], star.coords[1], 68, 68)
        full_path = os.path.join(path_to_tesscut, fits_only)

        if not os.path.exists(full_path):
            print("No cached file found. Downloading", full_path)

            if not os.path.exists(path_to_tesscut):
                os.mkdir(path_to_tesscut)

            print("Downloading sector", sector, "for TIC", ticid)
            hdulist = Tesscut.download_cutouts(
                objectname="TIC {}".format(ticid), sector=sector, size=68,
                path=path_to_tesscut)
        else:
            print("Found cached file ", full_path)


def get_sectors(ticid, any_observed_sector, lower_sector_limit=0,
                upper_sector_limit=50):
    star = eleanor.Source(tic=ticid, sector=int(any_observed_sector),
                          tc=True)
    outID, outEclipLong, outEclipLat, outSec, outCam, outCcd, outColPix, \
        outRowPix, scinfo = tess_stars2px_function_entry(
            int(ticid), star.coords[0], star.coords[1])
    sectors = outSec[(lower_sector_limit < outSec)
                     & (outSec < upper_sector_limit)]
    return sectors, star


def get_fits_filenames(tesscut_path, sector, camera, ccd, ra, dec, xpix=68,
                       ypix=68):
    path_to_tesscut = "{0}/astrocut_{1:12}_{2:13}_{3}x{4}px".format(
        tesscut_path, ra, dec, xpix, ypix)

    fits_image_filename = \
        "{0}/tess-s{1}-{2}-{3}_{4:.6f}_{5:.6f}_{6}x{7}_astrocut.fits".format(
            path_to_tesscut, str(int(sector)).zfill(4), camera, ccd, ra, dec,
            xpix, ypix)

    return fits_image_filename


def get_CPM_aperture(aperture, npix=13, xstart=27, ystart=28):
    x = np.arange(npix)
    row_index, column_index = np.meshgrid(x, x, indexing="ij")
    aperture_mask = aperture == 1
    column_inds = column_index[aperture_mask]
    row_inds = row_index[aperture_mask]
    xpixels = tuple(xstart + row_inds)
    ypixels = tuple(ystart + column_inds)
    return xpixels, ypixels


def CPM_one_sector(ticid, tesscut_path, sector, camera, ccd, ra, dec):
    fits_file = get_fits_filenames(tesscut_path, sector, camera, ccd, ra, dec,
                                   xpix=68, ypix=68)

    # Get the Eleanor aperture
    star = eleanor.Source(tic=ticid, sector=int(sector), tc=True)
    data = eleanor.TargetData(star)
    xpixels, ypixels = get_CPM_aperture(data.aperture)

    # Create CPM light curve
    select_aperture(sector, ypixels, xpixels, fits_file, plot=False)
    x, y = make_lc_single_sector(sector, ypixels, xpixels, fits_file,
                                    plot=False, save_to_file=False)
    return x, y


def CPM_recover(ticid, tesscut_path, any_observed_sector=1,
                lower_sector_limit=0, upper_sector_limit=1000):

    print("Searching for observed sectors...")
    sectors, star = get_sectors(ticid, any_observed_sector,
                                lower_sector_limit=lower_sector_limit,
                                upper_sector_limit=upper_sector_limit)
    print("sectors found: ", sectors)

    print("Creating light curve..")
    xs, ys = [], []
    for sector in sectors:
        print("sector", sector)
        star = eleanor.Source(tic=ticid, sector=int(sector), tc=True)
        x, y = CPM_one_sector(ticid, tesscut_path, star.sector, star.camera,
                              star.chip, star.coords[0], star.coords[1])
        xs.append(x)
        ys.append(y)

    return xs, ys, sectors


def stitch_light_curve(ticid, sectors, time, flux):

    gap_times = [time[i][0] for i in range(len(time))]
    time = np.array([i for j in time for i in j])
    flux = np.array([i for j in flux for i in j])

    # sector_times = pd.read_csv(st_file)

    # gap_times = []
    # for sector in sectors:
    #     st = sector_times.event.values == "start"
    #     m = sector == sector_times.sector.values[st]
    #     gap_times.append(float(sector_times.TJD.values[st][m]))

    gap_times.pop(0)

    # Estimate flux uncertainties
    m = ss.sigma_clip(flux, nsigma=6)
    t1, f1 = time[m], flux[m]

    # Then a sigma clip using a Sav-Gol filter for smoothing
    mask, smooth = ss.filter_sigma_clip(t1, f1, window_length=99)
    t2, f2 = t1[mask], f1[mask]
    f2_err = np.ones_like(t2) * 1.5 \
        * aps.median_absolute_deviation(f2-smooth[mask])

    x, y, yerr = t2[::10], f2[::10], f2_err[::10]

    # Calculate the best-fit offsets
    print("Fitting GP and offset model...")
    steps = np.zeros(len(gap_times))
    star = ss.StitchModel(x, np.ascontiguousarray(y, dtype=np.float64),
                          yerr, gap_times, steps, 2.0)
    star.model_offsets()
    map_soln = star.find_optimum()
    mu_gp, var = star.evaluate_model(x)

    best_steps = []
    for i in range(len(steps)):
        best_steps.append(float(map_soln[0][f"step{i+1}"]))

    # Create the stitched light curve
    stitched = f2 - ss.step_model(t2, gap_times, best_steps)

    return t2, stitched, f2_err
