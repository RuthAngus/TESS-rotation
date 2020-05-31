import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tess_stars2px import tess_stars2px_function_entry
import tess_rotation as tr
import eleanor
from astroquery.mast import Tesscut, TesscutClass


def test_download_make_and_stitch():
    # ticid = "164668179"
    # ticid = 48504458
    # ticid = 270608461
    ticid = 176954932
    tr.download_tess_cuts(ticid, upper_sector_limit=16)

    # Create CPM light curve
    tesscut_path="/Users/rangus/projects/TESS-rotation/data/TESScut"
    time_cpm, flux_cpm, sectors = tr.CPM_recover(ticid, tesscut_path,
                                                upper_sector_limit=16)

    # Stitch light curve
    time, flux, flux_err = tr.stitch_light_curve(ticid, sectors, time_cpm,
                                                flux_cpm)

    plt.errorbar(time, flux, yerr=flux_err, fmt="k.", alpha=.5)
    plt.savefig("test")


def test_get_CPM_aperture():
    aperture = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])

    collims, rowlims = tr.get_CPM_aperture(aperture, xstart=0, ystart=0)
    assert len(collims) == 2
    assert len(rowlims) == 2


test_download_make_and_stitch()
# test_get_CPM_aperture()
