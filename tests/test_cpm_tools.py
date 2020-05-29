import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tess_stars2px import tess_stars2px_function_entry
import tess_rotation as tr
import eleanor
from astroquery.mast import Tesscut, TesscutClass

# def download_tess_cuts(ticid, lower_sector_limit=0, upper_sector_limit=1000,
#                        tesscut_path="/Users/rangus/projects/TESS-rotation/data/TESScut",
#                        ):

#     # Download light curves
#     sectors, star = tr.get_sectors(ticid, 14,
#                                    lower_sector_limit=lower_sector_limit,
#                                    upper_sector_limit=upper_sector_limit)
#     path_to_tesscut = "{0}/astrocut_{1:12}_{2:13}_{3}x{4}px".format(
#         tesscut_path, star.coords[0], star.coords[1], 68, 68)

#     for sector in sectors:
#         star = eleanor.Source(tic=ticid, sector=int(sector), tc=True)
#         fits_only = "tess-s{0}-{1}-{2}_{3:.6f}_{4:.6f}_{5}x{6}_astrocut.fits" \
#             .format(str(int(sector)).zfill(4), star.camera, star.chip,
#                     star.coords[0], star.coords[1], 68, 68)
#         full_path = os.path.join(path_to_tesscut, fits_only)

#         if not os.path.exists(full_path):
#             print("No cached file found. Downloading", full_path)

#             if not os.path.exists(path_to_tesscut):
#                 os.mkdir(path_to_tesscut)

#             print("Downloading sector", sector, "for TIC", ticid)
#             hdulist = Tesscut.download_cutouts(objectname="TIC 164668179",
#                                             sector=sector, size=68,
#                                             path=path_to_tesscut)
#         else:
#             print("Found cached file ", full_path)

# ticid = "164668179"


ticid = 48504458
tr.download_tess_cuts(ticid, upper_sector_limit=16)

# Create CPM light curve
tesscut_path="/Users/rangus/projects/TESS-rotation/data/TESScut"
time_cpm, flux_cpm = tr.CPM_recover(ticid, tesscut_path,
                                    any_observed_sector=14,
                                    upper_sector_limit=16)

# Stitch light curve
time, flux, flux_err = tr.stitch_light_curve(time, flux)

plt.errorbar(time, flux, yerr=flux_err, fmt="k.")
plt.savefig("test")
