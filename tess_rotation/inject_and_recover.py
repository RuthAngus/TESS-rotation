from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
import eleanor
from tqdm import tnrange
import starry

from scipy.interpolate import RectBivariateSpline
from .cpm_tools import get_sectors, get_fits_filenames, make_lc_single_sector
from .inject import get_random_light_curve, inject_one_sector_starry


class InjectStar(object):

    def __init__(self, ticid, tesscut_path, lower_sector_limit=0,
                 upper_sector_limit=1000, xpix=68, ypix=68):

        self.ticid = ticid
        self.upper_sector_limit = upper_sector_limit
        self.lower_sector_limit = lower_sector_limit
        self.xpix, self.ypix = 68, 68

        print("Finding the sectors that star was observed in.")
        sectors, _ = get_sectors(self.ticid,
                                 upper_sector_limit=self.upper_sector_limit,
                                 lower_sector_limit=self.lower_sector_limit)
        self.sectors = sectors
        print("Found sectors", sectors)

        # Now get the list of eleanor star objects, one for each sector, and
        # the names of fits files for each sector.
        # Also, load the time arrays
        print("Loading time arrays")
        time, inj_file_names = [], []
        elstar, fits_file_names = [], []
        for i, sector in enumerate(self.sectors):
            print("sector", sector)
            s = eleanor.Source(tic=self.ticid, sector=int(sector), tc=True)
            elstar.append(s)

            # Get file names
            f = get_fits_filenames(tesscut_path, sector, s.camera, s.chip,
                                   s.coords[0], s.coords[1])
            fits_file_names.append(f)

            # Load the TESScut FFI data (get time array)
            hdul = fits.open(f)
            postcard = hdul[1].data
            t = postcard["TIME"]*1.
            flux = postcard["FLUX"]*1.
            time.append(t)

            # Get injection file names
            path_to_tesscut = "{0}/astrocut_{1:12}_{2:13}_{3}x{4}px".format(
                tesscut_path, s.coords[0], s.coords[1], xpix, ypix)
            inj_dir = "{0}/injected".format(path_to_tesscut)
            injection_filename = \
                "{0}/tess-s{1}-{2}-{3}_{4:.6f}_{5:.6f}_{6}x{7}_astrocut.fits"\
                .format(inj_dir, str(int(sector)).zfill(4), s.camera, s.chip,
                        s.coords[0], s.coords[1], xpix, ypix)
            inj_file_names.append(injection_filename)

        self.elstar = elstar
        self.fits_file_names = fits_file_names
        self.time = time
        self.time_array = np.array([i for j in self.time for i in j])
        self.inj_file_names = inj_file_names

    def generate_signal(self, period, amplitude, baseline):
        # Generate the starry signal
        # Create the multi-sector signal
        signal = baseline + get_random_light_curve(self.time_array, period,
                                                   amplitude)
        self.signal = signal

    def inject_signal(self):

        for i, sector in enumerate(self.sectors):

            print("sector", sector)
            s = self.elstar[i]

            mask = (self.time[i][0] <= self.time_array) & \
                (self.time_array <= self.time[i][-1])

            assert len(self.signal[mask]) == len(self.time[i])
            inject_one_sector_starry(self.ticid, sector, self.signal[mask],
                                     s.camera, s.chip, s.position_on_chip,
                                     self.fits_file_names[i],
                                     self.inj_file_names[i])

    def CPM_one_sector(self, i, sector):
        """
        Create CPM light curve for one sector.
        """

        fits_file = self.fits_file_names[i]
        injected_filename = self.inj_file_names[i]

        # Create CPM light curve
        x, y = tr.make_lc_single_sector(sector, [33, 35], [33, 35],
                                        injected_filename, plot=False,
                                        save_to_file=False)
        return x, y

    def CPM_recover(self):
        """
        Create CPM light curve from all sectors.
        """

        print("Creating light curve..")
        xs, ys = [], []

        for i, sector in enumerate(self.sectors):
            print("sector", sector)
            x, y = self.CPM_one_sector(i, sector)
            xs.append(x)
            ys.append(y)

        self.time_cpm = xs
        self.flux_cpm = ys
        return xs, ys
