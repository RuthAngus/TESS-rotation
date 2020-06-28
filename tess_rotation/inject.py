from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
import eleanor
from tqdm import tnrange
import starry

from scipy.interpolate import RectBivariateSpline
from .cpm_tools import get_sectors, get_fits_filenames


def pathLookup(ccd, camera, sector):
    """
    Gets the datestring and the subdirectory for the specified PRF.
    The datestring and directory name can be found from the ccd, camera and sector.

    Inputs
    -------
    ccd
        (int) number of the TESS ccd. Accepts values from 1-4.
    camera
        (int) number of the TESS camera. Accepts values from 1-4
    sector
        (int) number of the TESS sector. Accepts values of 1 and above.

    Returns
    -------
    datestring
        (str) Date string used in the TESS prf files name.
    add_path
        (str) Directory name where the TESS PRF is stored. (e.g. "/start_s0001/")
    """
    if sector < 1:
        raise ValueError("Sector must be greater than 0.")
    if (camera > 4) | (ccd > 4):
        raise ValueError("Camera or CCD is larger than 4.")

    if sector <= 3:
        add_path = "/start_s0001/"
        datestring = "2018243163600"

        if camera >= 3:
            datestring = "2018243163601"
        elif (camera == 2) & (ccd == 4):
            datestring = "2018243163601"

    else:
        add_path = "/start_s0004/"
        datestring = "2019107181900"
        if (camera == 1) & (ccd >= 2):
            datestring = "2019107181901"
        elif (camera == 2):
            datestring = "2019107181901"
        elif (camera == 3) & (ccd >= 2) :
            datestring = "2019107181902"
        elif (camera == 4):
            datestring = "2019107181902"

    return datestring, add_path


def readOnePrfFitsFile(ccd, camera, col, row, path, datestring):
    """
    reads in the full, interleaved prf Array for a single row,col,ccd,camera location.

    Inputs
    -------
    ccd
        (int) CCD number
    camera
        (int) Camera number
    col
        (float) Specific column where the PRF was sampled.
    row
        (float) Specific row where the PRF was sampled.
    path
        (string) The full path of the data file. Can be the MAST Web address

    Returns
    ------
    prfArray
        (np array) Full 117 x 117 interleaved prf Array for the requested file.
    """

    fn = "cam%u_ccd%u/tess%13s-prf-%1u-%1u-row%04u-col%04u.fits" % \
        (camera, ccd, datestring, camera, ccd, row, col)

    filepath = os.path.join(path, fn)
    hdulistObj = fits.open(filepath)
    prfArray = hdulistObj[0].data

    return prfArray


def determineFourClosestPrfLoc(col, row):
    """
    Determine the four pairs of col,row positions of your target.
    These are specific to TESS and where they chose to report their PRFs.
    Inputs
    ------
    col
        (float) Column position
    row
        (float) Row position.

    Returns
    -------
    imagePos
        (list) A list of (col,row) pairs.
    """

    posRows = np.array([1, 513, 1025, 1536, 2048])
    posCols = np.array([45, 557, 1069, 1580,2092])

    difcol = np.abs(posCols - col)
    difrow = np.abs(posRows - row)

    # Expand out to the four image position to interpolate between,
    # Return as a list of tuples.
    imagePos = []
    for r in posRows[np.argsort(difrow)[0:2]]:
        for c in posCols[np.argsort(difcol)[0:2]]:
            imagePos.append((c,r))

    return imagePos


def getOffsetsFromPixelFractions(col, row):
    """
    Determine just the fractional part (the intra-pixel part) of the col,row position.
    For example, if (col, row) = (123.4, 987.6), then
    (colFrac, rowFrac) = (.4, .6).

    Function then returns the offset necessary for addressing the interleaved PRF array.
    to ensure you get the location appropriate for your sub-pixel values.

    Inputs
    ------
    col
        (float) Column position
    row
        (float) Row position.

    Returns
    ------
    (colFrac, rowFrac)
       (int, int) offset necessary for addressing the interleaved PRF array.
    """
    gridSize = 9

    colFrac = np.remainder(float(col), 1)
    rowFrac = np.remainder(float(row), 1)

    colOffset = gridSize - np.round(gridSize * colFrac) - 1
    rowOffset = gridSize - np.round(gridSize * rowFrac) - 1

    return int(colOffset), int(rowOffset)


def getRegSampledPrfFitsByOffset(prfArray, colOffset, rowOffset):
    """
    The 13x13 pixel PRFs on at each grid location are sampled at a 9x9 intra-pixel grid, to
    describe how the PRF changes as the star moves by a fraction of a pixel in row or column.
    To extract out a single PRF, you need to address the 117x117 array in a funny way
    (117 = 13x9). Essentially you need to pull out every 9th element in the array, i.e.

    .. code-block:: python

        img = array[ [colOffset, colOffset+9, colOffset+18, ...],
                     [rowOffset, rowOffset+9, ...] ]

    Inputs
    ------
    prfArray
        117x117 interleaved PRF array
    colOffset, rowOffset
        The offset used to address the column and row in the interleaved PRF

    Returns
    ------
    prf
        13x13 PRF image for the specified column and row offset

    """
    gridSize = 9

    assert colOffset < gridSize
    assert rowOffset < gridSize

    # Number of pixels in regularly sampled PRF. Should be 13x13
    nColOut, nRowOut = prfArray.shape
    nColOut /= float(gridSize)
    nRowOut /= float(gridSize)

    iCol = colOffset + (np.arange(nColOut) * gridSize).astype(np.int)
    iRow = rowOffset + (np.arange(nRowOut) * gridSize).astype(np.int)

    tmp = prfArray[iRow, :]
    prf = tmp[:,iCol]

    return prf


def interpolatePrf(regPrfArray, col, row, imagePos):
    """
    Interpolate between 4 images to find the best PRF at the specified column and row.
    This is a simple linear interpolation.

    Inputs
    -------
    regPrfArray
        13x13x4 prf image array of the four nearby locations.

    col and row
        (float) the location to interpolate to.

    imagePos
        (list) 4 floating point (col, row) locations

    Returns
    ----
    Single interpolated PRF image.
    """
    p11, p21, p12, p22 = regPrfArray
    c0 = imagePos[0][0]
    c1 = imagePos[1][0]
    r0 = imagePos[0][1]
    r1 = imagePos[2][1]

    assert c0 != c1
    assert r0 != r1

    dCol = (col-c0) / (c1-c0)
    dRow = (row-r0) / (r1 - r0)

    # Intpolate across the rows
    tmp1 = p11 + (p21 - p11) * dCol
    tmp2 = p12 + (p22 - p12) * dCol

    # Interpolate across the columns
    out = tmp1 + (tmp2-tmp1) * dRow
    return out


def getNearestPrfFits(col, row, ccd, camera, sector, path):
    """
    Main Function
    Return a 13x13 PRF image for a single location. No interpolation

    This function is identical to getPrfAtColRowFits except it does not perform the interpolation step.

    Inputs
    ---------
    col, row
        (floats) Location on CCD to lookup. The origin of the CCD is the bottom left.
        Increasing column increases the "x-direction", and row increases the "y-direction"
        The column coordinate system starts at column 45.
    ccd
        (int) CCD number. There are 4 CCDs per camera
    camera
        (int) Camera number. The instrument has 4 cameras
    sector
        (int)  Sector number, greater than or equal to 1.

    Returns
    ---------
    A 13x13 numpy image array of the nearest PRF to the specifed column and row.
    """
    col = float(col)
    row = float(row)
    prfImages = []

    # Determine a datestring in the file name and the path based on ccd/camer/sector
    datestring, addPath = pathLookup(ccd, camera, sector)
    path = path + addPath

    # Convert the fractional pixels to the offset required for the interleaved pixels.
    colOffset, rowOffset = getOffsetsFromPixelFractions(col, row)

    # Determine the 4 (col,row) locations with exact PRF measurements.
    imagePos = determineFourClosestPrfLoc(col, row)
    bestPos = imagePos[0]
    prfArray = readOnePrfFitsFile(ccd, camera, bestPos[0], bestPos[1], path, datestring)

    prfImage = getRegSampledPrfFitsByOffset(prfArray, colOffset, rowOffset)

    return prfArray, prfImage


def getPrfAtColRowFits(col, row, ccd, camera, sector, path):
    """
    Main Function
    Lookup a 13x13 PRF image for a single location

    Inputs
    ---------
    col, row
        (floats) Location on CCD to lookup. The origin of the CCD is the bottom left.
        Increasing column increases the "x-direction", and row increases the "y-direction"
        The column coordinate system starts at column 45.
    ccd
        (int) CCD number. There are 4 CCDs per camera
    camera
        (int) Camera number. The instrument has 4 cameras
    sector
        (int)  Sector number, greater than or equal to 1.
    path
        (str) Directory or URL where the PRF fits files are located

    Returns
    ---------
    A 13x13 numpy image array of the interpolated PRF.
    """
    col = float(col)
    row = float(row)
    prfImages = []

    # Determine a datestring in the file name and the path based on ccd/camera/sector
    datestring, subDirectory = pathLookup(ccd, camera, sector)
    path = path + subDirectory

    # Convert the fractional pixels to the offset required for the interleaved pixels.
    colOffset, rowOffset = getOffsetsFromPixelFractions(col, row)

    # Determine the 4 (col,row) locations with exact PRF measurements.
    imagePos = determineFourClosestPrfLoc(col, row)

    # Loop over the 4 locations and read in each file and extract the sub-pixel location.
    for pos in imagePos:
            prfArray = readOnePrfFitsFile(ccd, camera, pos[0], pos[1], path, datestring)

            #img = getRegSampledPrfFitsByOffset(prfArray, colOffset, rowOffset)
            prfImages.append(prfArray)

    # Simple linear interpolate across the 4 locations.
    interpolatedPrf = interpolatePrf(prfImages, col, row, imagePos)

    return interpolatedPrf


def move_prf(prf, xshift, yshift, npix=13.):

    x = np.linspace(0., npix, 117)
    y = np.linspace(0., npix, 117)
    X, Y = np.meshgrid(x, y)

    # Interpolate the prf onto a 13x13 pixel grid
    interp_spline = RectBivariateSpline(y, x, prf)

    # Then shift the prf to the centroid of the star.
    dx2, dy2 = 0.01, 0.01
    x2 = np.arange(0.-xshift, npix-xshift, dx2)
    y2 = np.arange(0.-yshift, npix-yshift, dy2)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = interp_spline(y2, x2)

    return np.sum(np.reshape(Z2, (npix,100,npix,100)), axis=(1,3))/100/100


def inject_signal(ticid, period, amplitude, baseline, tesscut_path,
                  upper_sector_limit, xpix=68, ypix=68):

    sectors, star = get_sectors(ticid, upper_sector_limit=upper_sector_limit)

    # Eleanor object
    print("Finding Eleanor object...")
    time, inds, max_inds = [], [], 0
    elstar = []
    for sector in sectors:
        print("sector", sector)

        star = eleanor.Source(tic=ticid, sector=int(sector), tc=True)
        sec, camera, ccd = star.sector, star.camera, star.chip
        colrowpix = star.position_on_chip
        elstar.append(star)

        fits_image_filename = get_fits_filenames(tesscut_path, sec, camera,
                                                 ccd, star.coords[0],
                                                 star.coords[1])


        # Load the TESScut FFI data (get time array)
        hdul = fits.open(fits_image_filename)
        postcard = hdul[1].data
        t = postcard["TIME"]*1.
        flux = postcard["FLUX"]*1.

        time.append(t)

    # Create the multi-sector signal
    time_array = np.array([i for j in time for i in j])
    signal = baseline + get_random_light_curve(time_array, period, amplitude)

    for i, sector in enumerate(sectors):

        print("sector", sector)
        star = elstar[i]
        sec, camera, ccd = star.sector, star.camera, star.chip
        colrowpix = star.position_on_chip

        fits_image_filename = get_fits_filenames(tesscut_path, sec, camera,
                                                 ccd, star.coords[0],
                                                 star.coords[1])
        path_to_tesscut = "{0}/astrocut_{1:12}_{2:13}_{3}x{4}px".format(
            tesscut_path, star.coords[0], star.coords[1], xpix, ypix)
        inj_dir = "{0}/injected".format(path_to_tesscut)
        injection_filename = "{0}/tess-s{1}-{2}-{3}_{4:.6f}_{5:.6f}_{6}x{7}_astrocut.fits".format(
            inj_dir, str(int(sector)).zfill(4), camera, ccd, star.coords[0],
            star.coords[1], xpix, ypix)

        mask = (time[i][0] <= time_array) & (time_array <= time[i][-1])
        assert len(signal[mask]) == len(time[i])
        inject_one_sector_starry(ticid, sector, signal[mask], camera, ccd,
                                 colrowpix, fits_image_filename,
                                 injection_filename)

    return time_array, signal


def inject_one_sector_starry(ticid, sector, signal, camera, ccd,
                             colrowpix, fits_image_filename,
                             injection_filename, offset_x=0., offset_y=0.):

    # Load the TESScut FFI data
    print("Loading TESScut FFI...")
    hdul = fits.open(fits_image_filename)
    postcard = hdul[1].data
    time = postcard["TIME"]*1.
    flux = postcard["FLUX"]*1.

    # Get the PRF
    print("Fetching PRF...")
    path = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/"
    prf = getPrfAtColRowFits(colrowpix[0], colrowpix[1], ccd, camera, sector,
                             path)  # col, row, ccd, camera, sector

    # Inject the signal and save the new file.
    print("Injecting signal and saving...")
    injected_flux = flux + signal[:, None, None] * move_prf(
        prf, offset_x, offset_y, npix=68)[None, :, :]
    hdul[1].data["FLUX"] = injected_flux
    if os.path.exists(injection_filename):
        os.remove(injection_filename)
    hdul.writeto(injection_filename)
    hdul.close()
    return time, signal


def inject_one_sector(ticid, sector, period, amplitude, baseline, sec,
                      camera, ccd, colrowpix, fits_image_filename,
                      injection_filename, offset_x=0., offset_y=0.,
                      signal="starry"):

    # Load the TESScut FFI data
    print("Loading TESScut FFI...")
    hdul = fits.open(fits_image_filename)
    postcard = hdul[1].data
    time = postcard["TIME"]*1.
    flux = postcard["FLUX"]*1.

    # Get the PRF
    print("Fetching PRF...")
    path = "https://archive.stsci.edu/missions/tess/models/prf_fitsfiles/"
    prf = getPrfAtColRowFits(colrowpix[0], colrowpix[1], ccd, camera, sec,
                             path)  # col, row, ccd, camera, sector

    # Simulate the signal
    if signal == "sinusoid":
        signal = baseline + amplitude*np.sin(time*2*np.pi/period)
    elif signal == "starry":
        signal = baseline + get_random_light_curve(time, period, amplitude)

    # Inject the signal and save the new file.
    print("Injecting signal and saving...")
    injected_flux = flux + signal[:, None, None] * move_prf(
        prf, offset_x, offset_y, npix=68)[None, :, :]
    hdul[1].data["FLUX"] = injected_flux
    if os.path.exists(injection_filename):
        os.remove(injection_filename)
    hdul.writeto(injection_filename)
    hdul.close()
    return time, signal


# Define our spatial power spectrum
def power(l, amp=1e-1):
    return amp * np.exp(-((l / 10) ** 2))


def get_random_light_curve(t, p, a, inclination=90., seed=None):

    if seed is not None:
        np.random.seed(seed)

    starry.config.lazy = False

    # Instantiate a 10th degree starry map
    map = starry.Map(10)

    # Random inclination (isotropically distributed ang. mom. vector)
    if inclination == "random":
        map.inc = np.arccos(np.random.random()) * 180 / np.pi
    else:
        map.inc = inclination

    # Random period, U[1, 30]
    # p = 1 + 29 * np.random.random()

    # Random surface map
    for l in range(1, map.ydeg + 1):
        map[l, :] = np.random.randn(2 * l + 1) * power(l, amp=a) / (2 * l + 1)

    # Compute the flux
    flux = map.flux(theta=360.0 * t / p)

    # Median-correct it
    flux -= np.median(flux)
    flux += 1

    return flux
