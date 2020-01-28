import numpy as np
import math
import pyLARDA
import datetime
import pyLARDA.helpers as h
from itertools import groupby

def rolling_window(a, window):
    """
    Args:
        a: 1-d variable over which moving window should be applied
        window: (integer) length of window to be applied (elements of a)

    Returns: array with window-sized number of columns, containing subset of a over which function can be applied

    Examples:
        >>> rolling_window(np.asarray([1, 2, 3]), 2)
        array([[1, 2],
               [2, 3]])

    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def air_density_correction(MDV_container, pressure="standard", temperature="standard"):
    """"correcting fall velocities with respect to air density.
    Per default, the standard atmosphere is used for correction, but for pressure, also a vector containing
    p values can be supplied.
    Args
       MDV_container: larda data container, "var" is array of fall velocities to be corrected and "rg" is range in m
        **pressure: if set to "standard" use standard atmospheric pressure gradient to correct MDV
         """
    g = 9.80665  # gravitational acceleration
    R = 287.058  # specific gas constant of dry air

    def get_density(pressure, temperature):
        R = 287.058
        density = pressure / (R * temperature)
        return density

    def cal(p0, t0, L, h0, h1):
        if L != 0:
            t1 = t0 + L * (h1 - h0)
            p1 = p0 * (t1 / t0) ** (-g / L / R)
        else:
            t1 = t0
            p1 = p0 * math.exp(-g / R / t0 * (h1 - h0))
        return t1, p1

    def isa(altitude):
        """international standard atmosphere
        numbers from https://en.wikipedia.org/wiki/International_Standard_Atmosphere
        code: https://gist.github.com/buzzerrookie/5b6438c603eabf13d07e"""

        L = [-0.0065, 0]  # Lapse rate in troposphere, tropopause
        h = [11000, 20000]  # height of troposphere, tropopause
        p0 = 108900  # base pressure
        t0 = 292.15  # base temperature, 19 degree celsius
        prevh = -611
        if altitude < 0 or altitude > 20000:
            AssertionError("altitude must be in [0, 20000]")
        for i in range(0, 2):
            if altitude <= h[i]:
                temperature, pressure = cal(p0, t0, L[i], prevh, altitude)
            else:
                t0, p0 = cal(p0, t0, L[i], prevh, h[i])
                prevh = h[i]
        density = get_density(pressure, temperature)
        return pressure, density, temperature

    if pressure == "standard" and temperature == "standard":
        d = [isa(range)[1] for range in list(MDV_container['rg'])]

    elif pressure == "standard" and temperature != "standard":
        p = [isa(range)[0] for range in list(MDV_container['rg'])]
        t = temperature
        d = [get_density(pi, ti) for pi, ti in (p, t)]

    elif temperature == "standard" and pressure != "standard":
        t = [isa(range)[2] for range in list(MDV_container['rg'])]
        p = pressure
        d = [get_density(pi, ti) for pi, ti in (p, t)]

    # Vogel and Fabry, 2018, Eq. 1
    rho_0 = 1.2  # kg per m^3
    corr_fac = [np.sqrt(di / rho_0) for di in d]
    MDV_corr = MDV_container['var'] * np.asarray(corr_fac)

    return MDV_corr


def compute_width(spectra, **kwargs):
    """
    wrapper for width_fast; compute Doppler spectrum edge width
    Args:
        spectra: larda dictionary of spectra

    Returns:

    """
    widths_array = width_fast(spectra['var'], **kwargs)
    vel_res = abs(np.median(np.diff(spectra['vel'])))
    width = vel_res*widths_array

    return width


def width_fast(spectra, **kwargs):
    """
    implementation without loops; compute Doppler spectrum edge width

    Args:
        spectra (np.ndarray): 3D array of time, height, velocity

    Returns:
        2D array of widths (measured in number of bins, has to be multiplied by Doppler velocity resolution)

    """
    # define the threshold above which edges are found as the larger one of either
    # 0.05% of the peak reflectivity, or the minimum of the spectrum + 6 dBZ
    thresh_1 = 6 if not 'thresh1' in kwargs else kwargs['thresh1']
    thresh_2 = 0.0005 if not 'thresh2' in kwargs else kwargs['thresh2']
    bin_number = spectra.shape[2]

    thresh = np.maximum(thresh_2*np.nanmax(spectra, axis=2), np.nanmin(spectra, axis=2) * 10**(thresh_1/10))

    # find the first bin where spectrum is larger than threshold
    first = np.argmax(spectra > np.repeat(thresh[:, :, np.newaxis], bin_number, axis=2), axis=2)
    # same with reversed view of spectra
    last = bin_number - np.argmax(spectra[:, :, ::-1] > np.repeat(thresh[:, :, np.newaxis], bin_number, axis=2),
                                  axis=2)

    return last-first


def denoise_and_compute_width(spectra, **kwargs):
    """
    combines noise floor removal and spectrum width computation. This is needed e.g. when many spectra are read in,
    the width is computed and spectra are deleted from memory.

    Args:
        spectra: larda dictionary of spectra

    Returns:
        spectrum edge width (2D array)

    """
    spec_denoise = spectra['var']
    for t in range(len(spectra['ts'])):
        for hi in range(len(spectra['rg'])):
            spectrum = spectra['var'][t, hi, :]
            mean, thresh, var, _, _, _, _ = pyLARDA.spec2mom_limrad94.estimate_noise_hs74(spectrum)
            noise_mask = (spectrum <= (mean + 5 * var))
            np.putmask(spectrum, noise_mask, np.nan)
            spec_denoise[t, hi, :] = spectrum
    spectra['var'] = spec_denoise
    width = compute_width(spectra, **kwargs)

    return width


def read_apply(system, variable, time, rg, function, timedelta=datetime.timedelta(hours=1), **kwargs):
    """
    Do you get a memory error because to try to read in lots of spectra and apply the same function to them?
    This function allows to read them in using larda.read, chunk by chunk, and apply the function.
    It will return the result as a concatenated numpy array.

    Args:
        system (str): name of system to be passed to larda.read, e.g. "MIRA"
        variable (str): name of variable to be passed to larda.read, e.g. "Zg"
        time (list): list of datetime start, datetime end
        rg (list): list of range start, range end
        function (function): function to be applied to the read in data chun
        timedelta (datetime.timedelta): to define chunk size of to-be read in variable
        **larda: instance for reading in data, created with larda = pyLARDA.larda(...).connect(...)

    Returns:
        outp (ndarray): concatenated function results
        ts (ndarray): timestamp

    """
    if 'larda' in kwargs:
        larda = kwargs['larda']
    else:
        print("larda is not defined. This will crash... ")
    # thanks stackoverflow
    timelist = np.arange(time[0], time[1], timedelta).astype(datetime.datetime)
    timelist = np.hstack([timelist, time[1]])
    outp = []
    ts = []
    for i in range(len(timelist) - 1):
        # TODO remove print statements and put it into a logging functionality
        print(f'reading chunk from {timelist[i].strftime("%Y%m%d %H:%M")} to {timelist[i+1].strftime("%Y%m%d %H:%M")}')
        # call larda.read for each time interval
        try:
            # TODO place this function somewhere smart in the larda structure (within class LARDA?)
            #  so that larda is definitely defined
            # read in chunks of data with a 5 second overlap
            inp = larda.read(system, variable, [timelist[i], timelist[i + 1] + datetime.timedelta(seconds=5)], rg)
            # this leads to double entries.
            print('apply function...')
            result = function(inp)
            outp.append(result)
            ts.append(inp['ts'])
        except TypeError:
            print(f"File not found or function doesn't work for {time} - {timelist[i + 1]}")

    # concatenate all the list entries of outp and ts
    outp = np.vstack(outp)
    ts = np.hstack(ts)
    # check for double entries and remove them
    ts, idx = np.unique(ts, return_index=True)
    outp = outp[idx, :]
    # check boundaries
    idx_b = np.logical_and(ts < h.dt_to_ts(time[1]), ts > h.dt_to_ts(time[0]))
    outp = outp[idx_b, :]
    ts = ts[idx_b]

    return outp, ts


def get_target_classification(categorize_bits):
    """
    Function copied from cloudnetpy to get classification from cloudnet categorize bits given in
     lv1 netcdf files
    :param categorize_bits: dictionary containing category and quality bits (ndarrays)
    :return: classification
    """
    bits = categorize_bits["category_bits"]['var']
    clutter = categorize_bits["quality_bits"]['var'] == 2
    classification = np.zeros(bits.shape, dtype=int)

    classification[(bits == 1)] = 1 # 1 liquid cloud droplets only
    classification[(bits == 2)] = 2 # isbit(2, 1) drizzle or rain - falling bit is bit 1
    classification[bits == 3] = 3 # 0+1 isbit(3, 1) and isbit(3, 0) bits 0 and 1, 3 drizzle and liquid cloud
    classification[bits == 6] = 4 # ice (falling and cold) bits 1 and 2: 2+4=6
    classification[bits == 7] = 5 # 0,1,2
    classification[(bits == 8)] = 6 #3 only, melting ice
    classification[(bits == 9)] = 7 # 3 and 0 melting ice and droplets
    classification[(bits == 16)] = 8 # bit 4 is 16 this means it's class 8, so confusing: aerosol
    classification[(bits == 32) & ~clutter] = 9 # bit 5 is 32: insects
    classification[(bits == 48) & ~clutter] = 10 # 4 and 5 = 48: insects and aerosol
    classification[clutter & (~(bits == 16))] = 0 # clutter and no bit 4 (=16, aerosol)

    return classification


def isbit(integer, nth_bit):
    """copied from cloudnetpy. Tests if nth bit (0,1,2..) is on for the input number.
    Args:
        integer (int): A number.
        nth_bit (int): Investigated bit.
    Returns:
        bool: True if set, otherwise False.
    Raises:
        ValueError: negative bit as input.
    Examples:
        >>> isbit(4, 1)
            False
        >>> isbit(4, 2)
            True
    See also:
        utils.setbit()
    """
    if nth_bit < 0:
        raise ValueError('Negative bit number.')
    mask = 1 << nth_bit
    return integer & mask > 0


def apply_function_timeheight_grid(fun, container, new_range, new_time):
    """
    apply a function, e.g. mean over all elements of a container['var'] belonging to a grid cell in a new time-height
    grid defined by new_range and new_time

    Args:
        fun: Function to be applied over the time-height grid to obtain the new grid
        container: dictionary containing "rg" (1D), "ts" (1D), "mask" (2D) and "var" (2D)
        new_range: 1D array with new range axis, one element more than will be returned
        new_time: 1D array with new time axis, one element longer than what will be returned

    Returns: 2D array containing function output of variable on new time-height grid

    Examples:
        >>> apply_function_timeheight_grid(np.mean, {'var': np.array([[1.,2.,3.,4.], [1., 2., 5., 10.], [1.,2.,18.,800.]]), 'rg': [10, 20, 30, 40], 'ts': [0, 1,2,3], 'mask':np.full([3,4], False)}, new_range=[5, 35], new_time=[2,4])
            array([[  2.33333333,   7.        ],
            [  7.        , 800.        ]])
    """
    # apply mask of container
    np.putmask(container['var'], container['mask'], np.nan)

    h_coords = np.digitize(container['rg'], new_range)
    t_coords = np.digitize(container['ts'], new_time)
    # make zero the smallest index
    h_coords = h_coords - np.min(h_coords)
    t_coords = t_coords - np.min(t_coords)

    # initialize output array
    out_array = np.full([len(np.unique(t_coords)), len(np.unique(h_coords))], np.nan)

    # find time/height ranges to average over
    first_hit_t = np.searchsorted(t_coords, np.arange(0, len(np.unique(t_coords)) + 1))
    first_hit_h = np.searchsorted(h_coords, np.arange(0, len(np.unique(h_coords)) + 1))

    for x in range(out_array.shape[0]):
        for y in range(out_array.shape[1]):
            out_array[x, y] = fun(
                container['var'][first_hit_t[x]:first_hit_t[x + 1],
                first_hit_h[y]:first_hit_h[y + 1]])

    return out_array


def regrid_integer_timeheight(container, new_range, new_time, fraction=0.9):
    """
    integer arrays containing e.g. cloudnet bits cannot be interpolated  easily, so I take the element with the highest
    occurrence in a gridcell in a time-height grid. It has to make up 90% (fraction = 0.9) of the data in there, else
    the pixel is set to nan

    Args:
        container:    dict data container
        new_range:    np.array (1D) defining the new time-height grid
        new_time:     np.array (1D) defining the new time-height grid
        fraction:     number between 0 and 1, minimum fraction of pixel type

    Returns:
        2D array of regridded container['var']

    """

    h_coords = np.digitize(container['rg'], new_range)
    t_coords = np.digitize(container['ts'], new_time)
    # make zero the smallest index
    h_coords = h_coords - np.min(h_coords)
    t_coords = t_coords - np.min(t_coords)
    out_array = np.full([len(np.unique(t_coords)), len(np.unique(h_coords))], np.nan)

    # find time/height ranges to average over
    first_hit_t = np.searchsorted(t_coords, np.arange(0, len(np.unique(t_coords)) + 1))
    first_hit_h = np.searchsorted(h_coords, np.arange(0, len(np.unique(h_coords)) + 1))

    for x in range(out_array.shape[0]):
        for y in range(out_array.shape[1]):
            a = container['var'][first_hit_t[x]:first_hit_t[x+1],
                first_hit_h[y]:first_hit_h[y + 1]]
            foo  = [len(list(group)) for key, group in groupby(list(a.flatten()))]
            key = [key for key, group in groupby(list(a.flatten()))]
            index = np.argsort(foo)

            if len(index > 0):
                if foo[index[-1]] >= fraction*len(a.flatten()):
                    out_array[x, y] = key[index[-1]]

    return out_array

