import datetime
import numpy as np
import sys
sys.path.append('../')
import netCDF4
import functions
import copy
import functions.radar_functions as rf
import toml
from scipy.io import loadmat
import functools
import matplotlib
matplotlib.use('Qt4Agg')

config_case_studies = toml.load('../casestudies_toml/pamtra_files.toml')
with netCDF4.Dataset(config_case_studies['file']['high_noise']['path'], 'r') as ncD:
    spectra = ncD.variables['Radar_Spectrum'][:, :, 0, 1, 0, :] # 1 is Ka band
    ranges = ncD.variables['height'][:]

larda = functions.pyLARDA.LARDA().connect('arm_baecc', build_lists=True)
h_sta = 550
t_sta = datetime.datetime(2014, 2, 21, 22, 30)
t_end = datetime.datetime(2014, 2, 21, 22, 34)
h_end = 2000

Zg = rf.remove_broken_timestamp_arm(larda.read("KAZR", "Ze", [t_sta, t_end], [h_sta, h_end]))
spectra_KAZR = larda.read("KAZR", "spec", [t_sta, t_end], [h_sta, h_end])

Zg = functions.pyLARDA.Transformations.interpolate2d(Zg, new_time=spectra_KAZR['ts'])
noise_spectra = copy.deepcopy(spectra_KAZR)
denoised_spectra = rf.denoise_spectra(spectra_KAZR)
kazr_denoised_spectra = functions.h.put_in_container(denoised_spectra, spectra_KAZR)
moments = rf.skewness_from_denoised(kazr_denoised_spectra)

Z_from_spectra = functions.h.put_in_container(moments['Ze'], Zg)

fig, ax = functions.pyLARDA.Transformations.plot_timeheight(Zg, title='regular moments', z_converter='lin2z')
fig, ax = functions.pyLARDA.Transformations.plot_timeheight(Z_from_spectra, title='moments from spectra', z_converter='lin2z')

#skewness, ts_skewness = rf.read_apply("KAZR", "spec", [t_sta, t_end], [h_sta, h_end], rf.denoise_and_get_skewness, larda=larda)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
ax.hist(functions.h.lin2z(Zg['var']))

spectrum = functions.pyLARDA.Transformations.slice_container(noise_spectra, index={'time': [120]},
                                                                            value={'range': [np.ma.getdata(ranges[0])]})

fig, ax = functions.pyLARDA.Transformations.plot_spectra(spectrum, z_converter='lin2z')
ax.set_ylim([-70., 10])

spectrum_PAMTRA = functions.h.put_in_container(spectra[9,:,:], spectrum, system='PAMTRA')
fig, ax = functions.pyLARDA.Transformations.plot_spectra(spectrum_PAMTRA)
ax.set_ylim([-70., 10])

