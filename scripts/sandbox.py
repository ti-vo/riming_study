import datetime
import numpy as np
import sys
sys.path.append('../')
import functions
import functions.radar_functions as rf
import toml
from scipy.io import loadmat
import functools
import matplotlib
matplotlib.use('Qt4Agg')


larda = functions.pyLARDA.LARDA().connect('arm_baecc', build_lists=True)
h_sta = 550
t_sta = datetime.datetime(2014, 2, 21, 22, 30)
t_end = datetime.datetime(2014, 2, 21, 22, 34)
h_end = 2000

Zg = rf.remove_broken_timestamp_arm(larda.read("KAZR", "Ze", [t_sta, t_end], [h_sta, h_end]))
spectra = larda.read("KAZR", "spec", [t_sta, t_end], [h_sta, h_end])

Zg = functions.pyLARDA.Transformations.interpolate2d(Zg, new_time=spectra['ts'])
denoised_spectra = rf.denoise_spectra(spectra)
spectra['var'] = denoised_spectra
moments = rf.skewness_from_denoised(spectra)

Z_from_spectra = functions.h.put_in_container(moments['Ze'], Zg)

fig, ax = functions.pyLARDA.Transformations.plot_timeheight(Zg, title='regular moments', z_converter='lin2z')
fig, ax = functions.pyLARDA.Transformations.plot_timeheight(Z_from_spectra, title='moments from spectra', z_converter='lin2z')

skewness, ts_skewness = rf.read_apply("KAZR", "spec", [t_sta, t_end], [h_sta, h_end], rf.denoise_and_get_skewness, larda=larda)
