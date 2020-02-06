import os
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

larda = functions.pyLARDA.LARDA().connect('arm_baecc', build_lists=True)
#t_sta = datetime.datetime(2014, 2, 21, 22, 30)
#t_end = datetime.datetime(2014, 2, 21, 22, 34)
h_sta = 0
h_end = 2000

# 1 ) reading in in-situ data and computing rime mass fraction
in_situ_data = '/home/tvogl/PhD/conferences_workshops/201909_Bonn_Juelich/Exercises/Moisseev/Snow_retr_2014_2015_for_Python.mat'
PIP_data = loadmat(in_situ_data, mat_dtype=True)
rmf_pip, piptime = rf.rimed_mass_fraction_PIP(PIP_data)
rmf_pip = {'var': rmf_pip, 'ts': piptime, 'name': 'rime mass fraction', 'dimlabel': ['time'],
           'mask': np.isnan(rmf_pip), 'var_lims': [0, 1], 'system': 'PIP', 'var_unit': 'unitless'}

# 2) identify time chunks in in-situ data

jumps = np.where(np.diff(piptime) > 3600)[0]
i = 0
while jumps[i] < functions.h.dt_to_ts(datetime.datetime(2014, 3, 31, 0, 0)):
    if i == 0:
        t_sta = functions.h.ts_to_dt(piptime[0])
    else:
        t_sta = functions.h.ts_to_dt(piptime[jumps[i - 1] + 1])
    t_end = functions.h.ts_to_dt(piptime[jumps[i]])
    i += 1
    # 3) read in the data for all of the chunks

    # cloud-base height
    cbh = larda.read("CEILO", "cbh", [t_sta, t_end])

    # read in radar data (moments and spectra)
    Zg = rf.remove_broken_timestamp_arm(larda.read("KAZR", "Ze", [t_sta, t_end], [h_sta, h_end]))
    MDV = rf.remove_broken_timestamp_arm(larda.read("KAZR", "mdv", [t_sta, t_end], [h_sta, h_end]))
    sw = rf.remove_broken_timestamp_arm(larda.read("KAZR", "sw", [t_sta, t_end], [h_sta, h_end]))

    Zg['var'] = functions.h.lin2z(Zg['var'])
    widths, ts_widths = rf.read_apply("KAZR", "spec", [t_sta, t_end], [h_sta, h_end], rf.denoise_and_compute_width,
                                      larda=larda)
    rg_widths = larda.read("KAZR", "spec", [t_sta], [h_sta, h_end])['rg']
    widths = functions.h.put_in_container(widths, MDV, ts=ts_widths, rg=rg_widths, mask=widths < 0, name="spec edge width")

    # regrid, mask all above cbh, join together
    # to be sure, regrid all to the same time-height grid
    widths = functions.pyLARDA.Transformations.interpolate2d(widths, new_time=Zg['ts'], new_range=Zg['rg'],
                                                             method='nearest')
    cbh = functions.pyLARDA.Transformations.interpolate1d(cbh, new_time=Zg['ts'])
    cbh_mask = np.vstack([widths['rg'] > a for a in list(cbh['var'])]).shape
    rf = functions.pyLARDA.Transformations.interpolate1d(rmf_pip, new_time=Zg['ts'])

    if i == 1:
        Zg_all = Zg
        MDV_all = MDV
        sw_all = sw
        cbh_all = cbh
        cbh_mask_all = cbh_mask
        rf_all = rf
        widths_all = widths
    else:
        Zg_all = functions.pyLARDA.Transformations.join(Zg_all, Zg)
        MDV_all = functions.pyLARDA.Transformations.join(MDV_all, MDV)
        sw_all = functions.pyLARDA.Transformations.join(sw_all, sw)
        cbh_all = functions.pyLARDA.Transformations.join(cbh_all, cbh)
        cbh_mask_all = functions.pyLARDA.Transformations.join(cbh_mask_all, cbh_mask)
        rf_all = functions.pyLARDA.Transformations.join(rf_all, rf)
        widths_all = functions.pyLARDA.Transformations.join(widths_all, widths)


# scatter plots
plot_dir = '../plots/BAECC_scatter/'
fig, ax = functions.pyLARDA.Transformations.plot_scatter(Zg_all, MDV_all, identity_line=False, colorbar=True, title=True)
fig.savefig(plot_dir + f'testplot.png')