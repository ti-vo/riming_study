import datetime
import numpy as np
import sys

sys.path.append('../')
import functions
import functions.radar_functions as rf
from scipy.io import loadmat
import matplotlib.pyplot as plt

larda = functions.pyLARDA.LARDA().connect('arm_baecc', build_lists=True)

# 1 ) reading in in-situ data
#in_situ_data = '/home/tvogl/PhD/conferences_workshops/201909_Bonn_Juelich/Exercises/Moisseev/Snow_retr_2014_2015_for_Python.mat'
in_situ_data = '/media/sdig/arm-iop/2014/baecc/in-situ/Snow_retr_2014_2015_for_Python.mat'
PIP_data = loadmat(in_situ_data, mat_dtype=True)
rmf_pip, piptime = rf.rimed_mass_fraction_PIP(PIP_data)

# 2) identify time chunks in in-situ data
jumps = np.where(np.diff(piptime) > 3600)[0]
i = 0
while piptime[jumps[i]] < functions.h.dt_to_ts(datetime.datetime(2014, 3, 31, 0, 0)):
    if i == 0:
        t_sta = datetime.datetime(2014, 2, 1, 2, 0)
    else:
        t_sta = functions.h.ts_to_dt(piptime[jumps[i - 1] + 1])
    t_end = functions.h.ts_to_dt(piptime[jumps[i]])
    i += 1

    # 3) read in the data for all of the chunks
    print(f'time interval from {t_sta.strftime("%Y%m%d %H:%M")} to {t_end.strftime("%Y%m%d %H:%M")} \n')
    # cloud-base height
    cbh = larda.read("CEILO", "cbh", [t_sta, t_end])

    # read in radar data (for now only moments)
    Zg = rf.remove_broken_timestamp_arm(larda.read("KAZR", "Ze", [t_sta, t_end], [h_sta, h_end]))
    MDV = rf.remove_broken_timestamp_arm(larda.read("KAZR", "mdv", [t_sta, t_end], [h_sta, h_end]))
    sw = rf.remove_broken_timestamp_arm(larda.read("KAZR", "sw", [t_sta, t_end], [h_sta, h_end]))
    # read in signal-to-noise ratio for masking non-cloudy pixels
    snr = rf.remove_broken_timestamp_arm(larda.read("KAZR", "snr", [t_sta, t_end], [h_sta, h_end]))

    np.putmask(sw['var'], snr['var'] < 0, np.nan)
    np.putmask(MDV['var'], snr['var'] < 0, np.nan)
    np.putmask(Zg['var'], snr['var'] < 0, np.nan)

    Zg['var'] = functions.h.lin2z(Zg['var'])

    cbh = functions.pyLARDA.Transformations.interpolate1d(cbh, new_time=Zg['ts'])
    cbh_mask = np.vstack([MDV['rg'] > a for a in list(cbh['var'])])
    #    print(f'shape of cbh_mask, i = {i}: {cbh_mask.shape}')
    #    print(f'shape of MDV, i = {i}: {MDV["var"].shape}')
    #    print(f'dimlabel MDV: {MDV["dimlabel"]}')
    cbh_mask = functions.h.put_in_container(cbh_mask, MDV, name='cloud base height mask')

    if i == 1:
        Zg_all = Zg
        MDV_all = MDV
        sw_all = sw
        cbh_all = cbh
        cbh_mask_all = cbh_mask
    else:
        Zg_all = functions.pyLARDA.Transformations.join(Zg_all, Zg)
        MDV_all = functions.pyLARDA.Transformations.join(MDV_all, MDV)
        sw_all = functions.pyLARDA.Transformations.join(sw_all, sw)
        cbh_all = functions.pyLARDA.Transformations.join(cbh_all, cbh)
        cbh_mask_all = functions.pyLARDA.Transformations.join(cbh_mask_all, cbh_mask)



plot_dir = '../plots/BAECC_scatter_not_masked/'
fig, ax = plt.hist(Zg_all)
fig.savefig(plot_dir + 'histogram_Zg.png')

# below cloud base height
plot_dir = '../plots/BAECC_scatter_below_cbh/'
import copy
Zg_all_below = copy.deepcopy(Zg_all)
np.putmask(Zg_all_below['var'], cbh_mask_all['var'], np.nan)
MDV_all_below = copy.deepcopy(MDV_all)
np.putmask(MDV_all_below['var'], cbh_mask_all['var'], np.nan)
sw_all_below = copy.deepcopy(sw_all)
np.putmask(sw_all_below['var'], cbh_mask_all['var'], np.nan)

fig, ax = plt.hist(Zg_all_below, bins='auto')
fig.savefig(plot_dir + 'histogram_Zg.png')

# above cloud base height
plot_dir = '../plots/BAECC_scatter_above_cbh/'
cbh_mask_all['var'] = ~cbh_mask_all['var']
Zg_all_above = copy.deepcopy(Zg_all)
np.putmask(Zg_all_above['var'], cbh_mask_all['var'], np.nan)
MDV_all_above = copy.deepcopy(MDV_all)
np.putmask(MDV_all_above['var'], cbh_mask_all['var'], np.nan)

fig, ax = plt.hist(Zg_all_above, bins='auto')
fig.savefig(plot_dir + 'histogram_Zg.png')

