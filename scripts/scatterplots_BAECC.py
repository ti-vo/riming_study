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
h_sta = 550
h_end = 2000
compute_widths = True
# 1 ) reading in in-situ data and computing rime mass fraction
in_situ_data = '/media/sdig/arm-iop/2014/baecc/in-situ/Snow_retr_2014_2015_for_Python.mat'
PIP_data = loadmat(in_situ_data, mat_dtype=True)
rmf_pip, piptime = rf.rimed_mass_fraction_PIP(PIP_data)
rmf_pip = {'var': rmf_pip, 'ts': piptime, 'name': 'rime mass fraction', 'dimlabel': ['time'],
           'mask': np.isnan(rmf_pip), 'var_lims': [0, 1], 'system': 'PIP', 'var_unit': 'unitless',
           'filename': in_situ_data, 'file_history': '', 'paraminfo':{'location':'Hyytiälä'}}

# 2) identify time chunks in in-situ data

jumps = np.where(np.diff(piptime) > 3600)[0]
i = 0
while piptime[ jumps[i]] < functions.h.dt_to_ts(datetime.datetime(2014, 3, 31, 0, 0)):
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

    # read in radar data (moments and spectra)
    Zg = rf.remove_broken_timestamp_arm(larda.read("KAZR", "Ze", [t_sta, t_end], [h_sta, h_end]))
    MDV = rf.remove_broken_timestamp_arm(larda.read("KAZR", "mdv", [t_sta, t_end], [h_sta, h_end]))
    sw = rf.remove_broken_timestamp_arm(larda.read("KAZR", "sw", [t_sta, t_end], [h_sta, h_end]))
    np.putmask(sw['var'], abs(sw['var']) > 10, np.nan)
    Zg['var'] = functions.h.lin2z(Zg['var'])
    if compute_widths:
        widths, ts_widths = rf.read_apply("KAZR", "spec", [t_sta, t_end], [h_sta, h_end], rf.denoise_and_compute_width,
                                          larda=larda)
        rg_widths = larda.read("KAZR", "spec", [t_sta], [h_sta, h_end])['rg']
        widths = functions.h.put_in_container(widths, MDV, ts=ts_widths, rg=rg_widths, mask=widths < 0, name="spec edge width")

        # regrid, mask all above cbh, join together
        # to be sure, regrid all to the same time-height grid
        widths = functions.pyLARDA.Transformations.interpolate2d(widths, new_time=Zg['ts'], new_range=Zg['rg'],
                                                                 method='nearest')
    cbh = functions.pyLARDA.Transformations.interpolate1d(cbh, new_time=Zg['ts'])
    cbh_mask = np.vstack([MDV['rg'] > a for a in list(cbh['var'])])
#    print(f'shape of cbh_mask, i = {i}: {cbh_mask.shape}')
#    print(f'shape of MDV, i = {i}: {MDV["var"].shape}')
#    print(f'dimlabel MDV: {MDV["dimlabel"]}')
    cbh_mask = functions.h.put_in_container(cbh_mask, MDV, name='cloud base height mask')
    rmf = functions.pyLARDA.Transformations.interpolate1d(rmf_pip, new_time=Zg['ts'])

    if i == 1:
        Zg_all = Zg
        MDV_all = MDV
        sw_all = sw
        cbh_all = cbh
        cbh_mask_all = cbh_mask
        rmf_all = rmf
        if compute_widths:
            widths_all = widths
    else:
        Zg_all = functions.pyLARDA.Transformations.join(Zg_all, Zg)
        MDV_all = functions.pyLARDA.Transformations.join(MDV_all, MDV)
        sw_all = functions.pyLARDA.Transformations.join(sw_all, sw)
        cbh_all = functions.pyLARDA.Transformations.join(cbh_all, cbh)
        cbh_mask_all = functions.pyLARDA.Transformations.join(cbh_mask_all, cbh_mask)
        rmf_all = functions.pyLARDA.Transformations.join(rmf_all, rmf)
        if compute_widths:    
            widths_all = functions.pyLARDA.Transformations.join(widths_all, widths)

# scatter plots
plot_dir = '../plots/BAECC_scatter_not_masked/'
fig, ax = functions.pyLARDA.Transformations.plot_scatter(Zg_all, MDV_all, identity_line=False, colorbar=True, title=True)
ax.set_xlim([-30, 30])
fig.savefig(plot_dir + f'Ze_MDV.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(sw_all, MDV_all, identity_line=False, colorbar=True, title=True)
#ax.set_xlim([-3, 8])
fig.savefig(plot_dir + f'sw_MDV.png')

rmf_all['var'] = np.tile(rmf_all['var'][:, np.newaxis], MDV_all['var'].shape[1])
rmf_all['rg'] = MDV_all['rg']
rmf_all['mask'] = rmf_all['var'] < 0

if compute_widths:
    fig, ax = functions.pyLARDA.Transformations.plot_scatter(widths_all, MDV_all, identity_line=False, colorbar=True, title=True)
    fig.savefig(plot_dir + f'widths_MDV.png')
    fig, ax = functions.pyLARDA.Transformations.plot_scatter(widths_all, Zg_all, color_by=rmf_all, identity_line=False,
                                                             colorbar=True, scale='lin', title=True)
    ax.set_ylim([-40, 20])
    fig.savefig(plot_dir + 'width_Zg_by_rmf.png')



fig, ax = functions.pyLARDA.Transformations.plot_scatter(rmf_all, MDV_all, identity_line=False, colorbar=True, title=True)
fig.savefig(plot_dir + f'rmf_MDV.png')

# below cloud base height
plot_dir = '../plots/BAECC_scatter_below_cbh/'
import copy
Zg_all_below = copy.deepcopy(Zg_all)
np.putmask(Zg_all_below['var'], cbh_mask_all['var'], np.nan)
MDV_all_below = copy.deepcopy(MDV_all)
np.putmask(MDV_all_below['var'], cbh_mask_all['var'], np.nan)
if compute_widths:
    widths_all_below = copy.deepcopy(widths_all)
    np.putmask(widths_all_below['var'], cbh_mask_all['var'], np.nan)
sw_all_below = copy.deepcopy(sw_all)
np.putmask(sw_all_below['var'], cbh_mask_all['var'], np.nan)

fig, ax = functions.pyLARDA.Transformations.plot_scatter(Zg_all_below, MDV_all_below, identity_line=False, colorbar=True, title=True)
ax.set_xlim([-30, 30])
fig.savefig(plot_dir + f'Ze_MDV.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(sw_all_below, MDV_all_below, identity_line=False, colorbar=True, title=True)
ax.set_xlim([0, 2])
fig.savefig(plot_dir + f'sw_MDV.png')

if compute_widths:
    fig, ax = functions.pyLARDA.Transformations.plot_scatter(widths_all_below, MDV_all_below, identity_line=False, colorbar=True, title=True)
    fig.savefig(plot_dir + f'widths_MDV.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(rmf_all, MDV_all_below, identity_line=False, colorbar=True, title=True)
fig.savefig(plot_dir + f'rmf_MDV.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(rmf_all, sw_all_below, identity_line=False, colorbar=True, title=True)
ax.set_ylim([0, 2])
fig.savefig(plot_dir + f'rmf_sw.png')

if compute_widths:
    fig, ax = functions.pyLARDA.Transformations.plot_scatter(rmf_all, widths_all_below, identity_line=False, colorbar=True, title=True)
    fig.savefig(plot_dir + 'rmf_widths.png')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(widths_all_below, Zg_all_below, color_by=rmf_all, identity_line=False,
                                                             colorbar=True, scale='lin', title=True)
    ax.set_ylim([-40, 20])
    fig.savefig(plot_dir + 'width_Zg_by_rmf.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(sw_all_below, Zg_all_below, color_by=rmf_all, identity_line=False,
                                                         colorbar=True, scale='lin', title=True)
ax.set_ylim([-40, 20])
ax.set_xlim([0, 2])
fig.savefig(plot_dir + 'sw_Zg_by_rmf.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(MDV_all_below, Zg_all_below, color_by=rmf_all, identity_line=False,
                                                         colorbar=True, scale='lin', title=True)
ax.set_ylim([-40, 20])
fig.savefig(plot_dir + 'MDV_Zg_by_rmf.png')



# above cloud base height
plot_dir = '../plots/BAECC_scatter_above_cbh/'
cbh_mask_all['var'] = ~cbh_mask_all['var']
Zg_all_above = copy.deepcopy(Zg_all)
np.putmask(Zg_all_above['var'], cbh_mask_all['var'], np.nan)
MDV_all_above = copy.deepcopy(MDV_all)
np.putmask(MDV_all_above['var'], cbh_mask_all['var'], np.nan)
if compute_widths:
    widths_all_above = copy.deepcopy(widths_all)
    np.putmask(widths_all_above['var'], cbh_mask_all['var'], np.nan)
    fig, ax = functions.pyLARDA.Transformations.plot_scatter(widths_all_above, MDV_all_above, identity_line=False, colorbar=True, title=True)
    fig.savefig(plot_dir + f'widths_MDV.png')
    fig, ax = functions.pyLARDA.Transformations.plot_scatter(widths_all_above, Zg_all_above, color_by=rmf_all, identity_line=False,
                                                             colorbar=True, scale='lin', title=True)
    ax.set_ylim([-40, 20])
    fig.savefig(plot_dir + 'width_Zg_by_rmf.png')


sw_all_above = copy.deepcopy(sw_all)
np.putmask(sw_all_above['var'], cbh_mask_all['var'], np.nan)

fig, ax = functions.pyLARDA.Transformations.plot_scatter(Zg_all_above, MDV_all_above, identity_line=False, colorbar=True, title=True)
ax.set_xlim([-30, 30])
fig.savefig(plot_dir + f'Ze_MDV.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(sw_all_above, MDV_all_above, identity_line=False, colorbar=True, title=True)
ax.set_xlim([0, 2])
fig.savefig(plot_dir + f'sw_MDV.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(rmf_all, MDV_all_above, identity_line=False, colorbar=True, title=True)
fig.savefig(plot_dir + f'rmf_MDV.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(rmf_all, sw_all_above, identity_line=False, colorbar=True, title=True)
ax.set_ylim([0, 2])
fig.savefig(plot_dir + f'rmf_sw.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(rmf_all, sw_all_above, identity_line=False, colorbar=True, title=True)
fig.savefig(plot_dir + 'rmf_widths.png')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(MDV_all_above, Zg_all_above, color_by=rmf_all, identity_line=False,
                                                         colorbar=True, scale='lin', title=True)
ax.set_ylim([-40, 20])
fig.savefig(plot_dir + 'MDV_Zg_by_rmf.png')

