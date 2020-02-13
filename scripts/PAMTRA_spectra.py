import netCDF4
import numpy as np
import sys
sys.path.append('../')
import functions
import functions.radar_functions as rf
from scipy.io import  loadmat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt4Agg')

# read in PAMTRA forward simulated in-situ spectra and the original in-situ PIP data

PIP_data = loadmat('/home/tvogl/PhD/conferences_workshops/'
               '201909_Bonn_Juelich/Exercises/Moisseev/Snow_retr_2014_2015_for_Python.mat',
               mat_dtype=True)


# rime fraction according to equation (3) in Kneifel & Moisseev (2020)

# mass of unrimed snow
m_us = 0.0053 * PIP_data['Dmax']**2.05
mask = np.isnan(PIP_data['mass'])
mass = PIP_data['mass']
N = PIP_data['PSD']
timestamp = PIP_data['time']

np.putmask(m_us, mask, 0)
np.putmask(mass, mask, 0)
np.putmask(N, mask, 0)

rmf = 1 - (np.trapz((m_us*N), PIP_data['Dmax']))/np.trapz((mass * N), PIP_data['Dmax'])

rmf_container = {'var' : rmf[:, np.newaxis], 'ts' : timestamp[0,:], 'name' : 'rime mass fraction', 'dimlabel': ['time'],
                 'mask' : np.isnan(rmf[:, np.newaxis]), 'var_lims' : [0, 1], 'system': 'PIP', 'var_unit' : 'unitless'}


with netCDF4.Dataset('/home/tvogl/PhD/radar_data/PAMTRA_BAECC/forward_insitu_pamtra_ssrg.nc', 'r') as ncD:
    # PAMTRA ranges are on x, y, heightbins grid
    ranges = ncD.variables['height'][:].astype(np.float)[0, 0, :]
    # PAMTRA Ze output has dimensions x, y, height, reflectivity, polarisation, peak number
    Ze = ncD.variables['Ze'][:].astype(np.float)
    re = ncD.variables['Radar_RightEdge'][:].astype(np.float)
    le = ncD.variables['Radar_LeftEdge'][:].astype(np.float)
    spec = ncD.variables['Radar_Spectrum'][:].astype(np.float)
    frequency = ncD.variables['frequency'][:].astype(np.float)
    vel_bins = ncD.variables['Radar_Velocity'][:].astype(np.float)


# compute Dual Wavelength Ratios
DWR_X_Ka = {'var' : Ze[:, 0, 0, 0, 0, 0] - Ze[:, 0, 0, 1, 0, 0], 'name' : 'DWR_X_Ka',
            'mask': np.ma.getmask(Ze[:, 0, 0, 0, 0, 0]), 'system' : 'PAMTRA', 'var_unit' : 'dB'}
DWR_Ka_W = {'var' : Ze[:, 0, 0, 1, 0, 0] - Ze[:, 0, 0, 2, 0, 0], 'name' : 'DWR_Ka_W',
            'mask': np.ma.getmask(Ze[:, 0, 0, 1, 0, 0]), 'system' : 'PAMTRA', 'var_unit' : 'dB'}


fig, ax = functions.pyLARDA.Transformations.plot_scatter(DWR_Ka_W, DWR_X_Ka, identity_line=False, colorbar=True)
fig.savefig('../plots/triple_frequency.png')

# spectrum edge width vs. rime mass fraction

i = 0
spectra = spec[:, :, 0, i, 0, :]
Zg = {'var': Ze[:, :, 0, i, 0, 0], 'mask':np.ma.getmask(Ze[:, :, 0, 0, 0, 0]), 'name' : 'reflectivity',
      'system' : 'PIP / PAMTRA', 'var_unit' : 'dBZ'}

vel_res = abs(np.median(np.diff(vel_bins[i, :])))
edge_width = rf.width_fast(functions.h.z2lin(spectra)) * vel_res
ewc = functions.h.put_in_container(edge_width, rmf_container, name='spectrum edge width', var_unit='m/s',
                                   mask=edge_width < 0, var_lims=[0,4], system='PIP / PAMTRA')

fig, ax = functions.pyLARDA.Transformations.plot_scatter(ewc, Zg, color_by=rmf_container, identity_line=False, colorbar=True)
