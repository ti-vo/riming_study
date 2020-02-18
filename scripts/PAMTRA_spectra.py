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
               '201909_Bonn_Juelich/Exercises/Moisseev/Snow_retr_2014_2018_for_Python.mat',
               mat_dtype=True)

pamtra_datasets = {'tmatrix': '/home/tvogl/PhD/radar_data/PAMTRA_BAECC/forward_insitu_pamtra_tmatrix_2018.nc',
                   'ssrg': '/home/tvogl/PhD/radar_data/PAMTRA_BAECC/forward_insitu_pamtra_ssrg_2018.nc' }


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

for d in pamtra_datasets.keys():
    with netCDF4.Dataset(pamtra_datasets[d], 'r') as ncD:
        # PAMTRA ranges are on x, y, heightbins grid
        ranges = ncD.variables['height'][:].astype(np.float)[0, 0, :]
        # PAMTRA Ze output has dimensions x, y, height, reflectivity, polarisation, peak number
        Ze = ncD.variables['Ze'][:].astype(np.float)
        re = ncD.variables['Radar_RightEdge'][:].astype(np.float)
        le = ncD.variables['Radar_LeftEdge'][:].astype(np.float)
        spec = ncD.variables['Radar_Spectrum'][:].astype(np.float)
        mdv = ncD.variables['Radar_MeanDopplerVel'][:].astype(np.float)
        sw = ncD.variables['Radar_SpectrumWidth'][:].astype(np.float)
        skewness = ncD.variables['Radar_Skewness'][:].astype(np.float)
        frequency = ncD.variables['frequency'][:].astype(np.float)
        vel_bins = ncD.variables['Radar_Velocity'][:].astype(np.float)


    # compute Dual Wavelength Ratios
    DWR_X_Ka = {'var' : Ze[:, 0, 0, 0, 0, 0] - Ze[:, 0, 0, 1, 0, 0], 'name' : 'DWR_X_Ka',
                'mask': np.ma.getmask(Ze[:, 0, 0, 0, 0, 0]), 'system' : 'PAMTRA', 'var_unit' : 'dB'}
    DWR_Ka_W = {'var' : Ze[:, 0, 0, 1, 0, 0] - Ze[:, 0, 0, 2, 0, 0], 'name' : 'DWR_Ka_W',
                'mask': np.ma.getmask(Ze[:, 0, 0, 1, 0, 0]), 'system' : 'PAMTRA', 'var_unit' : 'dB'}


    fig, ax = functions.pyLARDA.Transformations.plot_scatter(DWR_Ka_W, DWR_X_Ka, identity_line=False, colorbar=True,
                                                             cmap='jet')
    ax.set_ylim([-2, 16])
    ax.set_yticks([0, 4, 8, 12])
    ax.set_xlim([-2, 16])
    ax.set_xticks([0, 5, 10, 15])
    ax.set_title(d, fontsize=15)
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/triple_frequency_{d}.png')

    # spectrum edge width vs. rime mass fraction

    i = 1
    spectra = spec[:, :, 0, i, 0, :]
    Zg = {'var': Ze[:, :, 0, i, 0, 0], 'mask':np.ma.getmask(Ze[:, :, 0, 0, 0, 0]), 'name' : 'reflectivity',
          'system' : 'PIP / PAMTRA', 'var_unit' : 'dBZ'}
    MDV = {'var' : mdv[:, :, 0, i, 0, :], 'mask' : np.ma.getmask(mdv[:, :, 0, 0, 0, 0]), 'name' : 'MDV',
           'system' : 'PIP / PAMTRA', 'var_unit' : 'm/s'}
    SW = {'var' : sw[:, :, 0, i, 0, :], 'mask' : np.ma.getmask(sw[:, :, 0, 0, 0, 0]), 'name' : 'spectrum width',
           'system' : 'PIP / PAMTRA', 'var_unit' : 'm2/s2'}
    skew = {'var': skewness[:, :, 0, i, 0, :], 'mask': np.ma.getmask(skewness[:, :, 0, 0, 0, 0]), 'name': 'skewness',
          'system': 'PIP / PAMTRA', 'var_unit': 'm3/s3?'}

    vel_res = abs(np.median(np.diff(vel_bins[i, :])))
    edge_width = rf.width_fast(functions.h.z2lin(spectra)) * vel_res
    ewc = functions.h.put_in_container(edge_width, rmf_container, name='spectrum edge width', var_unit='m/s',
                                       mask=edge_width < 0, var_lims=[0,4], system='PIP / PAMTRA')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(ewc, Zg, color_by=rmf_container, identity_line=False,
                                                             colorbar=True, scale='lin', c_lim=[0, 0.8], nbins=30)
    ax.set_title(d, fontsize=20)
    ax.set_ylim([-40, 60])
    ax.set_xlim([0, 3])
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/width_Zg_by_rf_{d}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(MDV, Zg, color_by=rmf_container, identity_line=False,
                                                             colorbar=True, scale='lin', c_lim=[0, 0.8], nbins=30)
    ax.set_ylim([-40, 60])
    ax.set_xlim([0, 3])
    ax.set_title(d, fontsize=20)
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/MDV_Zg_by_rf_{d}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(SW, Zg, color_by=rmf_container, identity_line=False,
                                                             colorbar=True, scale='lin', c_lim=[0, 0.8], nbins=30)
    ax.set_ylim([-40, 60])
    ax.set_xlim([0, 1])
    ax.set_title(d, fontsize=20)
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/SW_Zg_by_rf_{d}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(skew, ewc, color_by=rmf_container, identity_line=False,
                                                             colorbar=True, scale='lin', c_lim=[0, 0.8], nbins=30)
    #ax.set_ylim([-40, 60])
    ax.set_title(d, fontsize=20)
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/widths_skew_by_rf_{d}.png')

    fig, ax = plt.subplots(1)
    ax.scatter(ewc['var'], rmf_container['var'], c=Zg['var'])
    ax.set_xlim([0, 5])
    ax.set_xlabel('edge width')
    ax.set_ylabel('rime mass fraction')
    ax.set_ylim([0.3, 1])
    ax.set_title(d, fontsize=20)

    fig, ax = plt.subplots(1)
    ax.scatter(MDV['var'], rmf_container['var'], c=Zg['var'])
    ax.set_xlim([0, 3])
    ax.set_xlabel('mean Doppler velocity')
    ax.set_ylabel('rime mass fraction')
    ax.set_ylim([0.3, 1])
    ax.set_title(d, fontsize=20)

    fig, ax = plt.subplots(1)
    ax.scatter(skew['var'], rmf_container['var'], c=Zg['var'])
    #ax.set_xlim([0, 3])
    ax.set_xlabel('skewness ')
    ax.set_ylabel('rime mass fraction')
    ax.set_ylim([0.3, 1])
    ax.set_title(d, fontsize=20)

    fig, ax = plt.subplots(1)
    sc = ax.scatter(skew['var'], ewc['var'], c=rmf_container['var'], vmin=0)
    plt.colorbar(sc)
    #ax.set_xlim([0, 3])
    ax.set_ylim([0, 3])
    ax.set_xlabel('skewness ')
    ax.set_ylabel('edge width')
    ax.set_title(d, fontsize=20)
