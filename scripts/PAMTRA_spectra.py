import netCDF4
import numpy as np
import sys
sys.path.append('../')
import functions
import functions.radar_functions as rf
from scipy.io import  loadmat
import toml
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Qt4Agg')
matplotlib.use('Agg')

config_case_studies = toml.load('../casestudies_toml/pamtra_files.toml')
pip_config = toml.load('../casestudies_toml/path_to_pip.toml')

# read in PAMTRA forward simulated in-situ spectra and the original in-situ PIP data

PIP_data = loadmat(pip_config['pip_path']['path'])

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
#rmf = rmf[790:810]
#timestamp = timestamp[:, 790:810]

rmf_container = {'var' : rmf[:, np.newaxis], 'ts' : timestamp[0,:], 'name' : 'rime mass fraction', 'dimlabel': ['time'],
                 'mask' : np.isnan(rmf[:, np.newaxis]), 'var_lims' : [0, 1], 'system': 'PIP', 'var_unit' : 'unitless'}

for d in config_case_studies['file']:
    pamtra_dataset = config_case_studies['file'][d]
    with netCDF4.Dataset(pamtra_dataset['path'], 'r') as ncD:
        # PAMTRA ranges are on x, y, heightbins grid
        ranges = ncD.variables['height'][:].astype(np.float)[0, 0, :]
        # PAMTRA Ze output has dimensions x, y, height, reflectivity, polarisation, peak number
        Ze = ncD.variables['Ze'][:]
        re = ncD.variables['Radar_RightEdge'][:].astype(np.float)
        le = ncD.variables['Radar_LeftEdge'][:].astype(np.float)
        mdv = ncD.variables['Radar_MeanDopplerVel'][:].astype(np.float)
        sw = ncD.variables['Radar_SpectrumWidth'][:].astype(np.float)
        skewness = ncD.variables['Radar_Skewness'][:].astype(np.float)
        frequency = ncD.variables['frequency'][:].astype(np.float)
        vel_bins = ncD.variables['Radar_Velocity'][:].astype(np.float)



        spectra = ncD.variables['Radar_Spectrum'][:, :, 0, 1, 0, :]


    skewness_2 = rf.denoise_and_get_skewness({'var':functions.h.z2lin(spectra),
                                              'ts':rmf_container['ts'][:spectra.shape[0]], 'rg':ranges, 'vel':vel_bins})
    Zg = {'var': Ze[:, :, 0, 1, 0, 0], 'mask':np.ma.getmask(Ze[:, :, 0, 0, 0, 0]), 'name' : 'reflectivity',
          'system' : 'PIP / PAMTRA', 'var_unit' : 'dBZ'}
    MDV = {'var' : -mdv[:, :, 0, 1, 0, :], 'mask' : np.ma.getmask(mdv[:, :, 0, 0, 0, 0]), 'name' : 'MDV',
           'system' : 'PIP / PAMTRA', 'var_unit' : 'm/s'}
    SW = {'var' : sw[:, :, 0, 1, 0, :], 'mask' : np.ma.getmask(sw[:, :, 0, 0, 0, 0]), 'name' : 'spectrum width',
           'system' : 'PIP / PAMTRA', 'var_unit' : 'm2/s2'}
    skew = {'var': -skewness[:, :, 0, 1, 0, :], 'mask': np.ma.getmask(skewness[:, :, 0, 0, 0, 0]), 'name': 'skewness',
          'system': 'PIP / PAMTRA', 'var_unit': 'unitless'}

    skew2 = {'var': -skewness_2, 'mask': np.ma.getmask(skewness_2), 'name': 'skewness',
            'system': 'PIP / PAMTRA', 'var_unit': 'unitless'}
    vel_res = abs(np.median(np.diff(vel_bins[1, :])))
    edge_width = rf.width_fast(functions.h.z2lin(spectra)) * vel_res
    ewc = functions.h.put_in_container(edge_width, rmf_container, name='spectrum edge width', var_unit='m/s',
                                       mask=edge_width < 0, var_lims=[0, 4], system='PIP / PAMTRA')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(ewc, Zg, color_by=rmf_container, identity_line=False,
                                                             colorbar=True, scale='lin', c_lim=[0, 0.8], nbins=30)
    ax.set_title(d, fontsize=20)
    ax.set_ylim([-40, 60])
    ax.set_xlim([0, 5])
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/width_Zg_by_rf_{d}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(MDV, Zg, color_by=rmf_container, identity_line=False,
                                                             colorbar=True, scale='lin', c_lim=[0, 0.8], nbins=30)
    ax.set_ylim([-40, 60])
    ax.set_xlim([-3, 0])
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
    ax.set_ylim([0.8, 5])
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/widths_skew_by_rf_{d}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(skew2, ewc, color_by=rmf_container, identity_line=False,
                                                             colorbar=True, scale='lin', c_lim=[0, 0.8], nbins=30)
    #ax.set_ylim([-40, 60])
    ax.set_title(d, fontsize=20)
    ax.set_ylim([0.8, 5])
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/widths_skew_Willi_by_rf_{d}.png')

    # compute Dual Wavelength Ratios
    DWR_X_Ka = {'var': Ze[:, 0, 0, 0, 0, 0] - Ze[:, 0, 0, 1, 0, 0], 'name' : 'DWR_X_Ka',
                'mask': np.ma.getmask(Ze[:, 0, 0, 0, 0, 0]), 'system' : 'PAMTRA', 'var_unit' : 'dB'}
    DWR_Ka_W = {'var': Ze[:, 0, 0, 1, 0, 0] - Ze[:, 0, 0, 2, 0, 0], 'name' : 'DWR_Ka_W',
                'mask': np.ma.getmask(Ze[:, 0, 0, 1, 0, 0]), 'system' : 'PAMTRA', 'var_unit' : 'dB'}


    fig, ax = functions.pyLARDA.Transformations.plot_scatter(DWR_Ka_W, DWR_X_Ka, identity_line=False, colorbar=True,
                                                             cmap='jet')
    ax.set_ylim([-2, 16])
    ax.set_yticks([0, 4, 8, 12])
    ax.set_xlim([-2, 16])
    ax.set_xticks([0, 5, 10, 15])
    ax.set_title(d, fontsize=15)
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/triple_frequency_{d}.png')

    # Ze histogram plot
    np.putmask(Ze[:, :, 0, 1, 0, 0], np.ma.getmask(Ze[:, :, 0, 1, 0, 0]), np.nan)
    np.putmask(MDV['var'], np.ma.getmask(MDV['var']), np.nan)
    np.putmask(SW['var'], np.ma.getmask(SW['var']), np.nan)

    fig, ax = plt.subplots(1)
    a, b = np.histogram(Ze[:, :, 0, 1, 0, 0].flatten()[~np.isnan(Ze[:, :, 0, 1, 0, 0].flatten())], bins=100)
    ax.bar(b[:-1] + np.diff(b)/2, a, np.diff(b))
    ax.set_title(f'histogram of reflectivities, {d}')
    ax.set_xlabel('Reflectivity [dBZ]')
    ax.set_ylabel('FoO')
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/Zg_hist_{d}.png')

    fig, ax = plt.subplots(1)
    a, b = np.histogram(MDV['var'].flatten()[~np.isnan(MDV['var'].flatten())], bins=100)
    ax.bar(b[:-1] + np.diff(b) / 2, a, np.diff(b))
    ax.set_title(f'histogram of MDV, {d}')
    ax.set_xlabel('MDV [m s-1]')
    ax.set_ylabel('FoO')
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/MDV_hist_{d}.png')

    fig, ax = plt.subplots(1)
    a, b = np.histogram(SW['var'].flatten()[~np.isnan(SW['var'].flatten())], bins=100)
    ax.bar(b[:-1] + np.diff(b)/2, a, np.diff(b))
    ax.set_title(f'histogram of spectrum width, {d}')
    ax.set_xlabel('spectrum width [m/s]')
    ax.set_ylabel('FoO')
    fig.savefig(f'../plots/BAECC_PAMTRA_PIP/SW_hist_{d}.png')

#   fig, ax = plt.subplots(1)
 #   ax.scatter(ewc['var'], rmf_container['var'], c=Zg['var'])
 #   ax.set_xlim([0, 5])
 #   ax.set_xlabel('edge width')
 #   ax.set_ylabel('rime mass fraction')
 #   ax.set_ylim([0.3, 1])
 #   ax.set_title(d, fontsize=20)

#    fig, ax = plt.subplots(1)
#    ax.scatter(MDV['var'], rmf_container['var'], c=Zg['var'])
#    ax.set_xlim([0, 3])
#    ax.set_xlabel('mean Doppler velocity')
#    ax.set_ylabel('rime mass fraction')
#    ax.set_ylim([0.3, 1])
#    ax.set_title(d, fontsize=20)
#
#    fig, ax = plt.subplots(1)
#    ax.scatter(skew['var'], rmf_container['var'], c=Zg['var'])
#    #ax.set_xlim([0, 3])
#    ax.set_xlabel('skewness ')
#    ax.set_ylabel('rime mass fraction')
#    ax.set_ylim([0.3, 1])
#    ax.set_title(d, fontsize=20)
#
    #fig, ax = plt.subplots(1)
    #sc = ax.scatter(skew['var'], ewc['var'], c=rmf_container['var'], vmin=0)
    #plt.colorbar(sc)
    ##ax.set_xlim([0, 3])
    #ax.set_ylim([0, 3])
    #ax.set_xlabel('skewness ')
    #ax.set_ylabel('edge width')
    #ax.set_title(d, fontsize=20)

    #fig, ax = plt.subplots(1)
    #ax.hist(Zg['var'].flatten())
    #fig.savefig(f'../plots/BAECC_PAMTRA_PIP/Zg_hist_{d}.png')
