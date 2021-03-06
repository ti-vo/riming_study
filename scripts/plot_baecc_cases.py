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

# This script reads in a toml file that is stored in /../casestudies_toml/
# in this file, time-height boundaries of case studies are specified
# This script loops over the cases and plots the spectrum edge width, and the MDV-retrieved rime mass fraction.
# If a path to in-situ data is defined, it also plots timeseries of ground-based observations.
# It also reads in upper air sounding data if available, to compute spectrum broadening due to turbulence.


def plot_casestudy_baecc(case_study):
    window = 1  # minutes for computing turbulence broadening
    # create plot folder if it does not already exist
    if not os.path.exists(case_study['plot_dir']):
        os.makedirs(case_study['plot_dir'])
    # time-range intervals
    t_sta, t_end = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case_study['time_interval']]
    h_sta, h_end = case_study['range_interval']
    in_situ_data = case_study['in_situ_dir']

    # read MDV, add +/- one minute for turbulence broadening computation
    MDV = larda.read("KAZR", "mdv", [t_sta-2*datetime.timedelta(minutes=window),
                                     t_end+2*datetime.timedelta(minutes=window)], [h_sta, h_end])
    MDV['colormap'] = 'jet'
    MDV['mask'] = np.logical_or(MDV['mask'], abs(MDV['var']) > 1000)
    np.putmask(MDV['var'], MDV['mask'],  np.nan)
    Zg = larda.read("KAZR", "Ze", [t_sta, t_end], [h_sta, h_end])

    # remove broken timestamps in ARM radar data
    MDV = rf.remove_broken_timestamp_arm(MDV)
    Zg = rf.remove_broken_timestamp_arm(Zg)

    # apply air density correction
    MDV_corrected = rf.air_density_correction(MDV)
    MDV_corr = functions.h.put_in_container(MDV_corrected, MDV)
    MDV_corr['var_lims'] = [-3, 3]

    # time resolution
    t_res = np.median(np.diff(MDV['ts']))
    window_length = int(round(window * 60 / t_res))

    # compute timestamp for MDV variance
    time_roll = rf.rolling_window(MDV['ts'], window_length)
    timestamp = np.asarray([time_roll[x][round(window_length / 2)] for x, y in enumerate(np.nanmedian(time_roll, 1))])

    # compute variance of MDV, loop over all ranges
    var_out = []
    for j in range(len(MDV['rg'])):
        # calculate the MDV's variance over x minute windows
        variance = np.nanvar(rf.rolling_window(MDV_corr['var'][:, j], window_length), 1)
        var_out.append(variance)

    # rearrange the output array
    variance_out = np.swapaxes(np.vstack(var_out), 1, 0)
    variance_container = functions.h.put_in_container(variance_out, MDV, ts=timestamp, name='variance of MDV',
                                                      var_lims=[0, 1], colormap='jet', mask=variance_out < 0)
    # extract the time period of interest
    MDV = rf.crop_timeheight(MDV, [t_sta, t_end], [h_sta, h_end])
    MDV_corr = rf.crop_timeheight(MDV_corr, [t_sta, t_end], [h_sta, h_end])
    variance_container = rf.crop_timeheight(variance_container, [t_sta, t_end], [h_sta, h_end])
    Zg = rf.crop_timeheight(Zg, [t_sta, t_end], [h_sta, h_end])
    Zg['colormap'] = 'jet'
    Zg['var_lims'] = [-40, 20]

    # read in soundings and regrid them to radar bin range resolution, then merge them into one container
    soundings = rf.read_baecc_soundings('windspeed', [t_sta, t_end], larda=larda)
    wsp_remapped = [functions.pyLARDA.Transformations.interpolate1d(wspd, new_range=MDV['rg']) for wspd in soundings]
    sounding_container = functools.reduce(functions.pyLARDA.Transformations.join, wsp_remapped)

    # 20 min / 100 m grid
    # define new height and time indices of new grid
    # one time/ height step more in the beginning needed for np.digitize bins
    h_new = np.arange(h_sta - 1, stop=h_end + 50, step=100)
    h_new[-1] += 1
    t_new = np.arange(functions.h.dt_to_ts(t_sta), stop=functions.h.dt_to_ts(t_end) + (20 * 60), step=(20 * 60))
    t_new[-1] += 1
    h_center = h_new[:-1] + 0.5 * np.diff(h_new)
    t_center = t_new[:-1] + 0.5 * np.diff(t_new)

    # rempap MDV on the newly defined grid, taking np.nanmean of each grid cell
    MDV_coarse = rf.apply_function_timeheight_grid(np.nanmean, MDV_corr, h_new, t_new)
    MDV_coarse = functions.h.put_in_container(MDV_coarse, MDV, ts=t_center, rg=h_center,
                                              mask=np.isnan(MDV_coarse), var_lims=[-2, 2])
    Zg_coarse = rf.apply_function_timeheight_grid(np.nanmean, Zg, h_new, t_new)
    Zg_coarse = functions.h.put_in_container(Zg_coarse, Zg, ts=t_center, rg=h_center, mask=np.isnan(Zg_coarse))

    # compute the rime mass fraction for the new MDV
    rime_fraction = rf.rimed_mass_fraction_dmitri(MDV_coarse['var'])
    rime_fraction = functions.h.put_in_container(rime_fraction, MDV_coarse, var_lims=[0, 1], name="rime mass fraction",
                                                 mask=rime_fraction < 0)

    # compute the spectrum edge width, using the default values above minimum of spectra:
    widths, ts_widths = rf.read_apply("KAZR", "spec", [t_sta, t_end], [h_sta, h_end], rf.denoise_and_compute_width, larda=larda)
    spectrum_edge_width = functions.h.put_in_container(widths, MDV, ts=ts_widths, name="spectrum edge width",
                                                       mask=widths > 12, var_lims=[0, 5])
    spectrum_edge_width = functions.pyLARDA.Transformations.interpolate2d(spectrum_edge_width, new_time=MDV['ts'],
                                                                          new_range=MDV['rg'], method='nearest')

    skewness, ts_skewness = rf.read_apply("KAZR", "spec", [t_sta, t_end], [h_sta, h_end], rf.denoise_and_get_skewness, larda=larda)
    skewness = functions.h.put_in_container(skewness, MDV, ts=ts_skewness, name='skewness', var_units='')
    skewness = functions.pyLARDA.Transformations.interpolate2d(skewness, new_time=Zg['ts'])

    # remap the soundings to the time resolution of the spectrum edge width
    if sounding_container['ts'].shape[0] > 1:
        sounding_container = functions.pyLARDA.Transformations.interpolate2d(sounding_container,
                                                                             new_time=spectrum_edge_width['ts'])
    else:
        sounding_container['var'] = np.repeat(sounding_container['var'], len(spectrum_edge_width['ts']), axis=0)
        sounding_container['ts'] = spectrum_edge_width['ts']
        sounding_container['mask'] = sounding_container['var'] < 0
    sounding_container['var_lims'] = [0, 30]
    #sounding_container = rf.crop_timeheight(sounding_container, [t_sta, t_end], [h_sta, h_end])

    # correct widths for turbulence broadening
    # compute turbulence broadening according to Shupe et al. (2008) using horizontal wind from soundings and
    # variance over 1-minute time window

    sigma_T = rf.turbulence_broadening(sounding_container, variance_container)
    sigma_container = functions.h.put_in_container(sigma_T, spectrum_edge_width, name="sigma T", var_lims=[0, 1])

    # just subtract sigma T from the width
    widths_corrected = functions.h.put_in_container(spectrum_edge_width['var']-sigma_T, spectrum_edge_width, name="corrected edge width")

    # regridding of corrected and uncorrected spectrum edge widths
    edge_width_gridded = rf.apply_function_timeheight_grid(np.nanmean, spectrum_edge_width, h_new, t_new)
    edge_width_gridded = functions.h.put_in_container(edge_width_gridded, MDV, ts=t_center, rg=h_center,
                                                      mask=edge_width_gridded < 0,
                                                      var_lims=[0, 5], name="spectrum edge width")
    edge_width_corr_gridded = rf.apply_function_timeheight_grid(np.nanmean, spectrum_edge_width, h_new, t_new)
    edge_width_corr_gridded = functions.h.put_in_container(edge_width_corr_gridded, edge_width_gridded,
                                                           ts=t_center, rg=h_center, mask=edge_width_corr_gridded < 0)

    # reading in in-situ data and computing rime mass fraction
    PIP_data = loadmat(in_situ_data, mat_dtype=True)
    rmf_pip, piptime = rf.rimed_mass_fraction_PIP(PIP_data)
    rmf_pip = {'var': rmf_pip, 'ts': piptime, 'name': 'rime mass fraction', 'dimlabel': ['time'],
               'mask': np.isnan(rmf_pip), 'var_lims': [0, 1], 'system': 'PIP', 'var_unit': 'unitless'}
    rmf_all = functions.pyLARDA.Transformations.interpolate1d(rmf_pip, new_time=Zg['ts'])
    rmf_all['var'] = np.tile(rmf_all['var'][:, np.newaxis], MDV_corr['var'].shape[1])
    rmf_all['rg'] = MDV_corr['rg']
    rmf_all['mask'] = rmf_all['var'] < 0


    # plotting
    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(rime_fraction, title=True, time_diff_jumps=30 * 60)
    fig.savefig(case_study['plot_dir'] + f'rmf_{t_sta.strftime("%Y%m%d%H%M")}_{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(edge_width_gridded, title=True, time_diff_jumps=30*60)
    fig.savefig(case_study['plot_dir'] + f'edge_width_coarse_{t_sta.strftime("%Y%m%d%H%M")}_'
                f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(edge_width_corr_gridded, title=True,
                                                                time_diff_jumps=30*60)
    fig.savefig(case_study['plot_dir'] + f'edge_width_corr_coarse_{t_sta.strftime("%Y%m%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(spectrum_edge_width, title=True)
    fig.savefig(case_study['plot_dir'] + f'edge_width_fine_{t_sta.strftime("%Y%m%d%H%M")}_'
                f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(widths_corrected, title=True)
    fig.savefig(case_study['plot_dir'] + f'edge_width_corr_fine_{t_sta.strftime("%Y%m%d%H%M")}_'
                f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(sigma_container, title=True)
    fig.savefig(case_study['plot_dir'] + f'sigma_T{t_sta.strftime("%Y%m%d%H%M")}_'
                f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(sounding_container, title=True, time_interval=[t_sta,
                                                                                                               t_end])
    fig.savefig(case_study['plot_dir'] + f'horizontal_wind{t_sta.strftime("%Y%m%d%H%M")}_'
                f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(variance_container, title=True)
    fig.savefig(case_study['plot_dir'] + f'MDV_variance_{t_sta.strftime("%Y%m%d%H%M")}_'
                f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeseries(rmf_pip, time_interval=[t_sta, t_end],
                                                                time_diff_jumps=3600)
    # TODO improve ticker in larda so that the following becomes unnecessary
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))

    fig.savefig(case_study['plot_dir'] + f'insitu_timeseries_rmf.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(Zg_coarse, title=True, var_converter="lin2z",
                                                                time_diff_jumps=30 * 60)
    fig.savefig(case_study['plot_dir'] + f'Zg_coarse_{t_sta.strftime("%Y%M%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(Zg, title=True, var_converter="lin2z",
                                                                time_diff_jumps=30 * 60)
    fig.savefig(case_study['plot_dir'] + f'Zg_fine_{t_sta.strftime("%Y%M%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(MDV_corr, title=True)
    fig.savefig(case_study['plot_dir'] + f'MDV_corr_fine_{t_sta.strftime("%Y%M%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_scatter(skewness, spectrum_edge_width, color_by=rmf_all, identity_line=False,
                                                             colorbar=True, scale='lin', title=True, c_lim=[0, 0.8])
    fig.savefig(case_study['plot_dir'] + 'width_skewness_by_rmf.png')

if __name__ == '__main__':

    config_case_studies = toml.load('../casestudies_toml/baecc_case_studies.toml')
    larda = functions.pyLARDA.LARDA().connect('arm_baecc', build_lists=True)
    for case in config_case_studies['case']:
        case_study = config_case_studies['case'][case]
        print('case ' + case)
        dt_interval = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case_study['time_interval']]

        plot_casestudy_baecc(case_study)

    print('\n ...Done...\n')
