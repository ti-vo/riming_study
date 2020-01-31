import os
import datetime
import numpy as np
import sys
sys.path.append('../')
import functions
import functions.radar_functions as rf
import toml
import functools

# This script reads in a toml file that is stored in /../casestudies_toml/
# in this file, time-height boundaries of case studies are specified
# This script loops over the cases and plots the spectrum edge width, and the MDV-retrieved rime mass fraction.
# If a path to in-situ data is defined, it also plots timeseries of ground-based observations.
# It also reads in upper air sounding data if available, to compute spectrum broadening due to turbulence.


def plot_casestudy_baecc(case_study):
    window = 1 # minutes for computing turbulence broadening
    if not os.path.exists(case_study['plot_dir']):
        os.makedirs(case_study['plot_dir'])
    t_sta, t_end = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case_study['time_interval']]
    h_sta, h_end = case_study['range_interval']

    # read MDV, add +/- one minute for turbulence broadening computation
    MDV = larda.read("KAZR", "mdv", [t_sta-datetime.timedelta(minutes=window),
                                     t_end+datetime.timedelta(minutes=window)], [h_sta, h_end])
    MDV['mask'] = np.logical_or(MDV['mask'], abs(MDV['var']) > 1000)
    np.putmask(MDV['var'], MDV['mask'],  np.nan)
    MDV_corrected = rf.air_density_correction(MDV)
    MDV_corr = functions.h.put_in_container(MDV_corrected, MDV)

    # time resolution
    t_res = np.median(np.diff(MDV['ts']))
    window_length = int(round(window * 60 / t_res))

    # compute timestamp
    time_roll = rf.rolling_window(MDV['ts'], window_length)
    timestamp = np.asarray([time_roll[x][round(window_length / 2)] for x, y in enumerate(np.nanmedian(time_roll, 1))])
    #
    var_out = []
    for j in range(len(MDV['rg'])):
        # calculate the MDV's variance over x minute windows
        variance = np.nanvar(rf.rolling_window(MDV['var'][:, j], window_length), 1)
        var_out.append(variance)

    variance_out = np.swapaxes(np.vstack(var_out), 1, 0)
    idx_a = np.logical_and(timestamp > functions.h.dt_to_ts(t_sta), timestamp < functions.h.dt_to_ts(t_end))
    variance_out = variance_out[idx_a, :]
    timestamp = timestamp[idx_a]
    idx_b = np.logical_and(MDV['ts'] > functions.h.dt_to_ts(t_sta), MDV['ts'] < functions.h.dt_to_ts(t_end))
    MDV['var'] = MDV['var'][idx_b, :]
    MDV['mask'] = MDV['mask'][idx_b, :]
    MDV['ts'] = MDV['ts'][idx_b]

    soundings = rf.read_baecc_soundings('windspeed', [t_sta, t_end], larda=larda)
    # regrid all soundings on radar bin range resolution

    wsp_remapped = [functions.pyLARDA.Transformations.interpolate1d(wspd, new_range=MDV['rg']) for wspd in soundings]
    sounding_container = functools.reduce(functions.pyLARDA.Transformations.join, wsp_remapped)

    # 20 min / 100 m grid
    # define new height and time indices of new grid
    # one time/ height step more in the beginning needed for np.digitize bins

    h_new = np.arange(h_sta - 1, stop=h_end + 100, step=100)
    h_new[-1] += 1
    t_new = np.arange(functions.h.dt_to_ts(t_sta), stop=functions.h.dt_to_ts(t_end) + (20 * 60), step=(20 * 60))
    t_new[-1] += 1

    h_center = h_new[:-1] + 0.5 * np.diff(h_new)
    t_center = t_new[:-1] + 0.5 * np.diff(t_new)

    MDV_coarse = rf.apply_function_timeheight_grid(np.nanmean, MDV_corr, h_new, t_new)
    MDV_coarse = functions.h.put_in_container(MDV_coarse, MDV, ts=t_center, rg=h_center,
                                              mask=np.isnan(MDV_coarse), var_lims=[-2, 2])

    # rime mass fraction
    rime_fraction = rf.rimed_mass_fraction_dmitri(MDV_coarse['var'])
    rime_fraction = functions.h.put_in_container(rime_fraction, MDV_coarse, var_lims=[0, 1], name="rime mass fraction",
                                                 mask=rime_fraction < 0)

    # compute the spectrum edge width using the default values above minimum of spectra:
    widths, ts_widths = rf.read_apply("KAZR", "spec", [t_sta, t_end], [h_sta, h_end], rf.compute_width, larda=larda)
    # correct widths for turbulence broadening

    spectrum_edge_width = functions.h.put_in_container(widths, MDV, ts=ts_widths, name="spectrum edge width",
                                                       mask=widths > 100, var_lims=[0, 5])
    if sounding_container['ts'].shape[0] > 1:
        sounding_container = functions.pyLARDA.Transformations.interpolate2d(sounding_container, new_time=ts_widths)

    sigma_T = rf.turbulence_broadening(sounding_container, spectrum_edge_width)
    widths_corrected = functions.h.put_in_container(widths-sigma_T, spectrum_edge_width, name="corrected edge width")

    edge_width_gridded = rf.apply_function_timeheight_grid(np.nanmean, spectrum_edge_width, h_new, t_new)
    edge_width_gridded = functions.h.put_in_container(edge_width_gridded, MDV, ts=t_center, rg=h_center,
                                                      mask=np.isnan(edge_width_gridded),
                                                      var_lims=[0, 5], name="spectrum edge width")
    edge_width_corr_gridded = rf.apply_function_timeheight_grid(np.nanmean, spectrum_edge_width, h_new, t_new)
    edge_width_corr_gridded = functions.h.put_in_container(edge_width_corr_gridded, edge_width_gridded,
                                                           ts=t_center, rg=h_center, mask=np.isnan(edge_width_gridded))

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


if __name__ == '__main__':

    config_case_studies = toml.load('../casestudies_toml/baecc_case_studies.toml')
    for case in config_case_studies['case']:
        case_study = config_case_studies['case'][case]
        print('case ' + case)
        dt_interval = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case_study['time_interval']]

        larda = functions.pyLARDA.LARDA().connect('arm_baecc', build_lists=True)
        plot_casestudy_baecc(case_study)

    print('\n ...Done...\n')
