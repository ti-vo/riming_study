import os
import datetime
import numpy as np
import sys
sys.path.append('../')
import functions
import functions.radar_functions as rf
import toml

# This script reads in a toml file that is stored in /../casestudies_toml/
# in this file, time-height boundaries of case studies are specified
# This script loops over the cases and plots the spectrum edge width, and the MDV-retrieved rime mass fraction.
# It uses the cloudnet mask to identify ice and supercooled liquid regions of the cloud.


def plot_casestudy_tripex(case_study):
    window = 1  # minute for MDV variance computation
    if not os.path.exists(case_study['plot_dir']):
        os.makedirs(case_study['plot_dir'])
    t_sta, t_end = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case_study['time_interval']]
    h_sta, h_end = case_study['range_interval']

   # read MDV, add +/- one minute for variance computation

    MDV = larda.read("MIRA", "VELg", [t_sta-2*datetime.timedelta(minutes=window),
                                      t_end+2*datetime.timedelta(minutes=window)], [h_sta, h_end])
    MDV['mask'] = np.logical_or(MDV['mask'], abs(MDV['var']) > 1000)
    np.putmask(MDV['var'], MDV['mask'],  np.nan)
    MDV_corrected = rf.air_density_correction(MDV)
    MDV_corr = functions.h.put_in_container(MDV_corrected, MDV)
    Zg = larda.read("MIRA", "Zg", [t_sta, t_end], [h_sta, h_end])
    Zg['colormap'] = 'jet'

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
    MDV_corr['var_lims'] = [-3, 3]
    MDV_corr['colormap'] = 'jet'
    variance_container = rf.crop_timeheight(variance_container, [t_sta, t_end], [h_sta, h_end])
    Zg = rf.crop_timeheight(Zg, [t_sta, t_end], [h_sta, h_end])

    category = larda.read("CLOUDNET", "category_bits", [t_sta, t_end],  [h_sta, h_end])
    quality = larda.read("CLOUDNET", "quality_bits", [t_sta, t_end],  [h_sta, h_end])
    classes = rf.get_target_classification({"quality_bits": quality, "category_bits": category})
    cloudnet_class = functions.h.put_in_container(classes, category, name="CLASS")
    cloudnet_class = rf.crop_timeheight(cloudnet_class, [t_sta, t_end], [h_sta, h_end])

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

    cloudnet_regridded = rf.regrid_integer_timeheight(cloudnet_class, h_new, t_new, fraction=0.75)
    cloudnet_interpolated = functions.pyLARDA.Transformations.interpolate2d(cloudnet_class, method='nearest',
                                                                            new_time=t_center, new_range=h_center)
    cloudnet_regridded = functions.h.put_in_container(cloudnet_regridded, cloudnet_interpolated,
                                                      mask=np.isnan(cloudnet_regridded), ts=t_center, rg=h_center)

    Zg_coarse = rf.apply_function_timeheight_grid(np.nanmean, Zg, h_new, t_new)
    Zg_coarse = functions.h.put_in_container(Zg_coarse, Zg, ts=t_center, rg=h_center,
                                             mask=~((cloudnet_regridded['var'] == 4) | (
                                                     cloudnet_regridded['var'] == 5)))

    # rime mass fraction
    rime_fraction = rf.rimed_mass_fraction_dmitri(MDV_coarse['var'])
    rime_fraction = functions.h.put_in_container(rime_fraction, MDV_coarse, var_lims=[0, 1], name="rime mass fraction",
                                                 mask=~((cloudnet_regridded['var'] == 4) | (
                                                             cloudnet_regridded['var'] == 5)))

    # compute the spectrum edge width using the default values above minimum of spectra:
    widths, ts_widths = rf.read_apply("MIRA", "SPCco", [t_sta, t_end], [h_sta, h_end], rf.compute_width, larda=larda)
    spectrum_edge_width = functions.h.put_in_container(widths, MDV, ts=ts_widths, name="spectrum edge width",
                                                       mask=widths > 6, var_lims=[0, 5])
    edge_width_gridded = rf.apply_function_timeheight_grid(np.nanmean, spectrum_edge_width, h_new, t_new)
    edge_width_gridded = functions.h.put_in_container(edge_width_gridded, MDV, ts=t_center, rg=h_center,
                                                      mask=~((cloudnet_regridded['var'] == 4) |
                                                             (cloudnet_regridded['var'] == 5)),
                                                      var_lims=[0, 5], name="spectrum edge width")

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(rime_fraction, title=True, time_diff_jumps=60 * 60)
    fig.savefig(case_study['plot_dir'] + f'rmf_{t_sta.strftime("%Y%m%d%H%M")}_{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(edge_width_gridded, title=True, time_diff_jumps=60*60)
    fig.savefig(case_study['plot_dir'] + f'edge_width_coarse_{t_sta.strftime("%Y%m%d%H%M")}_'
                f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(spectrum_edge_width, title=True)
    fig.savefig(case_study['plot_dir'] + f'edge_width_fine_{t_sta.strftime("%Y%m%d%H%M")}_'
                f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(variance_container, title=True)
    fig.savefig(case_study['plot_dir'] + f'MDV_variance_{t_sta.strftime("%Y%m%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(cloudnet_regridded, title=True,
                                                                time_diff_jumps=60*60)
    fig.savefig(case_study['plot_dir'] + f'cloudnet_coarse_{t_sta.strftime("%Y%M%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(Zg_coarse, title=True, var_converter="lin2z",
                                                                time_diff_jumps=60*60)
    fig.savefig(case_study['plot_dir'] + f'Zg_coarse_{t_sta.strftime("%Y%M%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(Zg, title=True, var_converter="lin2z",
                                                                time_diff_jumps=30 * 60)
    fig.savefig(case_study['plot_dir'] + f'Zg_fine_{t_sta.strftime("%Y%M%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')

    fig, ax = functions.pyLARDA.Transformations.plot_timeheight(MDV_corr, title=True)
    fig.savefig(case_study['plot_dir'] + f'MDV_corr_fine_{t_sta.strftime("%Y%M%d%H%M")}_'
    f'{t_end.strftime("%Y%m%d%H%M")}.png')


if __name__ == '__main__':

    config_case_studies = toml.load('../casestudies_toml/tripex_case_studies.toml')
    for case in config_case_studies['case']:
        case_study = config_case_studies['case'][case]
        print('case ' + case)
        dt_interval = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case_study['time_interval']]

        larda = functions.pyLARDA.LARDA().connect('optimice_tripex', build_lists=True)
        plot_casestudy_tripex(case_study)

    print('\n ...Done...\n')
