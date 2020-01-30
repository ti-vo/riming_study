
import sys
sys.path.append('../')
import functions
import datetime
import functions.radar_functions as rf
import numpy as np

larda = functions.pyLARDA.LARDA().connect('optimice_tripex', build_lists=True)

t_sta = datetime.datetime(2019, 1, 5, 0, 5, 0)
t_end = datetime.datetime(2019, 1, 5, 9, 59, 59)

h_sta = 0
h_end = 9970

MDV = larda.read("MIRA", "VELg", [t_sta, t_end ], [h_sta, h_end])
MDV['mask'] = np.logical_or(MDV['mask'], abs(MDV['var']) > 1000)
np.putmask(MDV['var'], MDV['mask'],  np.nan)

category = larda.read("CLOUDNET", "category_bits", [t_sta, t_end],  [h_sta, h_end])
quality = larda.read("CLOUDNET", "quality_bits", [t_sta, t_end],  [h_sta, h_end])
classes = rf.get_target_classification({"quality_bits": quality, "category_bits": category})
cloudnet_class = functions.h.put_in_container(classes, category, name="CLASS")

idx_b = np.logical_and(MDV['ts'] > functions.h.dt_to_ts(t_sta), MDV['ts'] < functions.h.dt_to_ts(t_end))
MDV['var'] = MDV['var'][idx_b, :]
MDV['mask'] = MDV['mask'][idx_b, :]
MDV['ts'] = MDV['ts'][idx_b]

# compute the spectrum edge width using the default values above minimum of spectra:
widths, ts_widths = rf.read_apply("MIRA", "SPCco", [t_sta, t_end], [h_sta, h_end], rf.compute_width, larda=larda)
spectrum_edge_width = functions.h.put_in_container(widths, MDV, ts=ts_widths, name="spectrum edge width",
                                                   mask=widths > 100, var_lims=[0, 5])
# 20 min / 100 m grid
# define new height and time indices of new grid
# one time/ height step more in the beginning needed for np.digitize bins
h_new = np.arange(h_sta-1, stop=h_end+100, step=100)
h_new[-1] +=1
t_new = np.arange(functions.h.dt_to_ts(t_sta), stop=functions.h.dt_to_ts(t_end)+(20*60), step=(20*60))
t_new[-1] +=1
h_center = h_new[:-1] + 0.5*np.diff(h_new)
t_center = t_new[:-1] + 0.5*np.diff(t_new)

cloudnet_regridded = rf.regrid_integer_timeheight(cloudnet_class, h_new, t_new)
cloudnet_interpolated = functions.pyLARDA.Transformations.interpolate2d(cloudnet_class, method='nearest',
                                                                        new_time=t_center, new_range=h_center)
cloudnet_regridded = functions.h.put_in_container(cloudnet_regridded, cloudnet_interpolated,
                                                  mask=np.isnan(cloudnet_regridded), ts=t_center, rg=h_center)

edge_width_gridded = rf.apply_function_timeheight_grid(np.nanmean, spectrum_edge_width, h_new, t_new)
edge_width_gridded = functions.h.put_in_container(edge_width_gridded, MDV, ts=t_center, rg=h_center,
                                                  mask=~((cloudnet_regridded['var'] == 4) |
                                                         (cloudnet_regridded['var'] == 5)),
                                                  var_lims=[0, 5], name="spectrum edge width")

fig, ax = functions.pyLARDA.Transformations.plot_timeheight(edge_width_gridded, title=True, time_diff_jumps=30*60)
fig.savefig(f'../plots/TRIPEX_edge_width_coarse_{t_sta.strftime("%Y%m%d%H%M")}_{t_end.strftime("%Y%m%d%H%M")}.png')

fig, ax = functions.pyLARDA.Transformations.plot_timeheight(spectrum_edge_width, title=True)
fig.savefig(f'../plots/TRIPEX_edge_width_fine_{t_sta.strftime("%Y%m%d%H%M")}_{t_end.strftime("%Y%m%d%H%M")}.png')



