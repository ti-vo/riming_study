
import sys
sys.path.append('../')
import functions
import datetime
import functions.radar_functions as rf
import functions.other_functions as of
import numpy as np

larda = functions.pyLARDA.LARDA().connect('optimice_tripex', build_lists=True)

# careful when setting t_sta and t_end: it should be possible to read in +/- one window length or timestamps won't match
t_sta = datetime.datetime(2019, 1, 5, 0, 5, 0)
t_end = datetime.datetime(2019, 1, 5, 9, 59, 59)

h_sta = 0
h_end = 9970
window = 1  # minutes, used for computing variance of MDV


# read MDV and cloudnet class, for MDV adding one window length before and after the desired interval
MDV = larda.read("MIRA", "VELg", [t_sta - datetime.timedelta(minutes=window),
                                  t_end + datetime.timedelta(minutes=window)], [h_sta, h_end])
MDV['mask'] = np.logical_or(MDV['mask'], abs(MDV['var']) > 1000)
np.putmask(MDV['var'], MDV['mask'],  np.nan)

# time resolution
t_res = np.median(np.diff(MDV['ts']))
window_length = int(round(window*60 / t_res))

time_roll = rf.rolling_window(MDV['ts'], window_length)
timestamp = np.asarray([time_roll[x][round(window_length/2)] for x,y in enumerate(np.nanmedian(time_roll, 1))])

var_out = []
for j in range(len(MDV['rg'])):
    # calculate the MDV's variance over the specified windows
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

var_container = functions.h.put_in_container(variance_out, MDV, name=f'MDV variance over {window} min',
                                             var_lims=[0, 1])

# 20 min / 100 m grid
# define new height and time indices of new grid
# one time/ height step more in the beginning needed for np.digitize bins
h_new = np.arange(h_sta-1, stop=h_end+100, step=100)
h_new[-1] +=1
t_new = np.arange(functions.h.dt_to_ts(t_sta), stop=functions.h.dt_to_ts(t_end)+(20*60), step=(20*60))
t_new[-1] +=1
h_center = h_new[:-1] + 0.5*np.diff(h_new)
t_center = t_new[:-1] + 0.5*np.diff(t_new)

MDV_var = rf.apply_function_timeheight_grid(np.nanvar, MDV, h_new, t_new)
MDV_var = functions.h.put_in_container(MDV_var, MDV, ts=t_center, rg=h_center, mask=np.isnan(MDV_var),
                                       var_lims=[0, 2], name="MDV variance over 20 min")

fig, ax = functions.pyLARDA.Transformations.plot_timeheight(var_container, title=True)
fig.savefig(f'../plots/TRIPEX_MDVvar_fine_{t_sta.strftime("%Y%m%d%H%M")}_{t_end.strftime("%Y%m%d%H%M")}.png')

fig, ax = functions.pyLARDA.Transformations.plot_timeheight(MDV_var, title=True, time_diff_jumps=30*60)
fig.savefig(f'../plots/TRIPEX_MDVvar_20min_{t_sta.strftime("%Y%m%d%H%M")}_{t_end.strftime("%Y%m%d%H%M")}.png')
