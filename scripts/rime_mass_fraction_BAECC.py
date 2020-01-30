import sys
sys.path.append('../')
import functions
import datetime
import functions.radar_functions as rf
import numpy as np

larda = functions.pyLARDA.LARDA().connect('arm_baecc', build_lists=True)
t_sta = datetime.datetime(2014, 2, 21, 17, 5, 0)
t_end = datetime.datetime(2014, 2, 22, 3, 0, 0)

h_sta = 300
#h_end = 600
h_end = 6000

MDV = larda.read("KAZR", "mdv", [t_sta,  t_end], [h_sta, h_end])

# broken timestamp in ARM radar data
time_list = MDV['ts']
dt_list = [time for time in time_list]
jump_index = ~((abs(np.diff(time_list)) > 60) | (np.diff(time_list) == 0))

MDV['ts'] = MDV['ts'][np.where(jump_index)[0]]
MDV['var'] = MDV['var'][np.where(jump_index)[0]]
MDV['mask'] = MDV['mask'][np.where(jump_index)[0]]
# mask MDV values above 1000, and correct for air density
MDV['mask'] = np.logical_or(MDV['mask'], abs(MDV['var']) > 1000)
np.putmask(MDV['var'], MDV['mask'],  np.nan)
MDV_corrected = rf.air_density_correction(MDV)
MDV_corr = functions.h.put_in_container(MDV_corrected, MDV)

# 20 min / 100 m grid
# define new height and time indices of new grid
# one time/ height step more in the beginning needed for np.digitize bins
h_new = np.arange(h_sta-1, stop=h_end+100, step=100) # rime mass fraction

h_new[-1] +=1
t_new = np.arange(functions.h.dt_to_ts(t_sta), stop=functions.h.dt_to_ts(t_end)+(20*60), step=(20*60))
t_new[-1] +=1
h_center = h_new[:-1] + 0.5*np.diff(h_new)
t_center = t_new[:-1] + 0.5*np.diff(t_new)


MDV_coarse = rf.apply_function_timeheight_grid(np.nanmean, MDV_corr, h_new, t_new)
MDV_coarse = functions.h.put_in_container(MDV_coarse, MDV, ts=t_center, rg=h_center,
                                mask=np.isnan(MDV_coarse), var_lims=[-2, 2])

rime_fraction = rf.rimed_mass_fraction_dmitri(MDV_coarse['var'])
rime_fraction = functions.h.put_in_container(rime_fraction, MDV_coarse, var_lims=[0, 1], name="rime mass fraction",
                                             colormap='jet')

fig, ax = functions.pyLARDA.Transformations.plot_timeheight(rime_fraction, title=True, time_diff_jumps=30*60)
fig.savefig(f'../plots/BAECC_rmf_{t_sta.strftime("%Y%m%d%H%M")}_{t_end.strftime("%Y%m%d%H%M")}.png')

