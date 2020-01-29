# compute the rime mass fraction for 20 minute/ 100 m grid according to Kneifel & Moisseev (2020) for a TRIPEX-Pol case.
# The Cloudnet classification mask is used to only compute it for the ice/ supercooled liquid part of the cloud.
# plot time-height plot of rime mass fraction using larda

import sys
sys.path.append('../')
import functions
import datetime
import functions.radar_functions as rf
import functions.other_functions as of
import numpy as np

larda = functions.pyLARDA.LARDA().connect('optimice_tripex', build_lists=True)

t_sta = datetime.datetime(2019, 1, 5, 0, 0, 0)
t_end = datetime.datetime(2019, 1, 5, 9, 59, 59)
#t_end = datetime.datetime(2018, 12, 23, 7, 29, 59)

h_sta = 0
h_end = 9970


# read MDV and cloudnet
MDV = larda.read("MIRA", "VELg", [t_sta, t_end], [h_sta, h_end])
MDV['mask'] = np.logical_or(MDV['mask'], abs(MDV['var']) > 1000)
np.putmask(MDV['var'], MDV['mask'],  np.nan)
MDV_corrected = rf.air_density_correction(MDV)
MDV_corr = functions.h.put_in_container(MDV_corrected, MDV)
category = larda.read("CLOUDNET", "category_bits", [t_sta, t_end],  [h_sta, h_end])
quality = larda.read("CLOUDNET", "quality_bits", [t_sta, t_end],  [h_sta, h_end])
classes = rf.get_target_classification({"quality_bits": quality, "category_bits": category})
cloudnet_class = functions.h.put_in_container(classes, category, name="CLASS")

# 20 min / 100 m grid
# define new height and time indices of new grid
# one time/ height step more in the beginning needed for np.digitize bins

h_new = np.arange(h_sta-1, stop=h_end+100, step=100)
h_new[-1] +=1
t_new = np.arange(functions.h.dt_to_ts(t_sta), stop=functions.h.dt_to_ts(t_end)+(20*60), step=(20*60))
t_new[-1] +=1

h_center = h_new[:-1] + 0.5*np.diff(h_new)
t_center = t_new[:-1] + 0.5*np.diff(t_new)

MDV_coarse = rf.apply_function_timeheight_grid(np.nanmean, MDV_corr, h_new, t_new)
MDV_coarse = functions.h.put_in_container(MDV_coarse, MDV, ts=t_center, rg=h_center,
                                mask=np.isnan(MDV_coarse), var_lims=[-2,2])


cloudnet_regridded = rf.regrid_integer_timeheight(cloudnet_class, h_new, t_new)
cloudnet_interpolated = functions.pyLARDA.Transformations.interpolate2d(cloudnet_class, method='nearest',
                                                        new_time=t_center, new_range=h_center)
cloudnet_regridded = functions.h.put_in_container(cloudnet_regridded, cloudnet_interpolated,
                                        mask=np.isnan(cloudnet_regridded), ts=t_center, rg=h_center)


# rime mass fraction
rime_fraction = of.rimed_mass_fraction_dmitri(MDV_coarse['var'])
rime_fraction = functions.h.put_in_container(rime_fraction, MDV_coarse, var_lims=[0,1], name="rime mass fraction",
                                   mask=~((cloudnet_regridded['var']==4)| (cloudnet_regridded['var']==5)))

fig, ax = functions.pyLARDA.Transformations.plot_timeheight(rime_fraction, title=True, time_diff_jumps=30*60)
fig.savefig(f'../plots/TRIPEX_rmf_{t_sta.strftime("%Y%m%d%H%M")}_{t_end.strftime("%Y%m%d%H%M")}.png')

