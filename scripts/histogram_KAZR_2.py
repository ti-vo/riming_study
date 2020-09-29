import sys
sys.path.append('../')
import functions
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import functions.radar_functions as rf
import numpy as np

larda = functions.pyLARDA.LARDA().connect('arm_baecc', build_lists=True)
import datetime

t_sta = datetime.datetime(2014, 2, 22, 3, 5)
t_end = datetime.datetime(2014, 2, 22, 3, 10)
Ze = larda.read("KAZR", "Ze_spec2mom", [t_sta, t_end], [300, 500])

# 1 ) reading in in-situ data
#in_situ_data = '/home/tvogl/PhD/conferences_workshops/201909_Bonn_Juelich/Exercises/Moisseev/Snow_retr_2014_2015_for_Python.mat'
in_situ_data = '/media/sdig/arm-iop/2014/baecc/in-situ/Snow_retr_2014_2015_for_Python.mat'
PIP_data = loadmat(in_situ_data, mat_dtype=True)
rmf_pip, piptime = rf.rimed_mass_fraction_PIP(PIP_data)

height = 400
# 2) identify time chunks in in-situ data
jumps = np.where(np.diff(piptime) > 3600)[0]
i = 0
while piptime[jumps[i]] < functions.h.dt_to_ts(datetime.datetime(2014, 3, 31, 0, 0)):
    if i == 0:
        t_sta = datetime.datetime(2014, 2, 1, 2, 0)
    else:
        t_sta = functions.h.ts_to_dt(piptime[jumps[i - 1] + 1])
    t_end = functions.h.ts_to_dt(piptime[jumps[i]])
    i += 1

    # 3) read in the data for all of the chunks
    print(f'time interval from {t_sta.strftime("%Y%m%d %H:%M")} to {t_end.strftime("%Y%m%d %H:%M")} \n')

    # read in radar data (for now only moments)
    Zg = larda.read("KAZR", "Ze_spec2mom", [t_sta, t_end], [height])
    MDV = larda.read("KAZR", "mdv_spec2mom", [t_sta, t_end], [height])
    sw = larda.read("KAZR", "sw_spec2mom", [t_sta, t_end], [height])
    skew = larda.read("KAZR", "skew_spec2mom", [t_sta, t_end], [height])
    # read in signal-to-noise ratio for masking non-cloudy pixels
    #snr = rf.remove_broken_timestamp_arm(larda.read("KAZR", "snr", [t_sta, t_end], [height]))

    if i == 1:
        Zg_all = Zg
        MDV_all = MDV
        sw_all = sw
    else:
        Zg_all = functions.pyLARDA.Transformations.join(Zg_all, Zg)
        MDV_all = functions.pyLARDA.Transformations.join(MDV_all, MDV)
        sw_all = functions.pyLARDA.Transformations.join(sw_all, sw)

plot_dir = '../plots/BAECC_pam_spec2mom/'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

fig, ax = plt.hist(Zg_all)
ax.set_xlabel('Reflectivity')
fig.savefig(plot_dir + 'histogram_Zg.png')







