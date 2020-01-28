
import sys
sys.path.append('../')
import functions
import datetime
import functions.radar_functions as rf
import functions.other_functions as of
import numpy as np
from scipy.io import loadmat
import numpy as np
import matplotlib
import datetime

PIP_data = loadmat('/home/tvogl/PhD/conferences_workshops/'
               '201909_Bonn_Juelich/Exercises/Moisseev/Snow_retr_2014_2015_for_Python.mat',
               mat_dtype=True)

t_sta = datetime.datetime(2014, 2, 21, 17, 5, 0)
t_end = datetime.datetime(2014, 2, 22, 3, 0, 0)


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

rmf_container = {'var' : rmf, 'ts' : timestamp[0,:], 'name' : 'rime mass fraction', 'dimlabel': ['time'],
                 'mask' : np.isnan(rmf), 'var_lims' : [0, 1], 'system': 'PIP', 'var_unit' : 'unitless'}

fig, ax = functions.pyLARDA.Transformations.plot_timeseries(rmf_container, time_interval=[t_sta, t_end],
                                                            time_diff_jumps=3600)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 1)))

fig.savefig(f'../plots/BAECC_insitu_timeseries_rmf.png')

