
# %% import functions and packages

from mtm_funcs import *
from readin_funcs_CESM_LME import *
import xarray as xr
from os import listdir
import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from datetime import datetime
import pickle as pkl

#%%
# =============================================================================
# load in results and make plots
# =============================================================================
results_path = os.path.expanduser("~/mtm_local/CESM_LME_mtm_svd_results/")
files = listdir(results_path)

file = files[1] #select results file to unplickle

with open(results_path + file, 'rb') as f:
    data = pkl.load(f)
# %%
# =============================================================================
# plot all LFVs (forced and unforced LFV and plus/minus std and C.I)
# =============================================================================

num_iter = np.shape(data['lfv_all'])[1]
freq = data['freq_ref']
ci = data['conflevels_adjusted']
# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

# pick which periods to showcase (years)
xticks = [200,100,60,40,20,10]

# figure drawing
fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

# set x ticks and labels
xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.4,0.8)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# calculate ensemble means for all forcing and internal-only spectra
all_mean = np.mean(data['lfv_all'],axis=1)
all_sd = np.std(data['lfv_all'],axis=1)

inter_mean = np.mean(data['lfv_inter'],axis = 1)
inter_sd = np.std(data['lfv_inter'],axis = 1)
 
# plot lines
p1 = ax.plot(freq,all_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkred',
        label = 'Ensemble mean - Forced')

p2 = ax.plot(freq,inter_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkblue',
        label = 'Ensemble mean - Unforced')
# plot +1sd and -1sd shaded areas
p3 = ax.fill_between(freq, all_mean+all_sd, all_mean-all_sd,
                alpha=.2, linewidth=0, zorder = 2, color = 'darkred',
                label = 'Forced \u00B1 \u03C3')
p4 = ax.fill_between(freq, inter_mean+inter_sd, inter_mean-inter_sd,
                alpha=.2, linewidth=0, zorder = 1, color = 'darkblue',
                label = 'Unforced \u00B1 \u03C3')

# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]

ax.legend()