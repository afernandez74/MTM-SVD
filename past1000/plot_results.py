
# %% import functions and packages

from mtm_funcs import *
import xarray as xr
from os import listdir
import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
from datetime import datetime
import pickle as pkl

#%% load in results 

results_past1000_path = os.path.expanduser("~/mtm_local/past1000/mtm_svd_results/")
files = listdir(results_past1000_path)
files = [entry for entry in files if not entry.startswith('.')]

file = files[0] #select results file to unplickle

with open(results_past1000_path + file, 'rb') as f:
    past1000_lfv = pkl.load(f) 
    
del files, file, f, results_past1000_path

results_CESM_path = os.path.expanduser("~/mtm_local/CESM_LME_mtm_svd_results/")
files = listdir(results_CESM_path)
files = [entry for entry in files if not entry.startswith('.')]
file = files[0] #select results file to unplickle

with open(results_CESM_path + file, 'rb') as f:
    lme_lfv = pkl.load(f)
    
del files, file, f, results_CESM_path

results_past1000_unforced_path = os.path.expanduser("~/mtm_local/past1000/mtm_svd_unforced_results/")
files = listdir(results_past1000_unforced_path)
files = [entry for entry in files if not entry.startswith('.')]
file = files[0] #select results file to unplickle

with open(results_past1000_unforced_path + file, 'rb') as f:
    past1000_unforced_lfv = pkl.load(f)
    
del files, file, f,results_past1000_unforced_path

# %% adjust CMIP6 past1000 lfv spectra to CESM ref mean value

# load frequency and C.I values from CESM analysiw results dictionary
mean_ref_lfv = np.mean(lme_lfv['lfv_ref']) # mean of ref lfv to adjust others from past1000 exp.

# adjust past1000 lfv values
adj_past1000_lfv={}
for key, value in past1000_lfv.items():
    if key.endswith('lfv'):
        mean_i = np.mean(past1000_lfv[key])
        adj_fac = mean_ref_lfv / mean_i
        adj_past1000_lfv[key] = past1000_lfv[key] * adj_fac
    else:
        adj_past1000_lfv[key] = past1000_lfv[key]
        
# adjust past1000 unforced lfv values
adj_past1000_unforced_lfv={}
for key, value in past1000_unforced_lfv.items():
    if key.endswith('lfv'):
        mean_i = np.mean(past1000_unforced_lfv[key])
        adj_fac = mean_ref_lfv / mean_i
        adj_past1000_unforced_lfv[key] = past1000_unforced_lfv[key] * adj_fac
    else:
        adj_past1000_unforced_lfv[key] = past1000_unforced_lfv[key]
            
# %%plot CESM, MIROC and MRI forced and unforced LFV spectra

# load values
freq = lme_lfv['freq_ref']
ci = lme_lfv['conflevels_adjusted']

# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

# pick which periods to showcase (years)
xticks = [100,60,40,20,10]
# xticks = [10,7,3]
# 
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
plt.ylim(0.4,1.0)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
 
# # plot lines
# p1 = ax.plot(freq,lme_lfv['lfv_ref'],
#         linestyle = '--',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkred',
#         label = 'CESM forced')

# # plot lines
# p2 = ax.plot(freq,adj_past1000_unforced_lfv['CESM_lfv'],
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkred',
#         label = 'CESM unforced')

# p3 = ax.plot(freq,past1000_lfv['MIROC-ES2L_lfv'],
#         linestyle = '--',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkblue',
#         label = 'MIROC-ES2L forced')

# p4 = ax.plot(freq,past1000_unforced_lfv['MIROC-ES2L_lfv'],
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkblue',
#         label = 'MIROC-ES2L unforced')

# p5 = ax.plot(freq,adj_past1000_lfv['MRI-ESM2-0_lfv'],
#         linestyle = '--',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkolivegreen',
#         label = 'MRI-ESM2 forced')

# p6 = ax.plot(freq,adj_past1000_unforced_lfv['MRI-ESM2-0_lfv'],
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkolivegreen',
#         label = 'MRI-ESM2 unforced')

# p7 = ax.plot(freq,adj_past1000_lfv['ACCESS-ESM1-5_lfv'],
#         linestyle = '--',
#         linewidth=2,
#         zorder = 10,
#         color = 'orange',
#         label = 'ACCESS-ESM1-5 forced')

# p8 = ax.plot(freq,adj_past1000_unforced_lfv['ACCESS-ESM1-5_lfv'],
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'orange',
#         label = 'ACCESS-ESM1-5 unforced')

p9 = ax.plot(freq,adj_past1000_lfv['INM-CM4-8_lfv'],
        linestyle = '--',
        linewidth=2,
        zorder = 10,
        color = 'purple',
        label = 'INM-CM4-8 forced')

p10 = ax.plot(freq,adj_past1000_unforced_lfv['INM-CM4-8_lfv'],
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'purple',
        label = 'INM-CM4-8 unforced')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]

ax.legend()

# %%plot all LFVs (CMIP6 past 1000 and CESM LME ref simulation)
# SAME AS ABOVE BUT WITHOUT ADJUSTING MODELS TO MEAN


# load values
freq = lme_lfv['freq_ref']
ci = lme_lfv['conflevels_adjusted']

# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

# pick which periods to showcase (years)
xticks = [100,60,40,20,10]
# xticks = [10,7,3]
# 
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
plt.ylim(0.4,1.0)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
 
# plot lines
p1 = ax.plot(freq,lme_lfv['lfv_ref'],
        linestyle = '--',
        linewidth=2,
        zorder = 10,
        color = 'darkred',
        label = 'CESM forced')

# plot lines
p2 = ax.plot(freq,past1000_unforced_lfv['CESM_lfv'],
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkred',
        label = 'CESM unforced')

# p3 = ax.plot(freq,adj_past1000_lfv['MIROC-ES2L_lfv'],
#         linestyle = '--',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkblue',
#         label = 'MIROC-ES2L forced')

# p4 = ax.plot(freq,adj_past1000_unforced_lfv['MIROC-ES2L_lfv'],
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkblue',
#         label = 'MIROC-ES2L unforced')

# p5 = ax.plot(freq,adj_past1000_lfv['MRI-ESM2-0_lfv'],
#         linestyle = '--',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkolivegreen',
#         label = 'MRI-ESM2 forced')

# p6 = ax.plot(freq,adj_past1000_unforced_lfv['MRI-ESM2-0_lfv'],
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkolivegreen',
#         label = 'MRI-ESM2 unforced')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]

ax.legend()
