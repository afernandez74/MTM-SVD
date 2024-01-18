
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
# load values
freq = lme_lfv['freq_ref']
ci = lme_lfv['conflevels_adjusted']

# load frequency and C.I values from CESM analysiw results dictionary
mean_ref_lfv = np.mean(lme_lfv['lfv_ref']) # mean of ref lfv to adjust others from past1000 exp.
fr_sec = 2/(1000*1) # secular frequency value (nw/N/dt)
fr_sec_ix = np.where(freq < fr_sec)[0][-1] #index in the freq array where the secular frequency is located

# adjust past1000 lfv values
adj_past1000_lfv={}
for key, value in past1000_lfv.items():
    if key.endswith('lfv'):
        mean_i = np.mean(past1000_lfv[key][fr_sec_ix:])
        adj_fac = mean_ref_lfv / mean_i
        adj_past1000_lfv[key] = past1000_lfv[key] * adj_fac
    else:
        adj_past1000_lfv[key] = past1000_lfv[key]
        
# adjust past1000 unforced lfv values
adj_past1000_unforced_lfv={}
for key, value in past1000_unforced_lfv.items():
    if key.endswith('lfv'):
        mean_i = np.mean(past1000_unforced_lfv[key][fr_sec_ix:])
        adj_fac = mean_ref_lfv / mean_i
        adj_past1000_unforced_lfv[key] = past1000_unforced_lfv[key] * adj_fac
    else:
        adj_past1000_unforced_lfv[key] = past1000_unforced_lfv[key]
            
        

# %%plot CESM, MIROC and MRI forced and unforced LFV spectra


# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

# pick which periods to showcase (years)
# xticks = [100,60,40,20]
xticks = [10,7,5,3]
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

p5 = ax.plot(freq,adj_past1000_lfv['MRI-ESM2-0_lfv'],
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'red',
        label = 'MRI-ESM2 forced')

p6 = ax.plot(freq,adj_past1000_unforced_lfv['MRI-ESM2-0_lfv'],
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue',
        label = 'MRI-ESM2 unforced')

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

# p9 = ax.plot(freq,adj_past1000_lfv['INM-CM4-8_lfv'],
#         linestyle = '--',
#         linewidth=2,
#         zorder = 10,
#         color = 'purple',
#         label = 'INM-CM4-8 forced')

# p10 = ax.plot(freq,adj_past1000_unforced_lfv['INM-CM4-8_lfv'],
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'purple',
#         label = 'INM-CM4-8 unforced')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]
save_name = os.path.expanduser('~/mtm_local/AGU23_figs/MRI_spectrum_nino')
ax.legend()
plt.savefig(save_name, format = 'svg')

#%% plot forced unforced shaded area

data_dic = adj_past1000_lfv
# create LFV matrices for plotting
lfv = np.zeros((int(len(data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in data_dic.items():
    if key.endswith('_lfv'):
        lfv[i,:] = array
        i=i+1

data_dic_unforced = adj_past1000_unforced_lfv
# create LFV matrices for plotting
lfv_unforced = np.zeros((int(len(data_dic_unforced.keys())/2), int(freq.shape[0])))
i=0
for key, array in data_dic_unforced.items():
    if key.endswith('_lfv'):
        lfv_unforced[i,:] = array
        i=i+1


xticks = [100,80,60,40,20]
# xticks = [10,7,3]
# 
# figure drawing
fig = plt.figure(figsize=(30,12))
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

# 
lfv_mean = np.mean(lfv,axis=0)
lfv_std = np.std(lfv,axis=0)

lfv_unforced_mean = np.mean(lfv_unforced,axis = 0)
lfv_unforced_std = np.std(lfv_unforced,axis = 0)

# plot lines
p1 = ax.plot(freq,lfv_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkred',
        label = 'CMIP5 past1000')

p2 = ax.plot(freq,lfv_unforced_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkblue',
        label = 'CMIP5 past1000 internal')


# plot +1sd and -1sd shaded areas
p4 = ax.fill_between(freq, lfv_mean+lfv_std, lfv_mean-lfv_std,
                alpha=.2, linewidth=0, zorder = 2, color = 'darkred',
                label = 'CMIP5 past1000 \u00B1 \u03C3')

p5 = ax.fill_between(freq, lfv_unforced_mean+lfv_unforced_std, lfv_unforced_mean-lfv_unforced_std,
                alpha=.2, linewidth=0, zorder = 1, color = 'darkblue',
                label = 'CMIP5 past1000 internal \u00B1 \u03C3')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.legend()
plt.title('CMIP6 past1000 forced vs internal')
save_path = os.path.expanduser('~/mtm_local/AGU23_figs/cmip6_past1000_lfv_f_vs_int_fig')
plt.savefig(save_path, format = 'svg')

#%% plot forced unforced shaded area (NO CESM)

data_dic = adj_past1000_lfv
# create LFV matrices for plotting
lfv = np.zeros((int(len(data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in data_dic.items():
    if key.endswith('_lfv') and not key.startswith('CESM'):
        lfv[i,:] = array
        i=i+1

lfv = lfv[~(np.all(lfv == 0, axis=1))]

data_dic_unforced = adj_past1000_unforced_lfv
# create LFV matrices for plotting
lfv_unforced = np.zeros((int(len(data_dic_unforced.keys())/2), int(freq.shape[0])))
i=0
for key, array in data_dic_unforced.items():
    if key.endswith('_lfv') and not key.startswith('CESM'):
        lfv_unforced[i,:] = array
        i=i+1
lfv_unforced = lfv_unforced[~(np.all(lfv_unforced == 0, axis=1))]


xticks = [100,80,60,40,20]
# xticks = [10,7,3]
# 
# figure drawing
fig = plt.figure(figsize=(30,12))
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

# 
lfv_mean = np.mean(lfv,axis=0)
lfv_std = np.std(lfv,axis=0)

lfv_unforced_mean = np.mean(lfv_unforced,axis = 0)
lfv_unforced_std = np.std(lfv_unforced,axis = 0)

# plot lines
p1 = ax.plot(freq,lfv_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkred',
        label = 'CMIP5 past1000')

p2 = ax.plot(freq,lfv_unforced_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkblue',
        label = 'CMIP5 past1000 internal')


# plot +1sd and -1sd shaded areas
p4 = ax.fill_between(freq, lfv_mean+lfv_std, lfv_mean-lfv_std,
                alpha=.2, linewidth=0, zorder = 2, color = 'darkred',
                label = 'CMIP5 past1000 \u00B1 \u03C3')

p5 = ax.fill_between(freq, lfv_unforced_mean+lfv_unforced_std, lfv_unforced_mean-lfv_unforced_std,
                alpha=.2, linewidth=0, zorder = 1, color = 'darkblue',
                label = 'CMIP5 past1000 internal \u00B1 \u03C3')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.legend()
plt.title('CMIP6 past1000 forced vs internal')
save_path = os.path.expanduser('~/mtm_local/AGU23_figs/cmip6_past1000_lfv_f_vs_int_NOCESM_fig')
plt.savefig(save_path, format = 'svg')

