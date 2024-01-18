
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
import xrft.detrend as detrend

#%% load in CMIP5 past1000 compiled data
path = os.path.expanduser("~/mtm_local/cmip5_past1000/new_analysis/tas_dic/")
files = listdir(path)
files = [entry for entry in files if not entry.startswith('.')]

file = files[0] #select results file to unplickle

with open(path+file, 'rb') as f:
    past1000_dic = pkl.load(f)
del path, files, file

path = os.path.expanduser("~/mtm_local/cmip5_past1000/new_analysis/tas_dic_regrid_ensemb_mean/")
files = listdir(path)
files = [entry for entry in files if not entry.startswith('.')]
file = files [0]

with open(path +file, 'rb') as f:
    past1000_ensemb_mean = pkl.load(f)
    
del file, files, path, f

#%% load in results Mann data

#mtm-svd results for CMIP5 past1000 experiments (Mann et al., 2021 data)
path = os.path.expanduser("~/mtm_local/cmip5_past1000/Mann_analysis/mtm_svd_results_ALL/")
files = listdir(path)
files = [entry for entry in files if not entry.startswith('.')]

file = files[0] #select results file to unplickle

with open(path + file, 'rb') as f:
    cmip5_past1000_results = pkl.load(f) 
    
# load confidence intervals from Mann analysis 
ci = cmip5_past1000_results['conflevels_adjusted']

del files, file, f

#%% load results from own data compilation

#mtm-svd results for CMIP5 past1000 experiments (own data)
path = os.path.expanduser("~/mtm_local/cmip5_past1000/new_analysis/mtm_svd_results/")
files = listdir(path)
files = [entry for entry in files if not entry.startswith('.')]


file = files[0] #select results file to unplickle

with open(path + file, 'rb') as f:
    cmip5_past1000_results_new = pkl.load(f) 
    
del files, file, f, path


#%% load in results for unforced analysis
path = os.path.expanduser("~/mtm_local/cmip5_past1000/new_analysis/mtm_svd_unforced_results/")
files = listdir(path)
files = [entry for entry in files if not entry.startswith('.')]


file = files[0] #select results file to unplickle

with open(path + file, 'rb') as f:
    cmip5_past1000_unforced_results = pkl.load(f) 
    
del files, file, f, path

path = os.path.expanduser("~/mtm_local/cmip5_past1000/new_analysis/mtm_svd_unforced_results_oneGISS/")
files = listdir(path)
files = [entry for entry in files if not entry.startswith('.')]


file = files[0] #select results file to unplickle

with open(path + file, 'rb') as f:
    cmip5_past1000_unforced_results_oneGISS = pkl.load(f) 
    
del files, file, f, path

#%% plotting parameters
# modify global setting
SMALL_SIZE = 15
MEDIUM_SIZE = 30
BIGGER_SIZE = 50

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

del BIGGER_SIZE, MEDIUM_SIZE, SMALL_SIZE
 
#%% plot global mean temperature anomalies

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

for key, ds_i in past1000_dic.items():
    detrend(ds_i.mean(dim=['lat','lon']).tas - (ds_i.mean(dim=['lat','lon','year']).tas),dim = 'year',detrend_type = 'linear').plot(label=key, linewidth = 1, alpha = 0.8)
    
# (past1000_ensemb_mean.mean(dim=['lat','lon']).tas - past1000_ensemb_mean.mean(dim=['lat','lon','year']).tas).plot(label='ensemble mean', color = 'black', linewidth=1.5)
ax.grid(True,which='major',axis='both')

# plot mean
detrend((past1000_ensemb_mean.mean(dim=['lat','lon']).tas - past1000_ensemb_mean.mean(dim=['lat','lon','year']).tas),dim = 'year', detrend_type = 'linear').plot(color = 'black',label='ensemble mean', linewidth = 2, alpha =1, zorder = 100)


plt.title("Detrended temperature anomalies CMIP5 past1000" )
plt.legend()

#%% process LFV results for plotting 

data_dic = cmip5_past1000_results_new
lfv_ref = data_dic['GISS-E2-R_p121_lfv']
freq = data_dic['GISS-E2-R_p121_freq']

fr_sec = 2/(1000*1) # secular frequency value (nw/N/dt)
fr_sec_ix = np.where(freq < fr_sec)[0][-1] #index in the freq array where the secular frequency is located
mean_ref_lfv = np.nanmean(lfv_ref[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 

# adjust past1000 lfv values
adj_data_dic={}
for key, value in data_dic.items():
    if key.endswith('lfv'):
        mean_i = np.mean(data_dic[key][fr_sec_ix:])
        adj_fac = mean_ref_lfv / mean_i
        adj_data_dic[key] = data_dic[key] * adj_fac
    else:
        adj_data_dic[key] = data_dic[key]


# list of models in results
models = [keys.replace('_lfv', '') for keys in data_dic.keys()]
models = [name for name in models if not name.endswith('freq')]
models_no_GISS = [name for name in models if not name.startswith('GISS')]

models_GISS = [name for name in models if name.startswith('GISS')]
models_GISS_NV = ['GISS_E2_R_p123','GISS_E2_R_p126']#GISS simulations with no volcanic forcing
models_GRA = ['bcc-csm1-1',
              'CCSM4_lfv',
              'GISS-E2-R_p1221',
              'GISS-E2-R_p122',
              'GISS-E2-R_p125',
              'GISS-E2-R_p128',
              'IPSL-CM5A-LR',
              'MRI-CGCM3']
models_CEA = ['CSIRO-Mk3L-1-2'
              'FGOALS-gl',
              'GISS-E2-R_p121',
              'GISS-E2-R_p123',
              'GISS-E2-R_p124',
              'GISS-E2-R_p126',
              'GISS-E2-R_p127',
              'HadCM3',
              'MIROC-ESM',
              'MPI-ESM-P']
models_oneGISS = ['bcc-csm1-1',
              'CCSM4_lfv',
              'GISS-E2-R_p121',
              'IPSL-CM5A-LR',
              'MRI-CGCM3',
              'CSIRO-Mk3L-1-2',
              'FGOALS-gl',
              'HadCM3',
              'MIROC-ESM',
              'MPI-ESM-P']


# create LFV matrices for plotting
lfv = np.zeros((int(len(adj_data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic.items():
    if key.endswith('_lfv'):
        lfv[i,:] = array
        i=i+1

# matrix with all but GISS results for ease of plotting
lfv_no_GISS = np.zeros((int(len(adj_data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic.items():
    if key.endswith('_lfv') and not key.startswith('GISS'):
        lfv_no_GISS[i,:] = array
        i=i+1
lfv_no_GISS = lfv_no_GISS[~np.all(lfv_no_GISS == 0, axis=1)]

# matrix with only GISS results for ease of plotting
lfv_GISS = np.zeros((int(len(adj_data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic.items():
    if key.endswith('_lfv') and  key.startswith('GISS'):
        lfv_GISS[i,:] = array
        i=i+1
lfv_GISS = lfv_GISS[~np.all(lfv_GISS == 0, axis=1)]

# GISS runs with no volcanic forcing (NV)
lfv_GISS_NV = np.zeros((int(len(adj_data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic.items():
    if key.endswith('_lfv') and (key.startswith('GISS-E2-R_p123') or key.startswith('GISS-E2-R_p126')):
        lfv_GISS_NV[i,:] = array
        i=i+1
        print(key)
lfv_GISS_NV = lfv_GISS_NV[~np.all(lfv_GISS_NV == 0, axis=1)]

# all models that use GRA Volcanic forcing timeseries (Gao et al., 2008)
lfv_GRA = np.zeros((int(len(adj_data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic.items():
    if key.endswith('_lfv') and (
            key.startswith('GISS-E2-R_p122')
            or key.startswith('GISS-E2-R_p125')
            or key.startswith('GISS-E2-R_p128')
            or key.startswith('bcc')
            or key.startswith('CCSM')
            or key.startswith('IPSL')
            or key.startswith('MRI')):
        lfv_GRA[i,:] = array
        i=i+1
        print('models with GRA volcanic forcing:')
        print(key)
lfv_GRA = lfv_GRA[~np.all(lfv_GRA == 0, axis=1)]

# all models that use CEA Volcanic forcing timeseries (Crowley et al., 2008)
lfv_CEA = np.zeros((int(len(adj_data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic.items():
    if key.endswith('_lfv') and not(
            key.startswith('GISS-E2-R_p122')
            or key.startswith('GISS-E2-R_p125')
            or key.startswith('GISS-E2-R_p128')
            or key.startswith('bcc')
            or key.startswith('CCSM')
            or key.startswith('IPSL')
            or key.startswith('MRI')):
        lfv_CEA[i,:] = array
        i=i+1
        print('models with CEA or NO volcanic forcing:')
        print(key)
lfv_CEA = lfv_CEA[~np.all(lfv_CEA == 0, axis=1)]

# all models that use CEA Volcanic forcing timeseries (Crowley et al., 2008)
lfv_oneGISS = np.zeros((int(len(adj_data_dic.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic.items():
    if key.endswith('_lfv') and not(
            key.startswith('GISS-E2-R_p122')
            or key.startswith('GISS-E2-R_p125')
            or key.startswith('GISS-E2-R_p128')
            or key.startswith('GISS-E2-R_p122')
            or key.startswith('GISS-E2-R_p123')
            or key.startswith('GISS-E2-R_p124')
            or key.startswith('GISS-E2-R_p126')
            or key.startswith('GISS-E2-R_p127')):

        lfv_oneGISS[i,:] = array
        i=i+1
        print('models only one GISS realization:')
        print(key)
lfv_oneGISS = lfv_oneGISS[~np.all(lfv_oneGISS == 0, axis=1)]




#%% LFV plots
# pick which periods to showcase (years)
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

# # plot lines
# p1 = ax.plot(freq,lfv_ref,
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'darkred')
#         # label = 'GISS_E2_R_1')

# [ax.plot(freq,lfv[i,:],linestyle = '-',linewidth=1,label = models[i]) for i in range(0,len(models))]
# plt.title("MTM-SVD Last Millennium CMIP5")

# [ax.plot(freq,lfv_no_GISS[i,:],linestyle = '-',linewidth=1,label = models_no_GISS[i]) for i in range(0,len(models_no_GISS))]
# plt.title("MTM-SVD Last Millennium NO GISS")

# [ax.plot(freq,lfv_GISS[i,:],linestyle = 'solid',linewidth=1,label = models_GISS[i]) for i in range(0,len(models_GISS))]
# plt.title("MTM-SVD Last Millennium CMIP5 GISS only")

# [ax.plot(freq,lfv_GISS_NV[i,:],linestyle = 'solid',linewidth=1,label = models_GISS_NV[i]) for i in range(0,len(models_GISS_NV))]
# plt.title("MTM-SVD Last Millennium CMIP5 GISS NV")

# [ax.plot(freq,lfv_GRA[i,:],linestyle = 'solid',linewidth=1,label = models_GRA[i]) for i in range(0,len(models_GRA))]
# plt.title("MTM-SVD Last Millennium CMIP5 GRA volc")

# [ax.plot(freq,lfv_CEA[i,:],linestyle = 'solid',linewidth=1,label = models_CEA[i]) for i in range(0,len(models_CEA))]
# plt.title("MTM-SVD Last Millennium CMIP5 CEA or NO volc")

# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]

ax.legend()


#%% mean and std for volc and no volc 
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
lfv_GRA_mean = np.mean(lfv_GRA,axis=0)
lfv_GRA_std = np.std(lfv_GRA,axis=0)

lfv_CEA_mean = np.mean(lfv_CEA,axis = 0)
lfv_CEA_std = np.std(lfv_CEA,axis = 0)

lfv_GISS_NV_mean = np.mean(lfv_GISS_NV,axis=0)
lfv_GISS_NV_std = np.std(lfv_GISS_NV,axis=0)

# plot lines
p1 = ax.plot(freq,lfv_GRA_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkred',
        label = 'GRA forcing')

p2 = ax.plot(freq,lfv_CEA_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkblue',
        label = 'CEA forcing')

p3 = ax.plot(freq,lfv_GISS_NV_mean,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'green',
        label = 'No Volc')

# plot +1sd and -1sd shaded areas
p4 = ax.fill_between(freq, lfv_GRA_mean+lfv_GRA_std, lfv_GRA_mean-lfv_GRA_std,
                alpha=.2, linewidth=0, zorder = 2, color = 'darkred',
                label = 'GRA \u00B1 \u03C3')

p5 = ax.fill_between(freq, lfv_CEA_mean+lfv_CEA_std, lfv_CEA_mean-lfv_CEA_std,
                alpha=.2, linewidth=0, zorder = 1, color = 'darkblue',
                label = 'CEA \u00B1 \u03C3')

p6 = ax.fill_between(freq, lfv_GISS_NV_mean+lfv_GISS_NV_std, lfv_GISS_NV_mean-lfv_GISS_NV_std,
                alpha=.2, linewidth=0, zorder = 1, color = 'green',
                label = 'No Volc \u00B1 \u03C3')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.legend()
plt.title('CMIP5 past1000 Volc comparison')
save_path = os.path.expanduser('~/mtm_local/AGU23_figs/cmip5_past1000_lfv_volc_fig')
plt.savefig(save_path, format = 'svg')

#%% forced and no forced
data_dic = cmip5_past1000_unforced_results
lfv_ref = data_dic['GISS-E2-R_p121_lfv']
freq = data_dic['GISS-E2-R_p121_freq']

fr_sec = 2/(1000*1) # secular frequency value (nw/N/dt)
fr_sec_ix = np.where(freq < fr_sec)[0][-1] #index in the freq array where the secular frequency is located
mean_ref_lfv = np.nanmean(lfv_ref[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 

# adjust past1000 lfv values
adj_data_dic_unforced={}
for key, value in data_dic.items():
    if key.endswith('lfv'):
        mean_i = np.mean(data_dic[key][fr_sec_ix:])
        adj_fac = mean_ref_lfv / mean_i
        adj_data_dic_unforced[key] = data_dic[key] * adj_fac
    else:
        adj_data_dic_unforced[key] = data_dic[key]




# create LFV matrices for plotting
lfv_unforced = np.zeros((int(len(adj_data_dic_unforced.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic_unforced.items():
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
plt.title('CMIP5 past1000 forced vs internal')
save_path = os.path.expanduser('~/mtm_local/AGU23_figs/cmip5_past1000_lfv_f_vs_int_fig')
plt.savefig(save_path, format = 'svg')

#%% forced and no forced ONE GISS
data_dic = cmip5_past1000_unforced_results_oneGISS
lfv_ref = data_dic['GISS-E2-R_p125_lfv']
freq = data_dic['GISS-E2-R_p125_freq']

fr_sec = 2/(1000*1) # secular frequency value (nw/N/dt)
fr_sec_ix = np.where(freq < fr_sec)[0][-1] #index in the freq array where the secular frequency is located
mean_ref_lfv = np.nanmean(lfv_ref[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 

# adjust past1000 lfv values
adj_data_dic_unforced={}
for key, value in data_dic.items():
    if key.endswith('lfv'):
        mean_i = np.mean(data_dic[key][fr_sec_ix:])
        adj_fac = mean_ref_lfv / mean_i
        adj_data_dic_unforced[key] = data_dic[key] * adj_fac
    else:
        adj_data_dic_unforced[key] = data_dic[key]

# create LFV matrices for plotting
lfv_unforced_oneGISS = np.zeros((int(len(adj_data_dic_unforced.keys())/2), int(freq.shape[0])))
i=0
for key, array in adj_data_dic_unforced.items():
    if key.endswith('_lfv'):
        lfv_unforced_oneGISS[i,:] = array
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
lfv_mean = np.mean(lfv_oneGISS,axis=0)
lfv_std = np.std(lfv_oneGISS,axis=0)

lfv_unforced_mean = np.mean(lfv_unforced_oneGISS,axis = 0)
lfv_unforced_std = np.std(lfv_unforced_oneGISS,axis = 0)

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
plt.title('CMIP5 past1000 forced vs internal')
save_path = os.path.expanduser('~/mtm_local/AGU23_figs/cmip5_past1000_lfv_f_vs_int_fig')
plt.savefig(save_path, format = 'svg')
