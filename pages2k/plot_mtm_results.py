#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os 
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import numpy as np

#%% load data
path = os.path.expanduser('~/mtm_local/pages2k/mtm_results/')
files = os.listdir(path)

with open('/Users/afer/mtm_local/pages2k/mtm_results/pages2k_past1000_mtm_results', 'rb') as f:
    data= pickle.load(f)
    
#%% load in results Mann data

#mtm-svd results for CMIP5 past1000 experiments (Mann et al., 2021 data)
path = os.path.expanduser("~/mtm_local/pages2k/mtm_results/pages2k_past1000_mtm_results_ci")

with open(path, 'rb') as f:
    ci = pickle.load(f) 
    
#%%
freq = data['ALL_freq']
lfv_ref = data['ALL_lfv']

#lfv matrix for plotting
lfv = np.zeros((int(len(data.keys())/2), int(freq.shape[0])))
archives = ['' for _ in range(0,int(len(data.keys())/2))]

fr_sec = 2/(1000*1) # secular frequency value (nw/N/dt)
fr_sec_ix = np.where(freq < fr_sec)[0][-1] #index in the freq array where the secular frequency is located
mean_ref_lfv = np.nanmean(lfv_ref[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 

#adjust lfv data to same mean

i=0
for key, array in data.items():
    if key.endswith('_lfv'):
        mean_i = np.nanmean(array[fr_sec_ix:])
        adj_fac = mean_ref_lfv / mean_i
        adj_lfv = array * adj_fac

        lfv[i,:] = adj_lfv
        archives[i] = key[0:key.find('_lfv')]
        i=i+1


# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

# pick which periods to showcase (years)
xticks = [100,60,40,30,20]

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
plt.ylim(0.2,0.8)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
i=0
[plt.plot(freq,lfv[i,:], label = archives[i], linewidth = 1.0, alpha = 0.8) for i in range (1,lfv.shape[0])]
plt.plot(freq,lfv_ref, label = 'All Archives', linewidth=2.0, color='black')

# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]
plt.legend()
plt.xlabel('Period (yr)')
plt.ylabel('LFV')
#save fig
name = os.path.expanduser('~/mtm_local/AGU23_figs/pages2kmtm_fig')
plt.savefig(name, dpi=300, format='svg')


