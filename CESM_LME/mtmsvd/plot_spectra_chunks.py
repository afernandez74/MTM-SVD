#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:25:53 2024

@author: afer
"""


# %% import functions and packages

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#%% load data

path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv_chunk/')
files = os.listdir(path)

# =============================================================================
# file names
# =============================================================================
lfv_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.endswith('forc.nc')]
lfv_file.sort()
lfv_unforced_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.endswith('unforced.nc')]
lfv_unforced_file.sort()
# lfv_obs_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('HadCRUT5')]

ci_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('conf_int')]
# ci_obs_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('conf_int_HadCRUT')]

# =============================================================================
# load .nc files with xr
# =============================================================================
lfv = []
lfv.append([(xr.open_dataset(path+lfv_file[i])) for i in range (0,len(lfv_file))])
lfv = lfv[0]
lfv_unforced = []
lfv_unforced.append([(xr.open_dataset(path+lfv_unforced_file[i])) for i in range (0,len(lfv_unforced_file))])
lfv_unforced = lfv_unforced[0]
# lfv_obs = xr.open_dataset(path+lfv_obs_file[0])['HadCRUT5_lfv']
ci = np.load(path+ci_file[0])
# ci_obs = np.load(path+ci_obs_file[0])

del path, lfv_file, ci_file, lfv_unforced_file
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

#%%plot lfv mean spectra by case 

case = 'ALL'
unforced = True
whole_spec = False

# pick which periods to showcase (years)
xticks = [100,80,70,60,50,40,30,20]
if whole_spec:
    xticks = [100,80,70,60,50,40,30,20,10,5,3]

# figure drawing

fig = plt.figure(figsize=(25,10))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')
# plt.yscale('log')
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
plt.minorticks_off()


# create colormap for lines
# n_lines=len(lfv)
# if case == 'ALL':
#     colormap = plt.cm.rainbow
# elif case == 'VOLC':
#     colormap = plt.cm.Reds

# colors = colormap(np.linspace(0, 1, n_lines))
# ax.set_prop_cycle('color', colors)

dat = lfv 
if unforced:
    dat = lfv_unforced

for chunk_ds in dat:
    chunk_ds_case = {var_name: chunk_ds[var_name] for var_name 
               in chunk_ds.data_vars if var_name.startswith(case)}
    case_ds = xr.concat(chunk_ds_case.values(), dim = 'run')
    case_mean = case_ds.mean(dim = 'run')
    
    
    l = case_mean.plot(
            ax=ax,
            linestyle = '-',
            linewidth=2,
            zorder = 10,
            # color = 'blue',
            label = chunk_ds.attrs['period'])

        
        
        
# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title(f'MTM-SVD CESM LME Chunks {case}')
if unforced:
    plt.title(f'MTM-SVD CESM LME Chunks {case} unforced')
ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/GLOB/')
name = 'lfv_forced_ALL_VOLC_ORB_SOL'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%%specific chunks
periods = []
for tempy in range(len(lfv)):
    periods.append(lfv_unforced[tempy].period)

print(periods)

case = 'ALL'
unforced = False
whole_spec = False

# pick which periods to showcase (years)
xticks = [100,80,70,60,50,40,30,20]
if whole_spec:
    xticks = [100,80,70,60,50,40,30,20,10,5,3]

# figure drawing

fig = plt.figure(figsize=(25,10))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')
# plt.yscale('log')
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
plt.minorticks_off()


# create colormap for lines
# n_lines=len(lfv)

# if case == 'ALL':
#     colormap = plt.cm.rainbow
# elif case == 'VOLC':
#     colormap = plt.cm.Reds

# colors = colormap(np.linspace(0, 1, n_lines))
# ax.set_prop_cycle('color', colors)

plot_periods = [2,5]

dat = [lfv[per] for per in plot_periods] 
if unforced:
    dat = [lfv_unforced[per] for per in plot_periods]

for chunk_ds in dat:
    chunk_ds_case = {var_name: chunk_ds[var_name] for var_name 
               in chunk_ds.data_vars if var_name.startswith(case)}
    case_ds = xr.concat(chunk_ds_case.values(), dim = 'run')
    case_mean = case_ds.mean(dim = 'run')
    
    
    l = case_mean.plot(
            ax=ax,
            linestyle = '-',
            linewidth=2,
            zorder = 10,
            # color = 'blue',
            label = chunk_ds.attrs['period'])

        
        
        
# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title(f'MTM-SVD CESM LME Chunks {case}')
if unforced:
    plt.title(f'MTM-SVD CESM LME Chunks {case} unforced')
ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/GLOB/')
name = 'lfv_forced_ALL_VOLC_ORB_SOL'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
