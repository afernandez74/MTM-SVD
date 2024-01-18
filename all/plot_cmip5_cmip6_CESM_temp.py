#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 06:17:28 2023

@author: afer
"""

# %% import functions and packages

import xarray as xr
from os import listdir
import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pkl
import xrft.detrend as detrend

#%% load in CESM past1000 compiled data
# file path where dictionary was saved in mtm_preprocessing.py script
file_path = os.path.expanduser("~/mtm_local/CESM_LME_data_dic/")
# obtain name of file (only works if one file is present only)
file = listdir(file_path)[0]
# save dictionary to dictionary variable
with open(file_path + file, 'rb') as f:
    CESM_LME = pkl.load(f)

#%% load in CMIP5 past1000 compiled data
file_path = os.path.expanduser("~/mtm_local/cmip5_past1000/new_analysis/tas_dic/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000_cmip5 = pkl.load(f)

del file_path, files, f

# =============================================================================
# open ensemble mean of all past1000 datasets
# =============================================================================
file_path = os.path.expanduser("~/mtm_local/cmip5_past1000/new_analysis/tas_dic_regrid_ensemb_mean/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000_cmip5_ensemb_mean = pkl.load(f)
    
del file_path, files, f

# =============================================================================
# open unforced dictionary
# =============================================================================
file_path = os.path.expanduser("~/mtm_local/cmip5_past1000/new_analysis/tas_dic_regrid_unforced/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000_cmpi5_unforced = pkl.load(f)
    
del file_path, files, f



#%% load in CMIP6 past1000 compiled data
file_path = os.path.expanduser("~/mtm_local/past1000/tas_dic/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000_cmip6 = pkl.load(f)

del file_path, files, f

# =============================================================================
# open ensemble mean of all past1000 datasets
# =============================================================================
file_path = os.path.expanduser("~/mtm_local/past1000/tas_dic_regrid_ensemb_mean/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000_cmip6_ensemb_mean = pkl.load(f)
    
del file_path, files, f

# =============================================================================
# open unforced dictionary
# =============================================================================
file_path = os.path.expanduser("~/mtm_local/past1000/tas_dic_regrid_unforced/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000_cmpi6_unforced = pkl.load(f)
    
del file_path, files, f

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
years = past1000_cmip5['CCSM4']['year'].values

ds_i = past1000_cmip5['CCSM4']
temp_cmip5_past1000 = np.zeros((len(past1000_cmip5),ds_i.tas.mean(dim=['lat','lon']).shape[0]))

c=0
for key, ds_i in past1000_cmip5.items():
    if key != 'FGOALS-gl':
        temp_cmip5_past1000[c,:] = ds_i.tas.mean(dim=['lat','lon'])
        c=c+1

ds_i = past1000_cmip6['CESM']
temp_cmip6_past1000 = np.zeros((len(past1000_cmip6),ds_i.tas.mean(dim=['lat','lon']).shape[0]))
c=0
for key, ds_i in past1000_cmip6.items():
    if key != 'INM-CM4-8':
        temp_cmip6_past1000[c,:] = ds_i.tas.mean(dim=['lat','lon'])
        c=c+1

c=0
temp_CESM_LME = np.zeros((len(CESM_LME),999))
for key, ds_i in CESM_LME.items():
    temp_CESM_LME[c,:] = np.nanmean(ds_i.tas, axis =(2,1))[0:999]
    c=c+1

temp_past1000 = np.vstack((temp_CESM_LME,temp_cmip5_past1000,temp_cmip6_past1000))
temp_past1000_mean = np.nanmean(temp_past1000)

temp_past1000_plot = np.zeros_like(temp_past1000)
c = 0
for row in temp_past1000:
    linear_fit = np.polyfit(np.arange(len(row)), row, deg=1)
    detrended_series = row - np.polyval(linear_fit, np.arange(len(row)))
    temp_past1000_plot[c,:] = detrended_series#- temp_past1000_mean
    c=c+1


# plot 
fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

[plt.plot(years,temp_past1000_plot[c,:], linewidth = 0.5, alpha = 0.8) for c in range(temp_past1000_plot.shape[0])]
plt.plot(years, np.mean(temp_past1000_plot, axis = 0), linewidth = 1.5, color = 'black')
ax.grid(True,which='major',axis='both')
plt.title("Temperature anomalies Last Millennium Simulations" )
# plt.ylim([-264,-260.5])
plt.xlabel('Year AD')
plt.ylabel('Temperature (k)')
plt.legend()
save_path = os.path.expanduser('~/mtm_local/AGU23_figs/past1000_temp_fig')
plt.savefig(save_path, format = 'svg')
plt.show()
